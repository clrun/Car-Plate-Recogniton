import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    """双向LSTM层"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(x)
        T, B, H = recurrent.size()
        output = self.fc(recurrent.view(T * B, H))
        return output.view(T, B, -1)


class CRNN(nn.Module):
    """
    CRNN 端到端车牌字符识别模型
    结构：CNN特征提取 + BiLSTM序列建模 + CTC损失
    """
    def __init__(
        self,
        img_height: int = 32,
        in_channels: int = 3,
        num_classes: int = 65,
        hidden_dim: int = 256
    ):
        """
        :param img_height: 输入图片高度，必须为16的倍数
        :param in_channels: 输入通道数，RGB为3
        :param num_classes: 字符类别总数（含CTC空白符）
        :param hidden_dim: LSTM隐藏层维度
        """
        super().__init__()
        assert img_height % 16 == 0, "图片高度必须为16的倍数"

        # CNN特征提取网络
        self.cnn = nn.Sequential(
            # Block 1: 32 -> 16
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 16 -> 8
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 8 -> 4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4: 4 -> 2
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5: 2 -> 1
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 双向LSTM序列建模
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_dim, hidden_dim),
            BidirectionalLSTM(hidden_dim, hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] CNN特征提取
        feat = self.cnn(x)
        B, C, H, W = feat.size()
        assert H == 1, "特征图高度必须为1"

        # 转换为序列格式 [W, B, C]
        feat = feat.squeeze(2).permute(2, 0, 1)

        # RNN序列建模
        output = self.rnn(feat)

        # 输出对数概率 [W, B, num_classes]
        return F.log_softmax(output, dim=2)