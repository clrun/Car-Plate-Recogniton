import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from crnn_model import CRNN

# 中文字符集：省份简称 + 大写字母 + 数字
CHARS = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_CLASSES = len(CHARS) + 1  # +1 为CTC空白符
BLANK_IDX = NUM_CLASSES - 1


class PlateDataset(Dataset):
    """
    车牌识别数据集类
    需根据你的数据集格式自行实现 __len__ 和 __getitem__
    返回：(image_tensor, target_label, input_length, target_length)
    """
    def __init__(self, data_dir: str, img_height: int = 32, img_width: int = 160):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        # 此处加载你的数据集路径与标签
        self.samples = []  # 格式：[(图片路径, 车牌字符串), ...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 读取图片并resize -> Tensor
        # 2. 将车牌字符串转换为数字标签序列
        # 3. 返回图片、标签、序列长度
        pass


def train_crnn():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = CRNN(
        img_height=32,
        in_channels=3,
        num_classes=NUM_CLASSES,
        hidden_dim=256
    ).to(device)

    # 损失函数与优化器
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # ========= 数据集加载（替换为你的数据集） =========
    # train_dataset = PlateDataset("datasets/plate_rec/train")
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    train_loader = None  # 占位，替换为真实数据加载器

    if train_loader is None:
        print("请先实现 PlateDataset 并加载数据")
        return

    # 训练循环
    epochs = 50
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [T, B, C]

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "crnn_plate_best.pth")

    # 保存最终模型
    torch.save(model.state_dict(), "crnn_plate_final.pth")
    print(f"训练完成，最优损失: {best_loss:.4f}")


if __name__ == "__main__":
    train_crnn()