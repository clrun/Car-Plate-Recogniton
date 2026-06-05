import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 车牌字符字典
PLATE_CHARS = ["京","沪","津","渝","冀","晋","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","陕","甘","青","宁","新","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
CHAR_LEN = len(PLATE_CHARS)

# 轻量字符识别CNN
class PlateCharNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.ReLU(),nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*8,512),nn.ReLU(),
            nn.Linear(512,CHAR_LEN)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# 字符推理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,32))
])

def rec_plate_char(img_tensor, model):
    model.eval()
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0))
        idx = torch.argmax(out,dim=1).item()
    return PLATE_CHARS[idx]