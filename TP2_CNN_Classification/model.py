import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # shortcut connection 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels: # 차원이 다르면 1×1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))  # conv1 -> bn1 -> relu
        out = self.bn2(self.conv2(out))           # conv2 -> bn2

        out = out + self.shortcut(identity)       # out <- F(x) + x 

        out = self.relu(out)                      # 최종 ReLU

        return out


# ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # conv1 : 7×7, stride 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # output: (64, 112,112)

        # MaxPool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (64, 56,56)

        # ResNet layers
        self.conv2_x = self._make_layer(64, 64, num_blocks=2, stride=1)   # 56×56
        self.conv3_x = self._make_layer(64, 128, num_blocks=2, stride=2)  # 28×28
        self.conv4_x = self._make_layer(128, 256, num_blocks=2, stride=2) # 14×14
        self.conv5_x = self._make_layer(256, 512, num_blocks=2, stride=2) # 7×7

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 512, 1,1)
        self.fc = nn.Linear(512, num_classes)        # (B, 4)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # 첫 블록: stride 적용
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # 나머지 블록: stride=1
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        # conv1
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 64, 112,112)
        x = self.maxpool(x)                      # (B, 64, 56,56)

        # conv2_x
        x = self.conv2_x(x)

        # conv3_x
        x = self.conv3_x(x)

        # conv4_x
        x = self.conv4_x(x)

        # conv5_x
        x = self.conv5_x(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    # generate random tendsor
    x = torch.rand(size=(8, 3, 224, 224))
    print("Input shape:", x.shape)

    # define model
    model = ResNet18(num_classes=4)
    
    # forward pass
    output = model(x)

    print("Output shape:", output.shape)
    print(output)
    print("end of test")
