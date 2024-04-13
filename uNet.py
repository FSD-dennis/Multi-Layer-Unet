from pacKage import *
from loaddata import *

__all__ = ["DoubleConv2d", "Down2d", "Up2d", "BinaryOut", "UShapeNet", "DiceLoss", "CombinedLoss"]

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up2d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BinaryOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UShapeNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = 2
        self.bilinear = bilinear

        self.inputconv = DoubleConv2d(n_channels, 64) 
        self.down1 = Down2d(64, 128) 
        self.down2 = Down2d(128, 256)  
        self.down3 = Down2d(256, 512) 
        self.down4 = Down2d(512, 512) 
        self.up1 = Up2d(1024, 256, self.bilinear) 
        self.up2 = Up2d(512, 128, self.bilinear)   
        self.up3 = Up2d(256, 64, self.bilinear)  
        self.up4 = Up2d(128, 64, self.bilinear)   
        self.outc = BinaryOut(64, 2)          

    def forward(self, x):
        x1 = self.inputconv(x)      # 7,350,350 to 64,350,350
        x2 = self.down1(x1)         # 128,175,175
        x3 = self.down2(x2)         # 256, 87, 87
        x4 = self.down3(x3)         # 512, 43, 43
        bottom = self.down4(x4)         # 1024, 21, 21 
        x_4 = self.up1(bottom, x4)      # 512, 43, 43
        x_3 = self.up2(x_4, x3)     # 256, 87, 87
        x_2 = self.up3(x_3, x2)     # 128,175,175
        x_1 = self.up4(x_2, x1)     # 64,350,350
        binaryclass = self.outc(x_1)  # 2,350,350
        return binaryclass

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)  # Apply softmax to get probabilities
        # One-hot encode targets if not already done
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1, weight_ce=1):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        combined_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss
        return combined_loss
        


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6, gamma=2):
#         super(DiceLoss, self).__init__()
#         self.name = 'NDL'
#         self.smooth = smooth
#         self.gamma = gamma

#     def forward(self, y_true, y_pred):
#         y_true, y_pred = y_true.float(), y_pred.float()
#         numerator = 2 * torch.sum(y_pred * y_true) + self.smooth
#         denominator = torch.sum(y_pred ** self.gamma) + torch.sum(y_true ** self.gamma) + self.smooth
#         result = 1 - numerator / denominator
#         return result

class BinaryConv(nn.Module):
    def __init__(self):
        super().__init__()
        # Define initial convolutional layers for each input
        self.conv11 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # Process first input
        self.conv12 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # Process second input
        
        # Define additional layers to process combined features
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final layer to get to (250, 350, 1). Adjust the kernel size, stride, etc., as necessary
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((250, 350))  # Adjust size to desired output

    def forward(self, x1, x2):
        # Process each input through its convolutional layer
        x1 = F.relu(self.conv11(x1))
        x2 = F.relu(self.conv12(x2))
        
        # Merge the features from both inputs
        x = torch.cat((x1, x2), dim=1)
        
        # Pass through additional layers
        x = F.relu(self.conv3(x))
        
        # Final convolution to get the channel size right
        x = self.final_conv(x)
        
        # Use adaptive pooling to adjust to the exact desired output size
        x = self.adaptive_pool(x)
        return x
