import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import os
num_pixels = 256
import cv2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


def load_model(model_path, device, n_class=1):
    model = UNet(n_class=n_class)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def resize_image(img):
    image_np = np.array(img)
    top_row = np.mean(image_np[0, :, :])
    bottom_row = np.mean(image_np[-1, :, :])
    '''if top_row < 5 and bottom_row < 5:
        rows = np.where(np.mean(image_np, axis=(1, 2)) > 5)[0]
        if len(rows) > 0:
            if len(rows) < img.width:
                first_row = int((img.height - img.width) / 2)
                last_row = first_row + img.width
                img = img.crop((0, first_row, img.width, last_row))
            else:
                first_row, last_row = rows[0], rows[-1]
                img = img.crop((0, first_row, img.width, last_row))
    else:'''
    delta_w = img.height - img.width
    delta_h = 0
    padding = (delta_w // 2, delta_h, delta_w - (delta_w // 2), delta_h)
    img = ImageOps.expand(img, padding, fill=0)

    return img


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_image_np = np.array(image)
    original_size = image.size
    image = resize_image(image)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)
    return image, original_image_np, original_size


def predict(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).float()
    return preds


def main(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    image, original_image_np, original_size = preprocess_image(image_path)
    mask_pred = predict(model, image, device)

    mask_pred_np = mask_pred.cpu().numpy().squeeze()
    predicted_mask_resized = cv2.resize(mask_pred_np, original_size, interpolation=cv2.INTER_NEAREST)
    predicted_mask_resized = (predicted_mask_resized * 255).astype(np.uint8)

    colored_mask = np.zeros_like(original_image_np)
    colored_mask[predicted_mask_resized > 0] = [0, 255, 0]  # Green mask

    overlayed_image = cv2.addWeighted(original_image_np, 1, colored_mask, 0.5, 0)

    cv2.imshow('Overlayed Prediction', overlayed_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = 'frame_1440.jpg'
    model_path = 'models/UnetSegmentation_ep10_2024-03-19_17-36.pth'
    main(image_path, model_path)




