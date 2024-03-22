import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import random_split
import os
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime
import time
import torchviz
from torchviz import make_dot
import sys
from torchview import draw_graph
from graphviz import Digraph


class WeldDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [file for file in os.listdir(img_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def square_image_and_mask(image, mask):
        """
        Adjusts the image and mask to make them square: trimming or padding before resizing.
        """
        image_np = np.array(image)
        mask_np = np.array(mask)

        top_row = np.mean(image_np[0, :, :])
        bottom_row = np.mean(image_np[-1, :, :])
        if top_row < 5 and bottom_row < 5:
            rows = np.where(np.mean(image_np, axis=(1, 2)) > 5)[0]
            if len(rows) > 0:
                if len(rows) < image.width:
                    first_row = int((image.height - image.width)/2)
                    last_row = first_row + image.width
                    image = image.crop((0, first_row, image.width, last_row))
                    mask = mask.crop((0, first_row, mask.width, last_row))
                else:
                    first_row, last_row = rows[0], rows[-1]
                    image = image.crop((0, first_row, image.width, last_row))
                    mask = mask.crop((0, first_row, mask.width, last_row))
        else:
            delta_w = image.height - image.width
            delta_h = 0
            padding = (delta_w // 2, delta_h, delta_w - (delta_w // 2), delta_h)
            image = ImageOps.expand(image, padding, fill=0)
            mask = ImageOps.expand(mask, padding, fill=0)

        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('frame', 'mask').replace('.jpg', '.png'))

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            image, mask = self.square_image_and_mask(image, mask)

            if self.transform is not None:
                image = self.transform(image)
                mask = self.transform(mask)
                mask = (mask > 0).type(torch.FloatTensor)

            return image, mask
        except Exception as e:
            print(f"Error loading data for {img_name}: {e}")


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


def print_unet_summary(model):
    print("U-Net Architecture Summary:")
    print("----------------------------")
    print("Input Layer: Accepts images with 3 channels.")
    print("\nEncoder (Downsampling Path):")
    downsampling_layers = [model.dconv_down1, model.dconv_down2, model.dconv_down3, model.dconv_down4]
    for idx, layer in enumerate(downsampling_layers, start=1):
        print(
            f"  Downsample Block {idx}: Double Convolution and Max Pooling (Output Channels: {layer.double_conv[1].num_features})")

    print("\nBottleneck:")
    print(f"  Bottleneck Convolution Block (Output Channels: {model.dconv_down4.double_conv[1].num_features})")

    print("\nDecoder (Upsampling Path):")
    upsampling_layers = [model.dconv_up3, model.dconv_up2, model.dconv_up1]
    for idx, layer in enumerate(upsampling_layers, start=3, reverse=True):
        print(
            f"  Upsample Block {idx}: Double Convolution and Upsample (Concatenated Channels: {layer.double_conv[1].num_features // 2})")

    print("\nOutput Layer:")
    print(f"  Final Convolution (Output Channels: {model.conv_last.out_channels}, representing classes)")


def main():
    time_start = time.time()
    print('Start script at: ', time_start)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_dir = 'SegmentationDS/frames/'
    mask_dir = 'SegmentationDS/masks/'
    dataset = WeldDataset(img_dir, mask_dir, transform=transform)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_class=1).to(device)

    # model_graph = draw_graph(model, input_size=(1, 3, 224, 224), expand_nested=True)
    # model_graph.visual_graph
    print_unet_summary(model)
    sys.exit()
    # images, masks = next(iter(train_loader))
    # images = images.to(device)
    # yhat = model(images)
    # make_dot(yhat, params=dict(list(model.named_parameters()))).render("UNet_torchviz", format="png")
    # sys.exit()


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}')

    current_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
    torch.save(model.state_dict(), f'UnetSegmentation_ep{epoch + 1}_{current_date}.pth')
    print("Final model saved as final_model_epoch_{}.pth".format(epoch + 1))

    time_end = time.time()
    print('End script at: ', time_end)
    print('All computations took: ', time_end - time_start)


if __name__ == '__main__':
    main()
