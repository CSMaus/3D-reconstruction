import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import numpy as np
import os
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torch import nn
from torchvision import models
from datetime import datetime
import time
from torch.nn.parallel import DataParallel
thresh = 2


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

        # determine if top and bottom of the image can be trimmed
        # i e if they contain almost zero values
        top_row = np.mean(image_np[0, :, :])
        bottom_row = np.mean(image_np[-1, :, :])
        '''if top_row < thresh and bottom_row < thresh:
            rows = np.where(np.mean(image_np, axis=(1, 2)) > thresh)[0]
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
        else:'''
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


def get_deeplabv3_pretrained_model(num_classes):
    model = deeplabv3_resnet101(pretrained=True)  # 50
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


# always use 256. otherwise the predictions would be very bad
num_pixels = 256
time_start = time.time()
print('Start script at: ', time_start)
transform = T.Compose([
    T.Resize((num_pixels, num_pixels)),
    T.ToTensor(),
])
label_type = 'Electrode'  # 'CentralWeld' 'Electrode'
img_dir = f'SegmentationDS/{label_type}/frames/'
mask_dir = f'SegmentationDS/{label_type}/masks/'
print("Number of images: ", len(os.listdir(os.path.join(img_dir))))
dataset = WeldDataset(img_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = 1
model = deeplabv3_resnet101(pretrained=True)  # 50
model.classifier[4] = nn.Conv2d(num_pixels, 1, kernel_size=(1, 1), stride=(1, 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
# Problem with parallel training. Now will use only one GPU
'''if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs")
    model = DataParallel(model)'''
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
loss_history = []
for epoch in range(num_epochs):
    model.train()
    print("Model training epoch:", epoch, "\nTraining...")
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(images)['out']
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    loss_history.append(loss.item())

current_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
model_name = f'models/{label_type}-deeplabv3_resnet101-{current_date}.pth'
torch.save(model.state_dict(), model_name)


filename = f"histories/{label_type}-training_history-deeplabv3_resnet101-{current_date}.txt"
with open(filename, 'w') as f:
    for epoch, loss in enumerate(loss_history, 1):
        f.write(f"Epoch {epoch}, Loss: {loss}\n")

time_end = time.time()
print('End script at: ', time_end)
print('All computations took: ', (time_end - time_start)/60, "min")

print(model_name)
