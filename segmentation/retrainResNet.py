import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


class WeldDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [file for file in os.listdir(img_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('frame', 'mask').replace('.jpg', '.png'))

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if self.transform is not None:
                image = self.transform(image)
                mask = self.transform(mask)
                mask = (mask > 0).type(torch.FloatTensor)

            return image, mask
        except Exception as e:
            print(f"Error loading data for {img_name}: {e}")


transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

img_dir = 'SegmentationDS/frames/'
mask_dir = 'SegmentationDS/masks/'
dataset = WeldDataset(img_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(images)['out']
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'trained_model.pth')
