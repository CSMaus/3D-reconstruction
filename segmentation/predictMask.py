import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
import numpy as np
import cv2
import torchvision
import os

thresh = 2


def resize_image(img):
    image_np = np.array(img)
    top_row = np.mean(image_np[0, :, :])
    bottom_row = np.mean(image_np[-1, :, :])
    if top_row < thresh and bottom_row < thresh:
        rows = np.where(np.mean(image_np, axis=(1, 2)) > thresh)[0]
        if len(rows) > 0:
            if len(rows) < img.width:
                first_row = int((img.height - img.width) / 2)
                last_row = first_row + img.width
                img = img.crop((0, first_row, img.width, last_row))
            else:
                first_row, last_row = rows[0], rows[-1]
                img = img.crop((0, first_row, img.width, last_row))
    else:
        delta_w = img.height - img.width
        delta_h = 0
        padding = (delta_w // 2, delta_h, delta_w - (delta_w // 2), delta_h)
        img = ImageOps.expand(img, padding, fill=0)

    return img


num_pixels = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load('models/Electrode-deeplabv3_resnet101-2024-03-21_14-39.pth'), strict=False)  # , map_location='cpu'
model = model.to(device)
model.eval()  # to predict - evaluation mode

transform = T.Compose([
    T.Resize((num_pixels, num_pixels)),
    T.ToTensor(),
])

imgs_path = "SegmentationDS/Weld_Video_2023-04-20_02-34-42_Camera02.avi.avi/frames/"
imgs = os.listdir(os.path.join(imgs_path))
input_image_path = 'frame_0010.jpg'  # imgs_path+imgs[0]  #
image = Image.open(input_image_path).convert("RGB")
image = resize_image(image)
original_image_np = np.array(image)
original_size = image.size
image = transform(image).unsqueeze(0)
image = image.to(device)


with torch.no_grad():
    output = model(image)['out']
    probs = torch.sigmoid(output)
    predicted_mask = (probs > 0.5).float()

predicted_mask_np = predicted_mask.cpu().numpy().squeeze(0).squeeze(0)
predicted_mask_resized = cv2.resize(predicted_mask_np, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)
predicted_mask_resized = (predicted_mask_resized * 255).astype(np.uint8)

colored_mask = np.zeros_like(original_image_np)
colored_mask[predicted_mask_resized > 0] = [0, 255, 0]

overlayed_image = cv2.addWeighted(original_image_np, 1, colored_mask, 0.5, 0)

cv2.imshow('Overlayed Prediction', overlayed_image)
cv2.waitKey(0)
# cv2.destroyAllWindows()
