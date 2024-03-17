import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1))
model.load_state_dict(torch.load('trained_model.pth'), strict=False)
model = model.to(device)
model.eval()  # evaluation mode

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

input_image_path = 'frame_0004.jpg'
image = Image.open(input_image_path).convert("RGB")
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
