import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, TensorDataset
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

# Constants
ROOT_DIR = 'satellite_dataset/'
PATCH_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
N_CLASSES = 6

# Helper functions
def load_and_preprocess_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    return image

def create_patches(image, patch_size):
    patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
    return patches.reshape(-1, patch_size, patch_size, 3)

def rgb_to_label(mask):
    label_seg = np.zeros(mask.shape[:2], dtype=np.uint8)
    label_seg[np.all(mask == [60, 16, 152], axis=-1)] = 0  # Building
    label_seg[np.all(mask == [132, 41, 246], axis=-1)] = 1  # Land
    label_seg[np.all(mask == [110, 193, 228], axis=-1)] = 2  # Road
    label_seg[np.all(mask == [254, 221, 58], axis=-1)] = 3  # Vegetation
    label_seg[np.all(mask == [226, 169, 41], axis=-1)] = 4  # Water
    label_seg[np.all(mask == [155, 155, 155], axis=-1)] = 5  # Unlabeled
    return label_seg

# Data loading and preprocessing
def load_data():
    images, masks = [], []
    for path, _, files in os.walk(ROOT_DIR):
        if path.endswith('images'):
            for file in tqdm(files, desc="Loading images"):
                if file.endswith(".jpg"):
                    img_path = os.path.join(path, file)
                    mask_path = os.path.join(path.replace('images', 'masks'), file.replace('.jpg', '.png'))
                    
                    image = load_and_preprocess_image(img_path, PATCH_SIZE)
                    mask = load_and_preprocess_image(mask_path, PATCH_SIZE)
                    
                    image_patches = create_patches(image, PATCH_SIZE)
                    mask_patches = create_patches(mask, PATCH_SIZE)
                    
                    for img_patch, mask_patch in zip(image_patches, mask_patches):
                        images.append(img_patch)
                        masks.append(rgb_to_label(mask_patch))
    
    X = np.array(images)
    y = np.array(masks)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))
        x3 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))
        x4 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))

        x5 = self.avg_pool(x)
        x5 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(x5)))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out

class ImprovedDeepLabv3Plus(nn.Module):
    def __init__(self, n_classes):
        super(ImprovedDeepLabv3Plus, self).__init__()
        resnet = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers (avgpool and fc)
        self.aspp = ASPP(2048, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, outputs, targets):
        loss1 = self.criterion1(outputs, targets)
        loss2 = self.criterion2(outputs, targets)
        return loss1 + loss2

class IoUMetric(smp.utils.metrics.IoU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=1)
        return super().forward(y_pred, y_true)

def inference(model, image):
    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(next(model.parameters()).device)
        output = model(image)
        output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return output

def visualize_results(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask, cmap='nipy_spectral')
    ax[1].set_title("Predicted Mask")
    return fig

if __name__ == "__main__":
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_val = torch.tensor(y_val, dtype=torch.long)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load test data
    X_test, _, y_test, _ = load_data()  # Only unpack the relevant parts
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, optimizer, and metrics
    model = ImprovedDeepLabv3Plus(N_CLASSES)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    loss = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = IoUMetric()

    # Training and validation
    train_epoch = TrainEpoch(model, loss=loss, metrics=[metrics], optimizer=optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_epoch = ValidEpoch(model, loss=loss, metrics=[metrics], device='cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch+1}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

    # Test and visualization
    test_logs = valid_epoch.run(test_loader)
    test_image, test_mask = X_test[0], y_test[0]
    predicted_mask = inference(model, test_image.permute(1, 2, 0).numpy())
    fig = visualize_results(test_image.permute(1, 2, 0).numpy(), predicted_mask)
    plt.show()