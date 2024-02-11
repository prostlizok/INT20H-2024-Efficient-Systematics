# Imports
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model


# Building segmentation model 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.base_layers = list(resnet.children())

        self.inc = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.base_layers[1]
        self.relu = self.base_layers[2]
        self.maxpool = self.base_layers[3]

        self.down1 = self.base_layers[4]
        self.down2 = self.base_layers[5]
        self.down3 = self.base_layers[6]
        self.down4 = self.base_layers[7]

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)


        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv1(torch.cat([F.interpolate(x, size=x4.size()[2:], mode='bilinear', align_corners=False), x4], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=False), x3], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=False), x2], dim=1))

        x = self.up4(x)
        x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x = self.conv4(torch.cat([x, F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=False)], dim=1))  # Адаптувати x1 до розміру x, якщо потрібно

        x = F.interpolate(x, size=[256, 256], mode='bilinear', align_corners=False)
        logits = self.outc(x)

        return logits


class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, f"{os.path.splitext(self.images[idx])[0]}_mask.jpg")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def iou_score(outputs, labels):
    outputs = torch.sigmoid(outputs) > 0.5
    labels = labels > 0.5

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def train_seg(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    iou_sum = 0.0
    total_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iou = iou_score(outputs, masks)
        iou_sum += iou.item()


        print(f"\rEpoch {epoch}/{total_epochs} [{batch_idx}/{total_batches}]", end='')

    avg_loss = running_loss / total_batches
    avg_iou = iou_sum / total_batches
    print()
    return avg_loss, avg_iou


def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    iou_sum = 0.0
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader, start=1):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            iou = iou_score(outputs, masks)
            iou_sum += iou.item()


            print(f"\rValidating Epoch {epoch}/{total_epochs} [{batch_idx}/{total_batches}]", end='')

    avg_loss = running_loss / total_batches
    avg_iou = iou_sum / total_batches
    print()
    return avg_loss, avg_iou


# Building classification model
def custom_model(image_height, image_width, n_classes):
    base_model = DenseNet169(weights='imagenet',
                             include_top=False,
                             input_shape=(image_height, image_width, 3))

    # Freeze the layers except the last 14
    for layer in base_model.layers[:-14]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(8, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
