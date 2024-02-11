# Imports
import os
import numpy as np

from models import custom_model, train_seg, validate, UNet, CustomDataset

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.initializers import *
from keras.preprocessing import image, ImageDataGenerator

from sklearn.utils.class_weight import compute_class_weight

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch
from torchvision import transforms


# Base functions
def create_data_generators(directory, image_height, image_width, batch_size):
    """
    Creates data generators for training and validation.

    Parameters:
    - directory: Path to the directory containing the images.
    - image_height: The target height of the images after resizing.
    - image_width: The target width of the images after resizing.
    - BATCH_SIZE: The size of the batches of data.

    Returns:
    - A tuple containing the training and validation data generators.
    """
    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        brightness_range=[0.95, 1.05],
        horizontal_flip=False,
        vertical_flip=False,
        validation_split=0.2
    )

    train_generator = data_generator.flow_from_directory(
        directory=directory,
        target_size=(image_height, image_width),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=12,
        subset="training"
    )

    validation_generator = data_generator.flow_from_directory(
        directory=directory,
        target_size=(image_height, image_width),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=12,
        subset="validation"
    )

    return train_generator, validation_generator


def classification_model_train(train, val, epochs, batch_size, image_size):
    # Callbacks functions
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1,
                                   restore_best_weights=True
                                   )

    checkpointer = ModelCheckpoint(filepath='best_model.h5',
                                   save_best_only=True
                                   )

    # Optimizer
    optimizer = Adam(learning_rate=0.0001)

    # Weights
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=np.unique(train.classes),
                                         y=train.classes)

    class_weights = dict(zip(np.unique(train.classes), class_weights))

    # Building model
    classes = os.listdir('data/train_images_jpg')
    n_classes = len(classes)
    model = custom_model(image_size, image_size, n_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Training part
    hist = model.fit(train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=val,
                     callbacks=[early_stopping, checkpointer],
                     class_weight=class_weights)
    return hist


def segmentation_model_train(images_dir, mask_dir, epochs):
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = CustomDataset(images_dir=images_dir,
                            masks_dir=mask_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Building model part
    print("Building segmentation model.")
    model = UNet(n_channels=3, n_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training model part
    print("Starting segmentation model training.")
    for epoch in range(1, epochs + 1):
        train_loss, train_iou = train_seg(model, train_loader, optimizer, criterion, device, epoch, epochs)
        val_loss, val_iou = validate(model, val_loader, criterion, device, epoch, epochs)
        print(
            f"After Epoch {epoch}: Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved")
            print("")


def main():
    # Prepare images for training
    BATCH_SIZE = 32
    EPOCHS = 10
    image_size = 256

    # Data generators
    directory = 'data/train_images_resized'
    train, val = create_data_generators(directory, image_size, image_size, BATCH_SIZE)

    # Training models part
    images_dir = 'data/train_images_resized/Not Normal'
    mask_dir = 'data/train_images_masks/Not Normal'

    classification_model_train(train, val, EPOCHS, BATCH_SIZE, image_size)
    segmentation_model_train(images_dir, mask_dir, 20)


if __name__ == "__main__":
    main()






