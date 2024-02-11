import os

import numpy as np
import pandas as pd

from PIL import Image

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

from skimage.measure import label, regionprops

import torch
from torchvision import transforms, models

from models import UNet


def load_and_prepare_image(file_path, target_size=(256, 256)):
    """
    Loads an image from a file, resizes it, and converts it to a format suitable for
    deep learning models.
    Args:
        file_path: Path to the image file.
        target_size: A tuple (width, height) specifying the desired output size. Defaults to (256, 256).
    Returns:
        A preprocessed NumPy array of the image, ready for use in a deep learning model.
    """
    img = image.load_img(file_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return preprocess_input(img_array_expanded_dims)


def extract_optimal_bounding_boxes(binary_img, scale_factor):
    """
    Extracts bounding boxes tightly enclosing connected regions in a binary image.
    Args:
        binary_img: A binary image (NumPy array) where non-zero pixels represent objects.
        scale_factor: A scaling factor to apply to the bounding box coordinates.
    Returns:
        A list of bounding boxes, where each bounding box is a tuple (x_min, y_min, width, height).
    """
    labeled_img = label(binary_img)
    regions = regionprops(labeled_img)

    boxes = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        x_min = min_col * scale_factor
        y_min = min_row * scale_factor
        width = (max_col - min_col) * scale_factor
        height = (max_row - min_row) * scale_factor
        boxes.append((x_min, y_min, width, height))

    return boxes


def predict_lung(images_folder, model_class, model_seg):
    scale_factor = 1024 / 256

    results = []
    results_check = []
    for file_name in os.listdir(images_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(images_folder, file_name)

            img_prepared = load_and_prepare_image(file_path)

            prediction = model_class.predict(img_prepared, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)
            x_min, y_min, width, height = None, None, None, None
            res = ""

            if predicted_class[0] == 1:
                image_tensor = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])(Image.open(file_path).convert('RGB')).unsqueeze(0)

                with torch.no_grad():
                    output = model_seg(image_tensor)
                    predicted_mask = torch.sigmoid(output[0]).squeeze().cpu().numpy() > 0.5
                    bounding_boxes = extract_optimal_bounding_boxes(predicted_mask, scale_factor)

                    if bounding_boxes:
                        if bounding_boxes != []:
                            for i in range(len(bounding_boxes)):
                                x_min, y_min, width, height = bounding_boxes[i]
                                res = res + '1' + " " + str(x_min) + " " + str(y_min) + " " + str(width) + " " + str(
                                    height) + " "

            results.append({
                'patientId': file_name.split('.jpg')[0],
                'PredictionString': res
            })
            results_check.append({
                'File Name': file_name,
                'Predicted Class': predicted_class[0],
                'x-min': x_min,
                'y-min': y_min,
                'width': width,
                'height': height
            })

    results_df = pd.DataFrame(results)
    results_df2 = pd.DataFrame(results_check)

    return results_df, results_df2


def main(images_dir='data/test_images_jpg'):
    model_class = load_model('best_model_class.h5')
    model_path = 'best_model.pth'
    model_seg = UNet(n_channels=3, n_classes=1)
    model_seg.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_seg.eval()

    results_df, results_df2 = predict_lung(images_dir, model_class, model_seg)

    results_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    images_dir = 'data/test_images_jpg'  # or path to your directory with images
    main(images_dir)
