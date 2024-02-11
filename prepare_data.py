# Imports
import os
import csv

import pandas as pd
from PIL import Image, ImageDraw
import pydicom
import shutil


# Directory paths
input = 'data/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv'
input_images = 'data/rsna-pneumonia-detection-challenge/stage_2_train_images'
test_dcm = 'data/rsna-pneumonia-detection-challenge/stage_2_test_images'

classified_images = 'data/temporary/train_images'
train_jpg = 'data/temporary/train_images_jpg'

train_masks = 'data/train_images_masks'
resized_path = 'data/train_images_resized'
test_jpg = 'data/test_images_jpg'
output_file = 'data/output.csv'
train_destination = 'data/images'
masks_destination = 'data/masks'


# Util functions
def convert_to_jpg(input_folder, output_folder):
    """
    Converts DICOM medical images to JPEG format in a structured manner.
    Args:
        input_folder (str): Path to the directory containing DICOM files.
        output_folder (str): Path to the directory where converted JPEG images will be saved.
    """
    for root, dirs, files in os.walk(input_folder):
        for name in files:
            if name.endswith('.dcm'):
                dicom_path = os.path.join(root, name)
                output_subfolder = os.path.relpath(root, input_folder)
                output_subfolder_path = os.path.join(output_folder, output_subfolder)
                os.makedirs(output_subfolder_path, exist_ok=True)

                ds = pydicom.dcmread(dicom_path)
                img = ds.pixel_array.astype(float)
                img = img.astype('uint8')

                img = Image.fromarray(img)
                img.save(os.path.join(output_subfolder_path, name.replace('.dcm', '.jpg')))


def merge_patients(df):
    """
    Merges rows in a DataFrame with duplicate patient IDs by aggregating specified columns, preserving uniqueness while
    handling potential conflicts effectively.
    Args:
        df (pandas.DataFrame): The input DataFrame containing patient data. The DataFrame must have a column named
        'patientId' for patient identification.
    Returns:
        pandas.DataFrame: The merged DataFrame.
    """
    return df.groupby('patientId').agg(lambda x: ' '.join(x.astype(str)) if len(x) > 1 else x.iloc[0]).reset_index()


def create_folders_from_csv(input_folder, csv_file, output_folder):
    """
    Creates folders in output_folder based on image classes in csv_file.
    Args:
        input_folder (str): Input folder containing images.
        csv_file (str): CSV file with "image_name" and "image_class" columns.
        output_folder (str): Output folder for class folders.
    """
    image_classes = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            image_name, image_class = row
            image_name_without_ext = os.path.splitext(image_name)[0]
            image_classes[image_name_without_ext] = image_class

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_name_without_ext = os.path.splitext(image_name)[0]

        if image_name_without_ext in image_classes:
            class_folder = os.path.join(output_folder, image_classes[image_name_without_ext])
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            image_path = os.path.join(input_folder, image_name)
            os.rename(image_path, os.path.join(class_folder, image_name))

        else:
            os.remove(os.path.join(input_folder, image_name))


def merge_classes(csv_file, output_file):
    """
    Processes a pneumonia classification CSV file, merging specific classes and saving the result.
    Args:
        csv_file (str): Path to the input CSV file containing pneumonia classifications.
        output_file (str, optional): Path to save the processed CSV file.
    """
    df = pd.read_csv(csv_file)

    df['class'] = df['class'].replace('No Lung Opacity / Not Normal', 'Normal')
    df['class'] = df['class'].replace('Lung Opacity', 'Not Normal')

    df.to_csv(output_file, index=False)


def _split_and_handle_empty(column_values):
    """
    Splits a string by spaces, handling empty values efficiently.
    Args:
        column_values (str): String containing values separated by spaces.
    Returns:
        list: List of values, handling empty values gracefully.
    """
    if column_values.strip() == "":
        return []
    else:
        return column_values.split()


def create_masks(df, mask_dir):
    """
    Creates masks for each image in the dataframe.
    Args:
        df (pandas.DataFrame): Dataframe containing image names, coordinates,
            sizes, and targets.
        mask_dir (str): Path to the directory for saving the generated masks.
    """
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.join(mask_dir, "Normal"), exist_ok=True)
    os.makedirs(os.path.join(mask_dir, "Not Normal"), exist_ok=True)

    for _, row in df.iterrows():
        image_name = row[0]

        mask = Image.new("RGB", (1024, 1024), (0, 0, 0))

        target_values = _split_and_handle_empty(row["Target"])
        x_values = _split_and_handle_empty(row["x"])
        y_values = _split_and_handle_empty(row["y"])
        width_values = _split_and_handle_empty(row["width"])
        height_values = _split_and_handle_empty(row["height"])

        if len(target_values) != len(x_values) or len(target_values) != len(y_values) \
           or len(target_values) != len(width_values) or len(target_values) != len(height_values):
            raise ValueError(f"Length mismatch in coordinates or target values for image '{image_name}'.")

        draw = ImageDraw.Draw(mask)
        for target, x, y, w, h in zip(target_values, x_values, y_values, width_values, height_values):
            target, x, y, w, h = target.split(".")[0], x.split(".")[0],  y.split(".")[0], w.split(".")[0], h.split(".")[0]
            target, x, y, w, h = int(target), int(x), int(y), int(w), int(h)

            draw.rectangle([(x, y), (x + w, y + h)], fill="white")

        target_folder = "Normal" if target == 0 else "Not Normal"
        mask_image_name = f"{image_name}_mask.jpg"
        mask_path = os.path.join(mask_dir, target_folder, mask_image_name)
        mask = mask.resize((256, 256), Image.ANTIALIAS)
        mask.save(mask_path, 'JPEG')


def resize_images(input_folder_path, output_folder_path, image_size=256):
    """
    Resizes all images in a folder and its subfolders to the specified size, creating
    a new folder structure with the same hierarchy.

    Args:
        input_folder_path (str): Path to the folder containing images.
        output_folder_path (str): Path to the output folder where resized images will be saved.
        image_size (int, optional): Desired width and height of the resized images. Defaults to 256.
    """
    if os.path.exists(output_folder_path):
        print(f"Output folder '{output_folder_path}' already exists. Images will be saved there.")
    else:
        os.makedirs(output_folder_path)

    for subfolder in os.listdir(input_folder_path):

        input_subfolder_path = os.path.join(input_folder_path, subfolder)
        output_subfolder_path = os.path.join(output_folder_path, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)

        for filename in os.listdir(input_subfolder_path):
            input_image_path = os.path.join(input_subfolder_path, filename)
            output_image_path = os.path.join(output_subfolder_path, filename)

            image = Image.open(input_image_path)
            resized_image = image.resize((image_size, image_size), Image.ANTIALIAS)
            resized_image.save(output_image_path)


def main():
    # Start of preparation
    labels_data = pd.read_csv('data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

    labels = merge_patients(labels_data.copy())
    labels.fillna(0, inplace=True)
    labels = labels.astype(str)

    merge_classes(input, output_file)
    create_folders_from_csv(input_images, output_file, classified_images)

    # Converting images to jpg
    if not os.path.exists(train_jpg):
        os.makedirs(train_jpg)
    convert_to_jpg(classified_images, train_jpg)

    if not os.path.exists(train_jpg):
        os.makedirs(train_jpg)
    convert_to_jpg(test_dcm, test_jpg)

    # Creating masks for segmentation model
    create_masks(labels, train_masks)

    # Resize images to 256x256
    resize_images(train_jpg, resized_path, 256)  # resizing to 256x256


if __name__ == '__main__':
    main()