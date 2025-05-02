import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import opendatasets as od

def load_dataset(dataset_url, download_path):
    """Downloads the dataset if it doesn't exist."""
    if not os.path.exists(download_path):
        od.download(dataset_url)
    return download_path

def load_images_from_folder(folder_path, target_size=(128, 128)):
    """Loads images from a folder and assigns labels."""

    images = []
    labels = []
    class_names = os.listdir(folder_path)

    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                try:
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, dsize=target_size)  
                    images.append(image)
                    labels.append(class_index)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

    return np.array(images), np.array(labels), class_names

def visualize_class_distribution(labels, class_names):
    """Visualizes the distribution of classes in the dataset."""
    df = pd.DataFrame(labels, columns=['class'])
    sns.countplot(x='class', data=df, hue='class')
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.show()

def extract_glcm_features(image):
    """Extracts GLCM features from a grayscale image."""

    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    return [contrast, dissimilarity, homogeneity, energy, correlation]

def process_images_and_extract_features(images):
    """Extracts GLCM features from all images."""
    all_features = []
    for image in images:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = extract_glcm_features(image)
        all_features.append(features)
    return np.array(all_features)

def create_dataframe(features, labels, class_names):
    """Creates a Pandas DataFrame from features and labels."""

    combined_data = np.column_stack((features, labels))
    df = pd.DataFrame(combined_data, columns=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'class'])
    return df

def visualize_feature_distributions(df, class_names):
    """Visualizes the distribution of features by class."""

    for feature in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        plt.figure()
        sns.histplot(data=df, x=feature, hue='class', kde=True)
        plt.title(f'Distribution of {feature} by Class')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.xticks(ticks=range(len(class_names)), labels=class_names)
        plt.show()

if __name__ == '__main__':
    # Usage Example
    dataset_url = 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset'
    download_path = '/content/brain-tumor-mri-dataset'  # Or your desired path
    folder_path = load_dataset(dataset_url, download_path)
    training_path = os.path.join(folder_path, 'Training')
    images, labels, class_names = load_images_from_folder(training_path, target_size=(128, 128)) # Added target size

    print(f"Loaded {len(images)} images.")
    print(f"Class names: {class_names}")

    visualize_class_distribution(labels, class_names)

    features = process_images_and_extract_features(images)
    df = create_dataframe(features, labels, class_names)
    visualize_feature_distributions(df, class_names)