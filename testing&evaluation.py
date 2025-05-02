import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from data_loading_eda import load_images_from_folder, class_names
from model import model1, image_modifications, preprocess_images

def shuffle_in_unison(images, labels):
  """Shuffles two NumPy arrays in unison.

  Args:
    images: The first NumPy array (e.g., images).
    labels: The second NumPy array (e.g., labels).

  Returns:
    A tuple containing the shuffled images and labels arrays.
  """
  assert len(images) == len(labels), "Arrays must have the same length"

  # Generate a permutation of indices
  indices = np.arange(len(images))
  np.random.shuffle(indices)

  # Shuffle both arrays using the same permutation
  shuffled_images = images[indices]
  shuffled_labels = labels[indices]
  print("Shuffled images:", shuffled_images)
  print("Shuffled labels:", shuffled_labels)
  return shuffled_images, shuffled_labels

#plot predictions
def plot_predictions(model1, test_images, test_labels, class_names):
  # Predict on the test images
  predictions = model1.predict(test_images)
  predicted_labels = np.argmax(predictions, axis=1)

  # Plot the first ten images and predictions vs. real labels
  plt.figure(figsize=(12, 8))
  for i in range(10):
      plt.subplot(2, 5, i + 1)
      plt.imshow(test_images[i].reshape(128, 128), cmap='gray')
      plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[test_labels[i]]}")
      plt.axis('off')
  plt.tight_layout()
  plt.show()

def generate_cr_cm(model1, test_images, test_labels, class_names):
  # Predict on the test set
  y_pred_prob = model1.predict(test_images)
  y_pred = np.argmax(y_pred_prob, axis=1)

  # Assuming 'shuffled_labels' and 'y_pred' are defined as in your existing code
  cm = confusion_matrix(test_labels, y_pred)

  # Plot the confusion matrix
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()
  # Generate classification report
  report = classification_report(test_labels, y_pred, target_names=class_names)
  print(report)


if __name__ == '__main__':
  # Usage
  # Load images from the testing folder, resize, and convert to grayscale
  test_folder_path = '/content/brain-tumor-mri-dataset/Testing'
  test_images, test_labels, _ = load_images_from_folder(test_folder_path)

  test_images = np.array([image_modifications(image) for image in test_images])
  test_images = preprocess_images(test_images)
  print(test_images.shape, test_labels.shape)

  # Shuffle the test set
  test_images, test_labels = shuffle_in_unison(test_images, test_labels)
  print(test_images.shape, test_labels.shape)
  plot_predictions(model1, test_images, test_labels, class_names)
  generate_cr_cm(model1, test_images, test_labels, class_names)

