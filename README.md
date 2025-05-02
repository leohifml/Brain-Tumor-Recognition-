# Brain Tumor MRI Classification using ResNet18 and GLCM Features

This project aims to classify brain tumor MRI images into different categories (glioma, meningioma, pituitary, no tumor) using a combination of deep learning and traditional image processing techniques. 

## Methodology

1. **Dataset:** The project uses the Brain Tumor MRI Dataset from Kaggle.
2. **Image Preprocessing:** Images are preprocessed by resizing, converting to grayscale, and applying Gaussian blur.
3. **Feature Extraction:** Gray-Level Co-occurrence Matrix (GLCM) features are extracted to capture texture information from the images.
4. **Deep Learning Model:** A ResNet18 model is trained for image classification.
5. **Model Evaluation:** The trained model is evaluated using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the model's performance.

## How to Run

1. **Install Dependencies:** Make sure you have the necessary libraries installed. You can install them using `pip`:
2. **Download Dataset:** Download the Brain Tumor MRI Dataset from Kaggle and extract it to a folder. Update the `dataset_url` and `download_path` variables in the code accordingly.
3. **Run the Code:** Execute the Python code in a Google Colab environment or Jupyter Notebook. The code will load the dataset, preprocess the images, extract features, train the ResNet18 model, and evaluate its performance.

## Results

The project achieved an accuracy of 98% on the test set. The classification report and confusion matrix provide detailed insights into the model's performance for each tumor category.

![images](download.png)  
## Future Work

* Explore other deep learning architectures, such as VGG16 or InceptionV3.
* Experiment with different feature extraction techniques.
* Fine-tune hyperparameters to further improve model performance.
* Develop a user interface for easier interaction with the model.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs (CC BY-NC-ND) License. 

## Acknowledgments

* The Brain Tumor MRI Dataset from Kaggle.
* The creators of the ResNet18 architecture.
* The developers of the libraries used in this project.
