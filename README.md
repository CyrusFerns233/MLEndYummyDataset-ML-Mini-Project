# MLEndYummyDataset-ML-Mini-Project

# MLEnd Yummy Dataset - Rice vs Chips Classifier (Basic Solution)

## Machine Learning Pipeline:

### Data Import:

The dataset was initially extensive, containing instances of various dishes. The goal of the basic component was to create a model specifically for distinguishing dishes with rice and chips. Therefore, the data was filtered to retain only rows corresponding to rice and chips dishes. Here there is a class imbalance (Majority class: Rice, Minority Class: Chips) which will be tackled in the later part of the pipeline

- **Input:** MLEnd Yummy Dataset with images of various dishes and labels (3250 instances).
- **Output:** Filtered dataset with instances of only rice and chips dishes (612 instances).

### Data Preprocessing:

#### Train-Test Split:

Before any preprocessing, it was crucial to establish separate sets for training and testing. This division was necessary for model evaluation, allowing me to assess the model's performance on unseen data and ensure generalization.

- **Input:** Filtered dataset with instances of only rice and chips dishes (612 instances).
- **Output:** Training and testing sets containing labeled instances of rice and chips.

#### Resize Images:

Images in the dataset had varying dimensions, which could further hinder uniform processing. The images were resized to a size of 200 x 200 pixels. This ensured that subsequent preprocessing steps and the model itself could handle inputs of consistent dimensions.

- **Input:** Images of rice and chips from the training and testing sets.
- **Output:** Resized images with dimensions 200 x 200 pixels.

#### Flatten Images:

Traditional machine learning models often expect flat feature vectors as input. Flattened the resized images, originally 2D arrays, transformed them into 1D feature vectors. This step prepares the data in a format suitable for training classifiers that operate on flattened representations.

- **Input:** Resized images with dimensions 200 x 200 pixels.
- **Output:** Flattened feature vectors ready for model training.

### Feature Extraction:

#### GLCM Features:

GLCM features were used to capture texture information in the images, which is essential for distinguishing between rice and chips dishes.

- **Input:** Flattened images.
- **Output:** GLCM features (dissimilarity, correlation) extracted from each image.

### Model Building:

A Random Forest Classifier was chosen due to its versatility and robustness, making it suitable for handling complex relationships in the data, particularly in the context of GLCM features.

- **Input:** GLCM features.
- **Output:** The selected Random Forest Classifier.

#### Random Forest Classifier:

The Random Forest Classifier is a powerful ensemble learning model that combines multiple decision trees to improve predictive accuracy and generalization.

- **Input:** GLCM features.
- **Output:** Trained Random Forest Classifier.

### Model Training:

The model undergoes training using the resampled training set, where the SMOTE oversampler is employed to address class imbalance. During training, the Random Forest Classifier learns patterns and relationships present in the GLCM features of rice and chips dishes.

- **Input:** Training set with GLCM features (resampled using SMOTE).
- **Output:** Trained Random Forest Classifier.

### Model Evaluation:

The trained model is evaluated on the same training set to assess its performance and ensure it has learned the underlying patterns effectively.

- **Input:** Trained Random Forest Classifier, Training set with GLCM features.
- **Output:** Evaluation metrics such as accuracy, confusion matrix, and classification report.

### Training Set Evaluation:

#### Training Accuracy:

The model achieves perfect accuracy on the training set, scoring 100%. This indicates that the classifier has successfully learned the patterns and relationships present in the GLCM features of rice and chips dishes.

#### Confusion Matrix:

The matrix shows no misclassifications, with 374 instances correctly classified for both classes (0 and 1).

#### Classification Report:

Precision, recall, and f1-score for both classes are perfect, reflecting the model's ability to generalize well on the training data.

### Test Set Evaluation with GLCM Features:

#### Test Accuracy:

The model achieves a test accuracy of approximately 71.28% on a separate set of data, indicating its ability to generalize to new, unseen instances.

#### Test Confusion Matrix:

The model correctly predicts 126 instances of class 1 but struggles with class 0, where only 8 instances are correctly predicted, and 12 instances are misclassified.

#### Test Classification Report:

The classification report provides a detailed breakdown of precision, recall, and f1-score for both classes in the test set, offering insights into the model's performance across different metrics. The report indicates that while the model performs well for class 1, it has challenges correctly predicting instances of class 0.


# MLEnd Yummy Dataset - Diet Category Classifier (Advanced Solution)

## Advanced Machine Learning Pipeline:

### Data Loading and Preprocessing Stage:

- **Input:** Filepaths, labels (Diet information)
- **Output:** Preprocessed images and their corresponding encoded labels

#### Stages:

1. Loading filepaths and labels from MLENDYD_df.
2. Using the `load_images_from_dir` function to load and preprocess images, resizing them to the specified size (200x200), and normalizing pixel values to the range [0, 1].
3. Encoding labels using LabelEncoder and to_categorical.
4. Splitting the dataset into training, validation, and test sets using `train_test_split`.

### Model Definition and Compilation Stage:

- **Input:** Preprocessed images and encoded labels
- **Output:** Compiled CNN model

#### Stages:

1. Defining a more complex CNN model with additional convolutional layers, dropout, and batch normalization for better feature extraction.
2. Compiling the model with the Adam optimizer, categorical crossentropy loss, and accuracy as the metric.

### Hyperparameter Tuning Stage:

- **Input:** Compiled CNN model, training data, and labels
- **Output:** Fine-tuned model

#### Stages:

1. Using techniques like random search to tune hyperparameters such as learning rate, batch size, and model architecture.
2. Optimizing the model based on validation performance.

### Data Augmentation Stage:

- **Input:** Training images and labels
- **Output:** Augmented training data

#### Stages:

1. Using the `ImageDataGenerator` from Keras to perform data augmentation.
2. Augmenting the training data with various transformations like rotation, width and height shifts, shear, zoom, horizontal and vertical flips, and fill mode.

### Model Training Stage:

- **Input:** Augmented training data and labels, validation data, and callbacks
- **Output:** Trained model and training history

#### Stages:

1. Training the model using the augmented training data and labels from the data augmentation stage.
2. Using a learning rate reduction callback (`ReduceLROnPlateau`) to adjust the learning rate during training based on the validation loss.

### Plotting Training History Stage:

- **Input:** Training history
- **Output:** Visualization of training and validation metrics over epochs

#### Stages:

1. Plotting training accuracy, validation accuracy, training loss, and validation loss over epochs using the `plot_training_history` function.

### Model Evaluation Stage:

- **Input:** Test data and labels
- **Output:** Test accuracy

#### Stages:

1. Evaluating the trained model on the test set to obtain the test accuracy.
