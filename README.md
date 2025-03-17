Butterfly Image Classification Using Deep Learning (CNN)

📊 Project Summary¶
1. Introduction


This project aims to classify butterfly species from images using Convolutional Neural Networks (CNN). It involves data preprocessing, model training, and evaluation to achieve accurate species identification.

2.Technologies Used
 
Python,NumPy,Pandas,Matplotlib

TensorFlow/Keras - For building and training the CNN model.

OpenCV - For image processing.



3.Dataset

The Butterfly Image Classification dataset consists of 6,499 images, divided into:

Training Set: 5,000 images for training the CNN model.

Test Set: 1,499 images for evaluating the model's accuracy.


4. System Workflow

1️⃣ Data Preprocessing – Load, resize, normalize images, and split into training & test sets.

2️⃣ Model Development – Build, train, and evaluate a CNN model for classification.

3️⃣ Prediction & Evaluation – Use the trained model to classify butterfly images and assess performance.


📊 Model Development (CNN)

1️⃣ Data Preprocessing

 - Importing the train and test image files.

![image](https://github.com/user-attachments/assets/c0c379e9-f0cc-418b-b395-a148d4ff3f56)
![image](https://github.com/user-attachments/assets/f5f76762-8348-4bde-b88e-5052e3b6e7de)

 - Importing essential libraries for data processing, visualization, model training, and evaluation. It includes:

pandas, numpy, os – Data handling.

matplotlib, seaborn – Data visualization.

sklearn – Data splitting & model evaluation.

tensorflow.keras – CNN model creation and image preprocessing.

warnings – Suppressing unnecessary warnings.

![image](https://github.com/user-attachments/assets/357fc3a7-dee3-4701-8508-faa3b0f4660d)

 - Data Analysis Step

![image](https://github.com/user-attachments/assets/ab2522cf-9c77-4342-bc33-c20037d3854e)

![image](https://github.com/user-attachments/assets/3359e93b-c85b-42d3-b088-8083afab1521)

![image](https://github.com/user-attachments/assets/c79f45b7-93cf-4fb1-aa7c-972af6f4af16)

![image](https://github.com/user-attachments/assets/98d09e25-9899-4edf-a289-905027b1761c)

 - Analyzing the distribution of butterfly species in the dataset.

Counts the number of images per species.

Sorts the classes and plots a bar chart using Seaborn.

Displays class names on the x-axis and image count on the y-axis.

![image](https://github.com/user-attachments/assets/cce1309e-aa34-49ee-88ff-2a1f3becc10a)

![image](https://github.com/user-attachments/assets/8f33acd5-ed87-4299-8bd1-7b8ac76955f1)


-  Visualizing class distribution using a pie chart.

Counts butterfly species occurrences.

Uses a "magma" color palette for better distinction.

Customizes the pie chart with edge color, labels, and a central circle for aesthetics.

Displays percentage values for each species.

![image](https://github.com/user-attachments/assets/7f32dda6-4178-4fec-94cd-3084fd96bd32)

![image](https://github.com/user-attachments/assets/7cdd9b79-d98e-48e5-96da-cc078fc4958e)


 - Displaying sample images from the dataset.

Randomly selects 9 images using df.sample(9).Loads images from the training directory (train/train).
Resizes images to (224, 224) for consistency.
Normalizes pixel values to the range [0, 1].
Displays a 3x3 grid of images with their respective class labels.

![image](https://github.com/user-attachments/assets/29c2b0b4-7b13-45a9-9123-580e995f3095)

 - Preparing image data for training and validation using ImageDataGenerator.

Splits dataset into 4,100 training images and 900 validation images across 75 classes.

Applies data augmentation (rotation, shifting, zooming, flipping) to improve model generalization.

Rescales images to [0,1] for normalization.

Loads images in batches of 32 with a target size of (224, 224).

![image](https://github.com/user-attachments/assets/bee92733-4a5c-4474-9f33-828f5eaf01d5)

2️⃣ Model Development(CNN)

 - Defining and compile a CNN model for butterfly image classification.

Architecture:

4 convolutional layers (Conv2D) with ReLU activation.

MaxPooling layers to reduce spatial dimensions.

Flatten layer to convert features into a 1D vector.

Dense layers (256 neurons with ReLU, 75 neurons with softmax for classification).

Compilation:

Optimizer: Adam

Loss Function: Categorical Crossentropy (for multi-class classification).

Metric: Accuracy
 
   ![image](https://github.com/user-attachments/assets/a5a7c746-8549-43c4-9002-df9ede38deba)

   ![image](https://github.com/user-attachments/assets/cd9b9e9b-c6c5-4391-ac4c-438f9a1b4c31)


 - Running to train the CNN model with early stopping.

Early Stopping:

Monitors val_loss (or val_accuracy).

Stops training if no improvement after 10 epochs.

Restores the best model weights.

Model Training (fit):

Uses train_generator for training data.

Runs for 50 epochs (unless early stopping occurs).

Validates with val_generator.

![image](https://github.com/user-attachments/assets/a5fcd3de-a3bb-4c5d-8299-ecf814b0c491)

![image](https://github.com/user-attachments/assets/c1c04c5b-5fcb-40ed-8236-dbbb8dacd636)

3️⃣ Prediction & Evaluation

 - Visualizing training performance using Matplotlib.

First subplot:

Plots training and validation accuracy over epochs.

Helps track model improvement.

Second subplot:

Plots training and validation loss over epochs.

Helps detect overfitting or underfitting.

Legend & Layout:

Differentiates between train & validation data.

Uses tight_layout() for better spacing.

![image](https://github.com/user-attachments/assets/6ed8b8e5-8caa-4aee-8d8b-90f6e8fa0e77)
![image](https://github.com/user-attachments/assets/5b146221-e459-4156-94d1-cc8652b276fd)

- Saving the model as:
  
![image](https://github.com/user-attachments/assets/b95d7d79-4322-45ec-8ce4-dff7338df6f0)

![image](https://github.com/user-attachments/assets/42adf1c4-c7d6-4821-a9eb-fbeaa83928ae)

 - Making predictions on the test set and save results:

Predicts labels using the trained CNN model.

Maps predicted class indices to actual class names.

Creates a DataFrame with ID (image filenames) and label (predicted species).

Saves results to submission.csv for further analysis or competition submission.

Displays the first few predictions for verification.

![image](https://github.com/user-attachments/assets/e67b2577-5aa8-4122-9336-1a0be03cb055)

 - Install this result to your application:

![image](https://github.com/user-attachments/assets/7ccc2528-614c-49d8-a8bf-9c2dfcac1985)





























































