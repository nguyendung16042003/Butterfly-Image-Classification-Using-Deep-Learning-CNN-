Heart Disease Prediction System project(Deep Learning Web)

📊 Project Summary¶
1. Introduction


This project develops a system for predicting heart disease risk based on patient medical data. It utilizes Deep Learning (Artificial Neural Network - ANN) to analyze health factors and provide risk predictions.

2.Technologies Used
 
•	Machine Learning / Deep Learning: TensorFlow, Keras

•	Data Processing: Pandas, NumPy

•	Web Framework: Flask

•	Frontend: HTML, CSS, JavaScript

•	Database: SQLite / MySQL

•	Deployment: FastAPI, Docker

3.Dataset

•	The dataset consists of 1,025 samples with 14 medical features, including: 

  - age (Age), sex (Gender), cp (Chest Pain), chol (Cholesterol), thalach (Max Heart Rate), oldpeak (ST Depression), thal (Thalassemia index), etc.

  -	"target" column: The prediction outcome (0: No disease, 1: Has heart disease).

4. System Workflow

1️⃣ Data Preprocessing: Cleaning, normalizing, and handling missing data.

2️⃣ Model Training: Using an Artificial Neural Network (ANN) to learn from medical data.9

3️⃣ Prediction & Evaluation: Testing the model on new data and optimizing results.

4️⃣ Web Deployment: Users enter their health data on the web, and the system predicts heart disease risk.

📊 Model Development (ANN)

1.Importing essential libraries for data processing, visualization, machine learning, and deep learning.

Data Handling & Visualization: pandas, numpy, matplotlib, seaborn

Machine Learning: scikit-learn (train-test split, scaling, logistic regression, evaluation metrics)

Deep Learning: TensorFlow & Keras (Sequential model, Dense, Dropout layers)

![image](https://github.com/user-attachments/assets/8a5765bf-15e9-4449-a31b-5cf30ef6b18a)

2.Load dataset: Read CSV file and display basic info.

Train-Test Split: Split data (85% train, 15% test) with stratification.

![image](https://github.com/user-attachments/assets/e450d8e8-36af-45c5-a846-0943ee3b41b3)
![image](https://github.com/user-attachments/assets/e2e329f7-abe5-4c7a-ae5d-500f26e9486f)

3.Model Training: Fits the best model on training data.

- LogisticRegression

Prediction: Generates predictions on test data.

Confusion Matrix: Visualizes model performance with a heatmap.

Evaluation: Calculates accuracy and prints a classification report.

![image](https://github.com/user-attachments/assets/dbfc4c52-db30-436e-a757-521108e73717)
![image](https://github.com/user-attachments/assets/d6937df5-4b02-46c0-b6d8-26d2e71a6c63)

- ANN Model: Uses a Sequential model with Dense layers, ReLU activation, and Dropout for regularization.
  
![image](https://github.com/user-attachments/assets/d6b81e11-4caf-432f-853f-31627c3a87fe)
-![image](https://github.com/user-attachments/assets/f7253ea4-e6fb-40b8-835b-89b601c0b87a)


4.Plot Training Progress: Visualizes model accuracy over epochs.

Train vs Validation Accuracy: Compares model performance on training and validation data.

Graph Elements:

X-axis: Epochs

Y-axis: Accuracy
![image](https://github.com/user-attachments/assets/a72bb5bc-7b6d-478e-a0d2-98b964e1bdd5)

📊 Build The Website 

1.Main

This code sets up a Flask-based web app for heart disease prediction, handling data with pandas/numpy and loading a pre-trained model with TensorFlow.

![image](https://github.com/user-attachments/assets/16f8c28d-5cf5-4030-a1f5-74ba61a5e28d)

This code loads a pre-trained machine learning model from the file "model.h5" using TensorFlow's load_model function.

![image](https://github.com/user-attachments/assets/cb7c8262-b28b-4f28-960b-5ba0e8f02f40)


This Flask application provides a heart disease prediction web service:

index(): Renders the homepage (trangchu.html).

nhapthongtin(): Handles form submission (GET/POST). Converts user input into a DataFrame and predicts heart disease using a trained model. Redirects to the results page.

du_doan(): Processes POST requests for prediction. Converts input to DataFrame, makes a prediction, and renders the results (trangkqduketqua.html).

![image](https://github.com/user-attachments/assets/c85a7dd5-65dc-48e3-b822-852feabde6a0)

![image](https://github.com/user-attachments/assets/337ed293-1b67-4061-ae47-b1b7bc3c548a)

2. Trangchu.html

This is the homepage interface for the Heart Disease Prediction System, built using Flask and Bootstrap 5. Key features include:

Animated Navbar: Displays a welcome message with a scrolling effect.

Visually Appealing Design: Gradient background with animations (marquee, pulse, zoomOut, fadeIn).

![image](https://github.com/user-attachments/assets/6ef70803-46ab-4bf0-b831-b4cc4463d062)

Interactive "Enter Information" Button:

The main content fades out and disappears.

An email input form appears after clicking the button.

Email Confirmation Before Proceeding: Users must enter their email before being redirected to the health information input page (/trangnhapthongtin).


![image](https://github.com/user-attachments/assets/345bade5-f71e-4a63-a3a8-ab00b149233f)

3. Trangnhapthongtin.html

This is the Patient Information Input Page for the Heart Disease Prediction System, built using Flask and Bootstrap 5. Key features:

User Consent Prompt: Before entering information, users must agree to provide their details.

![image](https://github.com/user-attachments/assets/c8910be4-78e6-4cba-940a-a53d22503193)


Interactive Form:

Collects essential health metrics such as age, gender, cholesterol, ECG results, and more.

Uses dropdowns and number inputs for easy data entry.

Designed with validation (required attributes) to ensure correct input.

Modern Design:

Gradient background for a visually appealing look.

Animations (pulse effect) to enhance the user experience.

Submission Process:

Once completed, the form submits data to /trangxuatketqua for heart disease prediction processing.

![image](https://github.com/user-attachments/assets/9ba36041-df7e-47e6-828c-8ac696f8a8f4)

4, Trangxuatketqua.html

This is the Prediction Result Page for the Heart Disease Prediction System, designed with Flask and Bootstrap 5. Key features:

Result Display:

Shows the prediction outcome from the trained model ({{ ketqua }}).

Provides a friendly message advising users to consult a doctor for a proper diagnosis.

Modern & Clean UI:

Gradient background for an appealing design.

A card-like container with a soft shadow effect.

An icon for better visualization.

Navigation:

A custom-styled button allowing users to return to the homepage (/).

![image](https://github.com/user-attachments/assets/53f5c842-806a-4e99-b2a3-d6f6046b37f2)



































