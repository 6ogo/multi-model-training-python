# Model Training and Evaluation App

### Key Points in the README:
1. **Project Overview** - Provides a brief explanation of the app.
2. **Features** - Lists the features of the app, including the models supported, the types of evaluations, and visualizations available.
3. **Installation Instructions** - Guides users on how to clone the repository, set up a virtual environment, and install dependencies.
4. **Usage** - A step-by-step guide on how to use the app after it is set up, including how to upload data, select models, tune hyperparameters, and view results.
5. **Contributing** - Information on how to contribute to the repository.
6. **License** - Information about the project's license.


## Project Overview

This repository contains a Streamlit-based application designed for model training and evaluation on a given dataset. The app allows users to upload a CSV file, select target variables, choose from multiple machine learning models, and evaluate their performance. The application includes data preprocessing steps like handling missing values, feature selection based on correlation, and scaling, alongside hyperparameter tuning for some models. The app then outputs various performance metrics and visualizations, such as confusion matrices, ROC curves, feature importance plots, and model training histories.

## Features

- **Data Upload**: Upload a CSV file to the app.
- **Preprocessing**: 
  - Handle missing values.
  - Drop highly correlated features.
  - Scale numeric features.
- **Model Selection**: Choose from a variety of models including:
  - Logistic Regression
  - Random Forest Classifier
  - K-Nearest Neighbors
  - Support Vector Machine
  - Neural Network (using Keras)
  - XGBoost
  - CatBoost
  - LightGBM
- **Hyperparameter Tuning**: Customize model hyperparameters with options for grid search on some models.
- **Evaluation Metrics**: Evaluate the model performance using metrics like:
  - Accuracy
  - Log Loss
  - Confusion Matrix
  - Classification Report
  - ROC Curve
- **Visualizations**: 
  - ROC Curve for model performance.
  - Feature Importance plots for tree-based models.
  - Training History plots for Neural Network models.
  - Decision Tree visualization for Random Forest models.

## Installation Instructions

To run this application locally, you'll need Python 3.7+ and the following dependencies:

### Step 1: Clone the repository

```
git clone https://github.com/yourusername/model-training-app.git
cd model-training-app
```

### Step 2: Install dependencies
It is recommended to create a virtual environment before installing dependencies.

```
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the dependencies (mandatory)
pip install -r requirements.txt
```

#### Required Libraries
- Streamlit: The library for the app's interface.
- Pandas, Numpy: For data manipulation and numerical operations.
- Matplotlib, Seaborn: For plotting and visualization.
- Scikit-learn: For preprocessing, model selection, training, and evaluation.
- TensorFlow: For building and training the neural network.
- XGBoost, CatBoost, LightGBM: For tree-based model implementations.


### Step 3: Run the app
```
streamlit run multiModel.py
```
This will start the app in your browser.

## How to Use
Upload Dataset: Choose a CSV file to upload. The file should have numeric features and a target column for classification tasks.

Select Target Column: Choose the target variable (dependent variable) from the dataset for classification.

**Data Preprocessing:**
-Select columns to drop (if needed).
-Choose how to handle missing values (you can fill with a constant or mean).
-Optionally, choose a sample size for training.

**Model Selection:**
Pick the model to train from the dropdown list. The available models are:
-Logistic Regression
-Random Forest
-K-Nearest Neighbors
-Support Vector Machine
-Neural Network
-XGBoost
-CatBoost
-LightGBM

**Hyperparameter Tuning (Optional):**
Customize the hyperparameters for the selected model. The app will use grid search to find the best parameters for models that support it (e.g., Random Forest, XGBoost).
Training & Evaluation:

Click the "Start Training" button to train and evaluate the selected model. The app will display performance metrics such as accuracy, log loss, confusion matrix, classification report, and a ROC curve (for models with probability predictions).
Visualizations:

After training, you will be able to see various visualizations:
-ROC Curve
-Feature Importance (for tree-based models like Random Forest, XGBoost, CatBoost, and LightGBM)
-Training and Validation Accuracy & Loss (for Neural Networks)
-Decision Tree visualization (for Random Forest)

##Contributing
We welcome contributions to enhance the functionality and usability of this application. To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and ensure the code adheres to best practices.
Test your changes thoroughly.
Submit a pull request with a clear description of your changes.
Please adhere to the following guidelines when contributing:

Follow the code style and structure used in the repository.
Document any new features or updates in the README.md.
Provide clear and concise commit messages.
Ensure all dependencies are listed in requirements.txt.
If you encounter any issues, feel free to open an issue in the repository.

##License
This project is licensed under the MIT License.
