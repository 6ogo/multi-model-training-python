import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, log_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.tree import plot_tree

# Add error handling for imports
try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError as e:
    st.error(f"Missing required package: {e.name}. Please install it using pip.")

# Function to train and evaluate the model with hyperparameter tuning
def train_and_evaluate_model(X, y, model_choice, param_grid=None, correlation_threshold=0.82, test_size=0.3):
    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    X = X.drop(columns=to_drop)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_choice == "Neural Network":
        # Check if binary classification
        n_classes = len(np.unique(y))
        if n_classes == 2:
            # Binary classification
            model = Sequential([
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # No need for to_categorical for binary classification
            history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0, validation_split=0.2)
            
            # Make predictions
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
        else:
            # Multiclass classification
            model = Sequential([
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(n_classes, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train)
            y_test_cat = to_categorical(y_test)
            
            # Train the model
            history = model.fit(X_train_scaled, y_train_cat, epochs=50, batch_size=10, verbose=0, validation_split=0.2)
            
            # Make predictions
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)
            accuracy = accuracy_score(y_test, y_pred)

        # Common evaluation code for both binary and multiclass
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba.reshape(-1, 1) if n_classes == 2 else y_pred_proba)

        return model, accuracy, conf_matrix, class_report, y_test, y_pred_proba, None, logloss, X_train_scaled, y_train, history
    else:
        # Perform hyperparameter tuning with cross-validation if param_grid is provided
        if param_grid:
            try:
                grid_search = GridSearchCV(model_choice, param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
                grid_search.fit(X_train_scaled, y_train)
                best_params = grid_search.best_params_
                model = model_choice.set_params(**best_params)
            except Exception as e:
                st.error(f"Error during grid search: {str(e)}")
                return None
        else:
            model = model_choice
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        return model, accuracy, conf_matrix, class_report, y_test, y_pred_proba, best_params if param_grid else None, logloss, X_train_scaled, y_train, None

# Streamlit app
st.title('Model Training and Evaluation App')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    # Select separator and encoding
    separator = st.selectbox("Select separator", [",", ";", "\t", "|"])
    encoding = st.selectbox("Select encoding", ["utf-8", "ISO-8859-1", "latin1"])
    
    df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
    st.write("Data Preview:")
    st.write(df.head())
    
    # Select target variable
    target_column = st.selectbox("Select the target variable", df.columns)
    
    # Validate target variable
    if target_column not in df.columns:
        st.error("Selected target column not found in dataset")
    else:
        # Select variables to drop
        drop_columns = st.multiselect("Select variables to drop", ["Auto"] + list(df.columns))
        
        # Drop selected columns
        if "Auto" in drop_columns:
            X = df.drop(columns=[target_column])
            X = X.select_dtypes(include=[np.number])
        else:
            X = df.drop(columns=[target_column] + drop_columns)
            X = X.select_dtypes(include=[np.number])
        
        # Ensure that X is not empty after dropping columns
        if X.empty:
            st.error("No numeric columns available after dropping selected columns. Please check your input data and preprocessing steps.")
        else:
            # Fill missing values
            fill_value = st.text_input("Enter value to fill missing values (e.g., 0 or mean):", "0")
            if fill_value.lower() == 'mean':
                try:
                    df = df.fillna(df.mean())
                except TypeError:
                    st.error("Cannot calculate mean for non-numeric columns")
                    st.stop()  # Use st.stop() instead of return
            else:
                try:
                    df = df.fillna(float(fill_value))
                except ValueError:
                    st.error("Invalid fill value. Please enter a number or 'mean'")
                    st.stop()
            
            # Select sample size
            sample_size = st.number_input("Enter sample size (0 for full dataset):", min_value=0, value=0, step=1)
            if sample_size > 0:
                if sample_size > len(df):
                    st.error("Sample size larger than dataset")
                else:
                    # Sample both features and target together
                    sampled_indices = np.random.choice(len(df), size=sample_size, replace=False)
                    df = df.iloc[sampled_indices]
                    X = df.drop(columns=[target_column])
                    y = df[target_column]

            # Splitting the data into features and target
            y = df[target_column]
            
            # Encode the target variable if it's categorical
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Select correlation threshold and test size
            correlation_threshold = st.slider("Select correlation threshold", 0.0, 1.0, 0.82, 0.01)
            test_size = st.slider("Select test size", 0.1, 0.5, 0.3, 0.01)
            
            # Select model
            model_choice = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine", "Neural Network", "XGBoost", "LightGBM"])
            
            # Set hyperparameters
            param_grid = None
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
                param_grid = {
                    'C': st.multiselect('C', [0.01, 0.1, 1, 10, 100], default=[1]),
                    'max_iter': [st.slider("Select max iterations", 100, 3000, 1000, 50)],  # Wrapped in list
                    'penalty': st.multiselect('penalty', ['l1', 'l2'], default=['l2']),
                    'solver': st.multiselect('solver', ['liblinear', 'saga'], default=['saga']),
                    'class_weight': st.multiselect('class_weight', ['balanced', None], default=['balanced'])
                }
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
                param_grid = {
                    'n_estimators': st.multiselect('n_estimators', [50, 100, 200], default=[100]),
                    'max_depth': st.multiselect('max_depth', [None, 10, 20, 30], default=[None]),
                    'min_samples_split': st.multiselect('min_samples_split', [2, 5, 10], default=[2]),
                    'class_weight': st.multiselect('class_weight', ['balanced', 'balanced_subsample', None], default=['balanced'])
                }
            elif model_choice == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
                param_grid = {
                    'n_neighbors': st.multiselect('n_neighbors', [3, 5, 7, 9], default=[5]),
                    'weights': st.multiselect('weights', ['uniform', 'distance'], default=['uniform']),
                    'metric': st.multiselect('metric', ['euclidean', 'manhattan'], default=['euclidean'])
                }
            elif model_choice == "Support Vector Machine":
                model = SVC(probability=True)
                param_grid = {
                    'C': st.multiselect('C', [0.1, 1, 10, 100], default=[1]),
                    'kernel': st.multiselect('kernel', ['linear', 'rbf', 'poly'], default=['rbf']),
                    'class_weight': st.multiselect('class_weight', ['balanced', None], default=['balanced'])
                }
            elif model_choice == "Neural Network":
                model = "Neural Network"
            elif model_choice == "XGBoost":
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                param_grid = {
                    'n_estimators': st.multiselect('n_estimators', [50, 100, 200], default=[100]),
                    'max_depth': st.multiselect('max_depth', [3, 6, 9], default=[6]),
                    'learning_rate': st.multiselect('learning_rate', [0.01, 0.1, 0.2], default=[0.1]),
                    'scale_pos_weight': st.multiselect('scale_pos_weight', [1, 10, 100], default=[1])
                }
            elif model_choice == "LightGBM":
                model = lgb.LGBMClassifier()
                param_grid = {
                    'n_estimators': st.multiselect('n_estimators', [50, 100, 200], default=[100]),
                    'num_leaves': st.multiselect('num_leaves', [31, 50, 100], default=[31]),
                    'learning_rate': st.multiselect('learning_rate', [0.01, 0.1, 0.2], default=[0.1]),
                    'scale_pos_weight': st.multiselect('scale_pos_weight', [1, 10, 100], default=[1])
                }
            
            # Add a start button
            if st.button("Start Training"):
                # Ensure that the DataFrame is not empty and has columns after preprocessing
                if X.empty:
                    st.error("The DataFrame is empty after preprocessing. Please check your input data and preprocessing steps.")
                else:
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    # Train and evaluate the model
                    result = train_and_evaluate_model(X, y, model, param_grid, correlation_threshold, test_size)
                    if result is not None:
                        best_model, accuracy, conf_matrix, class_report, y_test, y_pred_proba, best_params, logloss, X_train_scaled, y_train, history = result
                        st.write(f'Accuracy: {accuracy:.4f}')
                        st.write(f'Log Loss: {logloss:.4f}' if logloss is not None else "Log Loss: N/A")
                        st.write(f'Best Parameters: {best_params}' if best_params is not None else "Best Parameters: N/A")
                        st.write("Confusion Matrix:")
                        st.write(conf_matrix)
                        st.write("Classification Report:")
                        st.write(class_report)
                        
                        # Plot ROC curve if applicable
                        if y_pred_proba is not None:
                            if len(np.unique(y_test)) == 2:
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                auc_score = roc_auc_score(y_test, y_pred_proba)
                            

                                plt.figure(figsize=(10, 8))
                                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
                                plt.plot([0, 1], [0, 1], 'k--')
                                plt.xlim([0.0, 1.0])
                                plt.ylim([0.0, 1.05])
                                plt.xlabel('False Positive Rate')
                                plt.ylabel('True Positive Rate')
                                plt.title('Receiver Operating Characteristic (ROC) Curve')
                                plt.legend(loc='lower right')
                                st.pyplot(plt)
                        else:
                                st.write("ROC curve is only available for binary classification")
                        # Plot feature importance for models that support it
                        if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
                            if model_choice == "Random Forest":
                                feature_importances = best_model.feature_importances_
                            elif model_choice == "XGBoost":
                                feature_importances = best_model.feature_importances_
                            elif model_choice == "LightGBM":
                                feature_importances = best_model.feature_importances_
                            
                            features = X.columns
                            importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
                            importance_df = importance_df.sort_values(by='Importance', ascending=False)
                            
                            plt.figure(figsize=(10, 8))
                            sns.barplot(x='Importance', y='Feature', data=importance_df)
                            plt.title('Feature Importances')
                            st.pyplot(plt)
                        
                            # Plot one of the decision trees for Random Forest
                            if model_choice == "Random Forest":
                                plt.figure(figsize=(20, 10))
                                plot_tree(best_model.estimators_[0], feature_names=features, filled=True, rounded=True, fontsize=10)
                                plt.title('Random Forest - Decision Tree')
                                st.pyplot(plt)
                        
                        # Plot training history for neural network
                        if model_choice == "Neural Network":
                            plt.figure(figsize=(10, 8))
                            plt.plot(history.history['accuracy'], label='Training Accuracy')
                            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                            plt.xlabel('Epoch')
                            plt.ylabel('Accuracy')
                            plt.title('Training and Validation Accuracy')
                            plt.legend()
                            st.pyplot(plt)
                            
                            plt.figure(figsize=(10, 8))
                            plt.plot(history.history['loss'], label='Training Loss')
                            plt.plot(history.history['val_loss'], label='Validation Loss')
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.title('Training and Validation Loss')
                            plt.legend()
                            st.pyplot(plt)
