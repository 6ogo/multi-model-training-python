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
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.utils import to_categorical
from sklearn.tree import plot_tree

# Add error handling for imports
try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError as e:
    st.error(f"Missing required package: {e.name}. Please install it using pip.")

# Function to train and evaluate the model with hyperparameter tuning
def train_and_evaluate_model(X, y, model_choice, param_grid=None, correlation_threshold=0.5, test_size=0.3):
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
        
        return model, accuracy, conf_matrix, class_report, y_test, y_pred_proba, None, logloss, X_train_scaled, X_test_scaled, y_train, history
    
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
        
        return model, accuracy, conf_matrix, class_report, y_test, y_pred_proba, best_params if param_grid else None, logloss, X_train_scaled, X_test_scaled, y_train, None

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
            # First, select target variable
            y = df[target_column]        
            # Initialize X by dropping the target column
            X = df.drop(columns=[target_column])
            
            # Identify numeric and datetime columns
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            datetime_cols = X.select_dtypes(include=['datetime64']).columns
            
            def parse_datetime_column(df, column):
                """
                Attempts to parse a column as datetime using multiple strategies.
                Returns the parsed datetime series or None if parsing fails.
                """
                # List of date formats to try, from most specific to least specific
                date_formats = [
                    '%Y-%m-%d %H:%M:%S.%f',  # 2024-03-27 14:30:00.000
                    '%Y-%m-%d %H:%M:%S',     # 2024-03-27 14:30:00
                    '%Y-%m-%d %H:%M',        # 2024-03-27 14:30
                    '%Y-%m-%d',              # 2024-03-27
                    '%d/%m/%Y %H:%M:%S',     # 27/03/2024 14:30:00
                    '%d/%m/%Y',              # 27/03/2024
                    '%m/%d/%Y %H:%M:%S',     # 03/27/2024 14:30:00
                    '%m/%d/%Y',              # 03/27/2024
                    '%d-%m-%Y',              # 27-03-2024
                    '%Y/%m/%d'               # 2024/03/27
                ]
                
                # Try each format
                for date_format in date_formats:
                    try:
                        return pd.to_datetime(df[column], format=date_format)
                    except ValueError:
                        continue
                
                # If none of the specific formats work, try pandas' flexible parser with error handling
                try:
                    # First try with dayfirst=False (assume American format)
                    return pd.to_datetime(df[column], infer_datetime_format=True, errors='raise')
                except ValueError:
                    try:
                        # If that fails, try with dayfirst=True (assume European format)
                        return pd.to_datetime(df[column], infer_datetime_format=True, dayfirst=True, errors='raise')
                    except ValueError:
                        return None

            # Update the datetime processing section in your code
            datetime_cols = []
            for col in X.columns:
                if col not in numeric_cols:
                    parsed_dates = parse_datetime_column(X, col)
                    if parsed_dates is not None:
                        # Convert to datetime if successful
                        X[col] = parsed_dates
                        datetime_cols.append(col)
                        
                        # Extract datetime features
                        X[f'{col}_year'] = X[col].dt.year
                        X[f'{col}_month'] = X[col].dt.month
                        X[f'{col}_day'] = X[col].dt.day
                        X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                        X[f'{col}_hour'] = X[col].dt.hour
                        X[f'{col}_minute'] = X[col].dt.minute
                        
                        # Add some additional useful features
                        X[f'{col}_quarter'] = X[col].dt.quarter
                        X[f'{col}_is_weekend'] = X[col].dt.dayofweek.isin([5, 6]).astype(int)
                        X[f'{col}_is_month_start'] = X[col].dt.is_month_start.astype(int)
                        X[f'{col}_is_month_end'] = X[col].dt.is_month_end.astype(int)
                        
                        # Drop the original datetime column
                        X = X.drop(columns=[col])
            
            # Select only numeric columns for the final feature set
            X = X.select_dtypes(include=[np.number])
            
            # Remove columns with constant values
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            X = X.drop(columns=constant_cols)
            
            # Handle any remaining non-numeric columns or missing values
            X = X.apply(pd.to_numeric, errors='coerce')
        else:
            # First, select target variable
            y = df[target_column]
            X = df.drop(columns=[target_column] + drop_columns)
            X = X.select_dtypes(include=[np.number])
        
        # Ensure that X is not empty after dropping columns
        if X.empty:
            st.error("No numeric columns available after preprocessing. Please check your input data.")
        else:
            # Fill missing values
            fill_value = st.text_input("Enter value to fill missing values (e.g., 0 or mean):", "0")
            if fill_value.lower() == 'mean':
                try:
                    X = X.fillna(X.mean())
                except TypeError:
                    st.error("Cannot calculate mean for non-numeric columns")
                    st.stop()
            else:
                try:
                    X = X.fillna(float(fill_value))
                except ValueError:
                    st.error("Invalid fill value. Please enter a number or 'mean'")
                    st.stop()
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Select sample size
            sample_size = st.number_input("Enter sample size (0 for full dataset):", min_value=0, value=0, step=1)
            if sample_size > 0:
                if sample_size > len(X):
                    st.error("Sample size larger than dataset")
                else:
                    # Sample both features and target together
                    sampled_indices = np.random.choice(len(X), size=sample_size, replace=False)
                    X = X.iloc[sampled_indices]
                    y = y.iloc[sampled_indices]
            
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
                    # Train and evaluate the model
                    result = train_and_evaluate_model(X, y, model, param_grid, correlation_threshold, test_size)
                    if result is not None:
                        
                        #unpack the models results
                        best_model, accuracy, conf_matrix, class_report, y_test, y_pred_proba, best_params, logloss, X_train_scaled, X_test_scaled, y_train, history = result                        
                        
                        # Display metrics
                        st.write(f'Accuracy: {accuracy:.4f}')
                        st.write(f'Log Loss: {logloss:.4f}' if logloss is not None else "Log Loss: N/A")
                        st.write(f'Best Parameters: {best_params}' if best_params is not None else "Best Parameters: N/A")

                        # Create columns for better layout
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("Confusion Matrix:")
                            st.write(conf_matrix)

                        with col2:
                            st.write("Classification Report:")
                            st.write(class_report)

                        # Visualization options
                        st.subheader("Visualization Options")

                        # General Data Visualization Section
                        st.write("### General Data Visualizations")
                        if st.checkbox("Show Data Distribution Plots"):
                            # Select variables for plotting
                            selected_features = st.multiselect("Select features to plot:", X.columns)
                            if selected_features:
                                plot_type = st.selectbox(
                                    "Select plot type:", 
                                    ["Distribution Plot", "Box Plot", "Violin Plot", "Bar Plot", "Histogram"]
                                )
                                
                                for feature in selected_features:
                                    plt.figure(figsize=(10, 6))
                                    if plot_type == "Distribution Plot":
                                        sns.kdeplot(data=X[feature], hue=y if len(np.unique(y)) < 10 else None)
                                    elif plot_type == "Box Plot":
                                        sns.boxplot(x=y if len(np.unique(y)) < 10 else None, y=X[feature])
                                    elif plot_type == "Violin Plot":
                                        sns.violinplot(x=y if len(np.unique(y)) < 10 else None, y=X[feature])
                                    elif plot_type == "Bar Plot":
                                        sns.barplot(x=y if len(np.unique(y)) < 10 else None, y=X[feature])
                                    elif plot_type == "Histogram":
                                        sns.histplot(data=X[feature], bins=30)
                                    plt.title(f'{plot_type} of {feature}')
                                    plt.xticks(rotation=45)
                                    st.pyplot(plt)

                        if st.checkbox("Show Correlation Analysis"):
                            plt.figure(figsize=(12, 8))
                            sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
                            plt.title('Feature Correlation Heatmap')
                            st.pyplot(plt)

                        if st.checkbox("Show Pairwise Relationships"):
                            selected_vars = st.multiselect(
                                "Select variables for pairplot (max 5):", 
                                X.columns, 
                                max_selections=5
                            )
                            if selected_vars:
                                plot_type = st.selectbox(
                                    "Select visualization type:",
                                    ["Scatter Plot Matrix", "Joint Plot", "Radar Chart"]
                                )
                                
                                if plot_type == "Scatter Plot Matrix":
                                    sns.pairplot(pd.DataFrame(X[selected_vars]), diag_kind='kde')
                                    st.pyplot(plt)
                                elif plot_type == "Joint Plot":
                                    if len(selected_vars) >= 2:
                                        var1, var2 = st.selectbox("Select first variable:", selected_vars), st.selectbox("Select second variable:", selected_vars)
                                        sns.jointplot(data=pd.DataFrame(X[selected_vars]), x=var1, y=var2, kind='reg')
                                        st.pyplot(plt)
                                elif plot_type == "Radar Chart":
                                    # Create radar chart
                                    angles = np.linspace(0, 2*np.pi, len(selected_vars), endpoint=False)
                                    stats = X[selected_vars].mean()
                                    stats = np.concatenate((stats, [stats[0]]))  # completing the loop
                                    angles = np.concatenate((angles, [angles[0]]))  # completing the loop
                                    
                                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                                    ax.plot(angles, stats)
                                    ax.fill(angles, stats, alpha=0.25)
                                    ax.set_xticks(angles[:-1])
                                    ax.set_xticklabels(selected_vars)
                                    plt.title("Radar Chart of Selected Features")
                                    st.pyplot(plt)

                        if st.checkbox("Show Feature Distributions by Target"):
                            selected_feature = st.selectbox("Select feature:", X.columns)
                            plot_type = st.selectbox(
                                "Select plot type:",
                                ["Violin Plot", "Box Plot", "Swarm Plot", "Strip Plot", "Point Plot"]
                            )
                            
                            plt.figure(figsize=(12, 6))
                            if plot_type == "Violin Plot":
                                sns.violinplot(x=y, y=X[selected_feature])
                            elif plot_type == "Box Plot":
                                sns.boxplot(x=y, y=X[selected_feature])
                            elif plot_type == "Swarm Plot":
                                sns.swarmplot(x=y, y=X[selected_feature])
                            elif plot_type == "Strip Plot":
                                sns.stripplot(x=y, y=X[selected_feature])
                            elif plot_type == "Point Plot":
                                sns.pointplot(x=y, y=X[selected_feature])
                            plt.title(f'{plot_type} of {selected_feature} by Target')
                            plt.xticks(rotation=45)
                            st.pyplot(plt)

                        # Model-specific visualizations
                        st.write("### Model-Specific Visualizations")

                        # Common visualizations for all models
                        if st.checkbox("Show Confusion Matrix Heatmap"):
                            plt.figure(figsize=(8, 6))
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                            plt.title('Confusion Matrix Heatmap')
                            st.pyplot(plt)

                        # Prediction Distribution
                        if st.checkbox("Show Prediction Distribution"):
                            plt.figure(figsize=(10, 6))
                            # Use existing predictions
                            if y_pred_proba is not None:
                                sns.histplot(y_pred_proba, bins=30)
                            else:
                                # If probabilities aren't available, use the raw predictions from the model
                                predictions = best_model.predict(X_test_scaled)
                                sns.histplot(predictions, bins=30)
                            plt.title('Distribution of Predictions')
                            st.pyplot(plt)



                        # ROC and PR curves for binary classification
                        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                            if st.checkbox("Show Classification Curves"):
                                curve_type = st.selectbox("Select curve type:", ["ROC Curve", "PR Curve", "Both"])
                                
                                if curve_type in ["ROC Curve", "Both"]:
                                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                    auc_score = roc_auc_score(y_test, y_pred_proba)
                                    
                                    plt.figure(figsize=(10, 8))
                                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
                                    plt.plot([0, 1], [0, 1], 'k--')
                                    plt.xlim([0.0, 1.0])
                                    plt.ylim([0.0, 1.05])
                                    plt.xlabel('False Positive Rate')
                                    plt.ylabel('True Positive Rate')
                                    plt.title('ROC Curve')
                                    plt.legend(loc='lower right')
                                    st.pyplot(plt)
                                
                                if curve_type in ["PR Curve", "Both"]:
                                    from sklearn.metrics import precision_recall_curve, average_precision_score
                                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                                    ap_score = average_precision_score(y_test, y_pred_proba)
                                    
                                    plt.figure(figsize=(10, 8))
                                    plt.plot(recall, precision, label=f'PR curve (AP = {ap_score:.4f})')
                                    plt.xlabel('Recall')
                                    plt.ylabel('Precision')
                                    plt.title('Precision-Recall Curve')
                                    plt.legend(loc='lower left')
                                    st.pyplot(plt)

                        # Model-specific visualizations
                        if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
                            if st.checkbox("Show Tree-Based Model Visualizations"):
                                viz_type = st.selectbox(
                                    "Select visualization type:",
                                    ["Feature Importance", "Feature Importance Heatmap", "Tree Visualization", "Feature Interaction"]
                                )
                                
                                # For the feature importance visualization (around line 510)
                                if viz_type == "Feature Importance":
                                    feature_importances = best_model.feature_importances_
                                    features = X.columns  # Using X.columns instead of undefined features variable
                                    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
                                    importance_df = importance_df.sort_values(by='Importance', ascending=False)
                                    
                                    plt.figure(figsize=(10, 8))
                                    sns.barplot(x='Importance', y='Feature', data=importance_df)
                                    plt.title('Feature Importances')
                                    st.pyplot(plt)

                                elif viz_type == "Feature Importance Heatmap":
                                    plt.figure(figsize=(12, 8))
                                    if model_choice == "Random Forest":
                                        features = X.columns  # Using X.columns instead of undefined features variable
                                        feature_imp_matrix = np.array([tree.feature_importances_ for tree in best_model.estimators_])
                                        sns.heatmap(feature_imp_matrix, xticklabels=features, yticklabels=False, cmap='viridis')
                                        plt.title('Feature Importance Across Trees')
                                        plt.xlabel('Features')
                                        plt.ylabel('Trees')
                                        st.pyplot(plt)
                                
                                elif viz_type == "Tree Visualization" and model_choice == "Random Forest":
                                    plt.figure(figsize=(20, 10))
                                    plot_tree(best_model.estimators_[0], feature_names=X.columns, filled=True, rounded=True, fontsize=10)
                                    plt.title('Sample Decision Tree')
                                    st.pyplot(plt)
                                    
                                elif viz_type == "Feature Interaction":
                                    if len(X.columns) >= 2:
                                        feature1 = st.selectbox("Select first feature:", X.columns)
                                        feature2 = st.selectbox("Select second feature:", X.columns)
                                        
                                        plt.figure(figsize=(10, 8))
                                        # Using y_pred_proba instead of undefined y_pred
                                        sns.scatterplot(x=X[feature1], y=X[feature2], hue=y_pred_proba if y_pred_proba is not None else y_test, palette='viridis')
                                        plt.title(f'Feature Interaction: {feature1} vs {feature2}')
                                        st.pyplot(plt)

                        elif model_choice == "Neural Network":
                            if st.checkbox("Show Neural Network Visualizations"):
                                viz_type = st.selectbox(
                                    "Select visualization type:",
                                    ["Training History", "Learning Curves", "Layer Activations", "Prediction Confidence"]
                                )
                                
                                if viz_type == "Training History":
                                    metrics = st.multiselect(
                                        "Select metrics to plot:",
                                        ["accuracy", "loss"],
                                        default=["accuracy", "loss"]
                                    )
                                    
                                    for metric in metrics:
                                        plt.figure(figsize=(10, 6))
                                        plt.plot(history.history[metric], label=f'Training {metric}')
                                        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                                        plt.xlabel('Epoch')
                                        plt.ylabel(metric.capitalize())
                                        plt.title(f'Training and Validation {metric.capitalize()}')
                                        plt.legend()
                                        st.pyplot(plt)

                                elif viz_type == "Learning Curves":
                                    train_sizes = np.linspace(0.1, 1.0, 10)
                                    train_scores = []
                                    val_scores = []
                                    for size in train_sizes:
                                        idx = int(len(y_train) * size)
                                        history = model.fit(
                                            X_train_scaled[:idx], 
                                            y_train[:idx],
                                            validation_split=0.2,
                                            epochs=5,
                                            verbose=0
                                        )
                                        train_scores.append(history.history['accuracy'][-1])
                                        val_scores.append(history.history['val_accuracy'][-1])
                                    
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(train_sizes, train_scores, label='Training Score')
                                    plt.plot(train_sizes, val_scores, label='Validation Score')
                                    plt.xlabel('Training Set Size')
                                    plt.ylabel('Accuracy')
                                    plt.title('Learning Curves')
                                    plt.legend()
                                    st.pyplot(plt)

                                elif viz_type == "Prediction Confidence":
                                    plt.figure(figsize=(10, 6))
                                    sns.histplot(y_pred_proba, bins=30)
                                    plt.title('Prediction Confidence Distribution')
                                    plt.xlabel('Confidence')
                                    plt.ylabel('Count')
                                    st.pyplot(plt)

                        elif model_choice in ["Logistic Regression", "Support Vector Machine"]:
                            if st.checkbox("Show Linear Model Visualizations"):
                                viz_type = st.selectbox(
                                    "Select visualization type:",
                                    ["Decision Boundary", "Coefficient Analysis", "Prediction Confidence"]
                                )
                                
                                if viz_type == "Decision Boundary":
                                    from sklearn.decomposition import PCA
                                    
                                    pca = PCA(n_components=2)
                                    X_pca = pca.fit_transform(X_train_scaled)
                                    
                                    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                                    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                                    np.arange(y_min, y_max, 0.1))
                                    
                                    Z = best_model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
                                    Z = Z.reshape(xx.shape)
                                    
                                    plt.figure(figsize=(10, 8))
                                    plt.contourf(xx, yy, Z, alpha=0.4)
                                    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.8)
                                    plt.xlabel('First Principal Component')
                                    plt.ylabel('Second Principal Component')
                                    plt.title('Decision Boundary (PCA)')
                                    st.pyplot(plt)
                                    
                                elif viz_type == "Coefficient Analysis" and model_choice == "Logistic Regression":
                                    plt.figure(figsize=(10, 6))
                                    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': best_model.coef_[0]})
                                    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
                                    sns.barplot(data=coef_df, x='Coefficient', y='Feature')
                                    plt.title('Feature Coefficients')
                                    st.pyplot(plt)
                                    
                                elif viz_type == "Prediction Confidence":
                                    plt.figure(figsize=(10, 6))
                                    sns.histplot(y_pred_proba, bins=30)
                                    plt.title('Prediction Confidence Distribution')
                                    plt.xlabel('Confidence')
                                    plt.ylabel('Count')
                                    st.pyplot(plt)
                                    
                        elif model_choice == "K-Nearest Neighbors":
                            if st.checkbox("Show KNN Visualizations"):
                                viz_type = st.selectbox(
                                    "Select visualization type:",
                                    ["Neighbor Distance Distribution", "Decision Regions", "Performance vs K", "Neighbor Density", "Distance Matrix"]
                                )
                                
                                if viz_type == "Neighbor Distance Distribution":
                                    from sklearn.neighbors import NearestNeighbors
                                    nbrs = NearestNeighbors(n_neighbors=2).fit(X_train_scaled)
                                    distances, _ = nbrs.kneighbors(X_train_scaled)
                                    
                                    plt.figure(figsize=(10, 6))
                                    sns.histplot(distances[:, 1], bins=50)
                                    plt.xlabel('Distance to Nearest Neighbor')
                                    plt.ylabel('Count')
                                    plt.title('Distribution of Distances to Nearest Neighbor')
                                    st.pyplot(plt)
                                    
                                elif viz_type == "Decision Regions":
                                    # Only show for 2D data or after PCA
                                    from sklearn.decomposition import PCA
                                    
                                    pca = PCA(n_components=2)
                                    X_pca = pca.fit_transform(X_train_scaled)
                                    
                                    # Train a new KNN model on PCA data
                                    knn_2d = KNeighborsClassifier(**best_model.get_params())
                                    knn_2d.fit(X_pca, y_train)
                                    
                                    # Create mesh grid
                                    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                                    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                                    np.arange(y_min, y_max, 0.1))
                                    
                                    # Make predictions
                                    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                                    Z = Z.reshape(xx.shape)
                                    
                                    plt.figure(figsize=(10, 8))
                                    plt.contourf(xx, yy, Z, alpha=0.4)
                                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.8)
                                    plt.xlabel('First Principal Component')
                                    plt.ylabel('Second Principal Component')
                                    plt.title('KNN Decision Regions (PCA)')
                                    plt.colorbar(scatter)
                                    st.pyplot(plt)
                                    
                                #KNN Performance vs K visualization
                                elif viz_type == "Performance vs K":
                                    k_values = range(1, 21)
                                    train_scores = []
                                    val_scores = []
                                    
                                    for k in k_values:
                                        knn = KNeighborsClassifier(n_neighbors=k)
                                        knn.fit(X_train_scaled, y_train)
                                        train_scores.append(knn.score(X_train_scaled, y_train))
                                        val_scores.append(knn.score(X_test_scaled, y_test))
                                    
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(k_values, train_scores, label='Training Score')
                                    plt.plot(k_values, val_scores, label='Validation Score')
                                    plt.xlabel('Number of Neighbors (k)')
                                    plt.ylabel('Accuracy')
                                    plt.title('Performance vs Number of Neighbors')
                                    plt.legend()
                                    st.pyplot(plt)

                                elif viz_type == "Neighbor Density":
                                    from sklearn.neighbors import KernelDensity
                                    
                                    if len(X.columns) >= 2:
                                        feature1, feature2 = st.multiselect("Select two features for density plot:", X.columns, max_selections=2)
                                        if len(feature1) == 2:
                                            X_subset = np.c_[X[feature1[0]], X[feature1[1]]]
                                            
                                            kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
                                            kde.fit(X_subset)
                                            
                                            # Create mesh grid
                                            x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
                                            y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
                                            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                                            np.linspace(y_min, y_max, 100))
                                            
                                            # Calculate density
                                            Z = kde.score_samples(np.c_[xx.ravel(), yy.ravel()])
                                            Z = Z.reshape(xx.shape)
                                            
                                            plt.figure(figsize=(10, 8))
                                            plt.contourf(xx, yy, np.exp(Z), levels=20)
                                            plt.colorbar(label='Density')
                                            plt.scatter(X_subset[:, 0], X_subset[:, 1], c='white', alpha=0.5, s=1)
                                            plt.xlabel(feature1[0])
                                            plt.ylabel(feature1[1])
                                            plt.title('Neighbor Density Plot')
                                            st.pyplot(plt)
                                
                                elif viz_type == "Distance Matrix":
                                    # Calculate and display distance matrix for a sample of points
                                    sample_size = min(100, len(X_train_scaled))
                                    sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
                                    X_sample = X_train_scaled[sample_indices]
                                    
                                    from sklearn.metrics.pairwise import euclidean_distances
                                    dist_matrix = euclidean_distances(X_sample)
                                    
                                    plt.figure(figsize=(10, 8))
                                    sns.heatmap(dist_matrix, cmap='viridis')
                                    plt.title('Distance Matrix Heatmap (Sample)')
                                    plt.xlabel('Sample Index')
                                    plt.ylabel('Sample Index')
                                    st.pyplot(plt)
