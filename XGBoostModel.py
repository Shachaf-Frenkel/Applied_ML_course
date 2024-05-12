#XG Boost Model 
!pip install xgboost
!pip install optuna

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score, balanced_accuracy_score, roc_curve, auc, classification_report_imbalanced
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold,StratifiedGroupKFold, cross_val_score, train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier,RUSBoostClassifier
import optuna
import learn.model_selection import cross_validate,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')
#from split_data_for_cv import split_data_for_cv

# from pipelines import *
from classes import *

X_train = pd.read_pickle('/content/data_cv_gan_boosting_models.pkl')
y_train = pd.read_pickle('labels_cv_gan.pkl')
folds_idx = pd.read_pickle('folds_idx.pickle')
folds_idx_synt = pd.read_pickle('folds_idx_synt.pickle')

a = LabelEncoder()
y_train = a.fit_transform(y_train)

# Identify categorical and numerical columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['int', 'float']).columns.tolist()

# Create transformers for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Adjust the OneHotEncoder in the categorical_transformer pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Ensure dense output
])

# Combine into a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough')
# Set up the full pipeline (preprocessing + XGBoost model)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False))
])

# Using pre-defined folds for training and evaluation
for i, (train_idx, test_idx) in enumerate(folds_idx):
    X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]  # Adjust if y_train is a DataFrame

    # Train the model using the fold data
    model_pipeline.fit(X_fold_train, y_fold_train)

    # Evaluate the model
    y_pred = model_pipeline.predict(X_fold_test)
    print(f"Results for fold {i + 1}:")
    print("Accuracy:", accuracy_score(y_fold_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_fold_test, y_pred))
    print("Classification Report:\n", classification_report(y_fold_test, y_pred))

print("Accuracy:", accuracy_score(y_fold_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_fold_test, y_pred))
print("Classification Report:\n", classification_report(y_fold_test, y_pred))

# Identify categorical columns
categorical_cols = [col for col in X_fold_train.columns if X_fold_train[col].dtype == 'object']

# Define the ColumnTransformer; apply OneHotEncoding to the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'  # keep all other columns untouched
)

# Create a pipeline that includes preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False))
])
model_pipeline.fit(X_fold_train, y_fold_train)
y_test_pred = model_pipeline.predict(X_fold_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_fold_test, y_test_pred)
print(conf_matrix)

# Metrics
print('Balanced accuracy: {:.2f}'.format(balanced_accuracy_score(y_fold_test, y_test_pred)))
print('Precision: {:.2f}'.format(precision_score(y_fold_test, y_test_pred, pos_label=0)))
print('Recall: {:.2f}'.format(recall_score(y_fold_test, y_test_pred, pos_label=0)))
print('F1: {:.2f}'.format(f1_score(y_fold_test, y_test_pred, pos_label=0)))

balanced_accuracies = []
for i, (train_idx, test_idx) in enumerate(folds_idx):
    X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]  # Works if X_train is a DataFrame
    y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]  # Standard numpy indexing for numpy arrays

    model_pipeline.fit(X_fold_train, y_fold_train)
    y_pred = model_pipeline.predict(X_fold_test)

    # Calculating balanced accuracy
    bal_acc = balanced_accuracy_score(y_fold_test, y_pred)
    balanced_accuracies.append(bal_acc)

    print(f"Fold {i+1} Balanced Accuracy: {bal_acc}")
