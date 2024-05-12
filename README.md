# Predicting Diabetes Patients Early Readmission using ML models
Roni Weinfeld, Maya Sheffer, Thea Meimoun and Shachaf Frenkel

## Project Description
This project was conducted as part of the course Applied Machine Learning for Life Sciences, Weizmann Insitute of Science - course lecturer Ortal Dayan.
This machine learning project aimed at forecasting patient readmission rates utilizing the 'Diabetes 130-US Hospitals for Years 1999-2008'. 
The performance of four ensemble algorithms: LightGBM, XGBoost, RandomForest, and CatBoost, is compared within this work.

## Code Overview
The code consists of three main files: classes.py, pre_processing.py, and hyperparameter_tuning_and_evaluation.py, each serving distinct purposes in the framework of the project.

In **classes.py,** custom transformer classes are defined using sklearn base classes to match sklearn pipelines and offer flexibility in data preprocessing, tailored to the specific requirements of the dataset. These transformers include dropping features based on variance, grouping of similar categories, and more.

**pre_processing.py** contains different pipelines that perform the preprocessing in steps. The script can execute preprocessing for different uses - (1) basic transformation and data handling for generating synthetic data using GAN and SMOTE algorithms, (2) advanced preprocessing after adding the synthetic data and combining it into one data frame ready for cross-validation. Option to use different steps depending on the model's requirements, (3) preparing the data for final evaluation over the test set.

**CV_with_synt_data.py** script splits the dataset into training sets and validation sets while ensuring that patients are either in the training set or the testing set but not in both. After splitting the dataset, the code synthesizes new examples using the GAN or SMOTE base in the training folds. The output of the code is a new dataset that contains 10 folds with synthetic training examples. This part of the code was run with COLAB’s GPU. T

hen the last script, **model_training_and_evaluation.py**, implements the hyper-parameter tuning using grid search and OPTUNA and evaluates the results over different seeds, on BalancedRandomForest and catBoost models. 

In addition, we did hyper-parameter tuning for lightGBM in **final_cv_LGBM.py**, and cross-validate XGBoost in **XGBoostModel.py**.

