from classes import *
import matplotlib.pyplot as plt
import time 
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
import optuna
from sklearn.model_selection import cross_validate,GridSearchCV
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb

from imblearn.under_sampling import RandomUnderSampler
optuna.logging.set_verbosity(optuna.logging.INFO)
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve, auc
warnings.filterwarnings('ignore')

def find_cat_indecies(X_train):
            indices = []
            for c in X_train.columns:
                col_type = X_train[c].dtype
                if col_type == 'object' or col_type.name == 'category':
                    X_train[c] = X_train[c].astype('category') 
            categorical_features = list(X_train.select_dtypes(include=['category', 'object']))
            for column_name in categorical_features:
                index = X_train.columns.get_loc(column_name)
                indices.append(index)
            return X_train,indices

def under_over_sample(X_train,y_train,folds_idx,folds_idx_synt,n_over_samp,n_under_samp,verbose = False) :
    """ function that perform over sample from the manority class and/or under sample from the majority
    
    input:
    X_train : pd.DataFrame of featrues for cross validation
    y_train : one columns of labels
    folds_idx : list of tuples, each element in the list is a fold in the cross validation
    the first elemnt in each tuple is the indecies of the train set and the second element in the tuple
     is the indecies for the vakidation set.
    folds_idx_synt : the indecies of the syntetic example in X_train.
    n_over_samp : number of synthetic data point from the manority class
    n_under_samp : number of samples to sample from the majority class in each fold

    output :
    fold : sampe as folds_idx but after oversample and undersample


      """
    majority_class = 'NO'
    fold = folds_idx.copy()
    # randomly sample n_over_samp synthetic data points
    folds_idx_synt_flat = [item for array in folds_idx_synt for item in array] 
    folds_idx_synt_samp = [np.random.choice(idx,n_over_samp,replace = False) for idx in folds_idx_synt]
    folds_idx_synt_flat_samp = [item for array in folds_idx_synt_samp for item in array] 
    idx_drop_synt = list(set(folds_idx_synt_flat)-set(folds_idx_synt_flat_samp))
    
    # perform under sampling of the majority class
    for i,tup in enumerate(folds_idx) :
        undersample = RandomUnderSampler(sampling_strategy={majority_class:n_under_samp},random_state=42)
        y = y_train[tup[0]]
        X = X_train.iloc[tup[0],:]
        undersample.fit_resample(X,y)
        idx = undersample.sample_indices_ + tup[0][0]
        idx_drop = list(set(list(set(y[y == majority_class].index) - set(idx)) + idx_drop_synt))
        fold[i] = (np.array(list(set(tup[0]) - set(idx_drop))),tup[1])
        if verbose :
            print(f"fold {i}")
            print('n majority:')
            print(sum(y_train[fold[i][0]]=='NO'))
            print('n minority')
            print(sum(y_train[fold[i][0]]=='<30'))
    return fold

def objective_1(model,trial,X_train,y_train,folds_idx,folds_idx_synt):
    n_under_samp = trial.suggest_int('n_under_samp', 10000, 50000)
    n_over_samp = trial.suggest_int('n_over_samp', 0, 0)
    folds_idx = under_over_sample(X_train,y_train,folds_idx,folds_idx_synt,n_over_samp = n_over_samp,n_under_samp = n_under_samp,verbose = False) 
    results = cross_validate(model,X_train,y_train,cv = folds_idx,return_train_score = True,scoring = 'balanced_accuracy')
    accuracy =  results['test_score'].mean()
    return accuracy



def objective_2(model,trial,X_train,y_train,folds_idx,model_name):
    
    if model_name == 'random_forest' :
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        min_samples_split = trial.suggest_int('min_samples_split', 2,10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1,10)
        bootstrap = trial.suggest_categorical('bootstrap', [True,False])
        n_estimators = trial.suggest_int('n_estimators', 100,300)
        max_depth = trial.suggest_int('max_depth', 10,20)
        class_weight = trial.suggest_categorical('class_weight', ['balanced','balanced_subsample'])
        params = {'max_depth' : max_depth,'n_estimators' : n_estimators,'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf,'class_weight' : class_weight,
              'bootstrap' : bootstrap,'criterion' : criterion}
    
    if model_name == 'catBoost':
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 1)
        learning_rate = trial.suggest_int('learning_rate', 0.01, 1)
        boosting_type = trial.suggest_categorical('boosting_type',['Ordered','Plain'])
        auto_class_weight = trial.suggest_categorical('class_weight', ['balanced','balanced_subsample'])
        depth = trial.suggest_int('depth', 3, 10)
        iterations = trial.suggest_int('iterations', 800, 1500)
        params = {
              'auto_class_weight' : auto_class_weight,'learning_rate':learning_rate,'boosting_type':boosting_type,
              'bagging_temperature' : bagging_temperature,'depth' : depth,'iterations':iterations}
    
    model.set_params(**params)
    # model.set_params(**params)
    results = cross_validate(model,X_train,y_train,cv = folds_idx,return_train_score = True,scoring = 'balanced_accuracy')
    accuracy =  results['test_score'].mean()
    return accuracy

def evaluate_model_with_seeds(model,X_train, y_train, X_test, y_test):
    results = []
    for seed in range(1,1):
        model.set_params(**{'random_seed' :seed})
        
        # Train the model
        model.fit(X_train, y_train)

        # Predict the labels for the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Calculate evaluation metrics
        confusion_matrix(y_test, y_pred)
        accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary',pos_label='<30')
        recall = recall_score(y_test, y_pred, average='binary',pos_label='<30')
        f1 = f1_score(y_test, y_pred, average='binary',pos_label='<30')
        # Append the results for this seed
        results.append({'Seed': seed,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        })
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    return results_df








if __name__ == "__main__":
    params = {'random_state' : 42}
    start_time = time.time() 
    tune = False  
    evaluate = True 
    model_name = 'catBoost' 
    n_over_samp = 0
    n_under_samp = 40000
    n_optuna_trials= 30
    if evaluate & (model_name =='random_forest') :
        params = {'n_estimators' : 287,
                'max_depth' :15,
                'criterion' :'entropy',
                'min_samples_split' : 3,
                'min_samples_leaf' : 1,
                'bootstrap':True,
                'class_weight' : 'balanced_subsample' 
                }
        params = {'max_depth': 16, 'n_estimators': 114,'random_state': 42}
    if evaluate & (model_name =='lightGBM') :
        params = {'objective':"binary",'is_unbalance':True,'num_leaves':25,'max_depth':15,'boosting_type':'dart'}
    
    if evaluate & (model_name =='catBoost') :
        params = {'iterations':1000, 'learning_rate':0.08, 'depth':6, 'auto_class_weights':'Balanced','boosting_type':'Ordered'}
    
    if model_name == 'random_forest' : 
        X_train = pd.read_pickle('data_cv_gan_rf.pkl')
        model = BalancedRandomForestClassifier()
        model.set_params(**params)

    if model_name in ['catBoost','lightGBM'] :
            X_train = pd.read_pickle('data_cv_gan_boosting_models.pkl')

    if model_name == 'lightGBM' :
        X_train, ind = find_cat_indecies(X_train)

        model = lgb.LGBMClassifier(categorical_feature=ind)
        model.set_params(**params)

    if model_name == 'catBoost':
        cat_features_indices = list(X_train.select_dtypes(include=['object']))
        X_train = X_train.fillna('NaN')
        model = CatBoostClassifier(iterations = 5,cat_features=cat_features_indices)
        model.set_params(**params)
    y_train = pd.read_pickle('labels_cv.pkl')
    folds_idx = pd.read_pickle('folds_idx.pickle')
    folds_idx_synt = pd.read_pickle('folds_idx_synt.pickle')
    # folds_idx = under_over_sample(X_train,y_train,folds_idx,folds_idx_synt,n_over_samp,n_under_samp,verbose = False) 


    
    # results = cross_validate(model,X_train,y_train,cv = folds_idx,return_train_score = True,scoring = 'balanced_accuracy')

    #print classifier accuracy
    # print('cross-validation with default parameters :')
    # print('train mean balanced accuracy :')
    # print(results['train_score'].mean())
    # print('test mean balanced accuracy :')
    # print(results['test_score'].mean())

    if tune :
     # first grid search

        params = {'random_state' : 42}
        max_depth = {'max_depth':range(3,4,1)}
        gsearch = GridSearchCV(param_grid=max_depth, estimator=model,
        scoring='balanced_accuracy', cv=folds_idx)
        gsearch.fit(X_train,y_train)
        params.update(gsearch.best_params_)
        model.set_params(**params)
        print(gsearch.best_params_)
        print('balanced accuracy - max_depth', gsearch.best_score_)

    # second grid search

        n_estimators = {'n_estimators':range(50,51,1)}
        gsearch = GridSearchCV(param_grid=n_estimators, estimator=model,
        scoring='balanced_accuracy', cv=folds_idx)
        gsearch.fit(X_train,y_train)
        params.update(gsearch.best_params_)
        model.set_params(**params)
        print(gsearch.best_params_)
        print('balanced accuracy - n_estimators', gsearch.best_score_)


# first OPTUNA for under and over sampling
        study = optuna.create_study(direction='maximize')
        objective_with_args = lambda trial: objective_1(model,trial, X_train, y_train,folds_idx,folds_idx_synt)
        study.optimize(objective_with_args, n_trials=n_optuna_trials)
        best_params = study.best_params
        folds_idx = under_over_sample(X_train,y_train,folds_idx,folds_idx_synt,n_over_samp = best_params['n_over_samp'],n_under_samp = best_params['n_under_samp'],verbose = True) 
        print('best params', params)
        print('best trial', study.best_trial)
        optuna.visualization.plot_optimization_history(study)
        plt.show()
# second OPTUNA for under and over sampling

        study = optuna.create_study(direction='maximize')
        objective_with_args = lambda trial: objective_2(model,trial, X_train, y_train,folds_idx,model_name)
        study.optimize(objective_with_args, n_trials=n_optuna_trials)
        best_params = study.best_params
        params.update(best_params)
        optuna.visualization.plot_optimization_history(study)

        print('best params', params)
        print('best trial', study.best_trial)

    if evaluate :
        if model_name == 'random_forest' : 
            X_train = pd.read_pickle('train_set_after_preprocessing_rf.pkl')
            X_test = pd.read_pickle('test_set_after_preprocessing_rf.pkl')
        if model_name in ['catBoost','lightGBM'] : 
            X_train = pd.read_pickle('train_set_after_preprocessing_boosting_models.pkl')
            X_test = pd.read_pickle('test_set_after_preprocessing_boosting_models.pkl')
        if model_name == 'lightGBM' :
            X_train,ind = find_cat_indecies(X_train)
            X_test,ind = find_cat_indecies(X_test)
            model = lgb.LGBMClassifier(categorical_feature=ind)
        if model_name == 'catBoost' :
                cat_features_indices = list(X_train.select_dtypes(include=['object']))
                X_train = X_train.fillna('NaN')
                X_test = X_test.fillna('NaN')
                params['cat_features'] = cat_features_indices
        model.set_params(**params)
        y_test = pd.read_pickle('labels_test_set.pkl') 
        y_train = pd.read_pickle('labels_train_set.pkl')
        undersample = RandomUnderSampler(sampling_strategy={'NO':n_under_samp},random_state=42)
        X_train,y_train = undersample.fit_resample(X_train,y_train)


        print(y_test.value_counts())
        results_df = evaluate_model_with_seeds(model, X_train, y_train, X_test, y_test)
        # print(results_df.drop(columns = 'Seed').mean())
        print(results_df)
        results_df.to_csv(f'results_df_{model_name}.csv')
        model.fit(X_train, y_train)
        # Predict the labels for the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)


    # Classification Report
        class_report = classification_report_imbalanced(y_test, y_pred)

    # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,0], pos_label='<30')
        roc_auc = auc(fpr, tpr)

    # Plotting
        plt.figure(figsize=(10, 5))

    # Confusion Matrix
        plt.subplot(1, 2, 1)
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks([0, 1], ['YES', 'NO'])
        plt.yticks([0, 1], ['YES', 'NO'])
        for i in range(conf_matrix.shape[0]):
             for j in range(conf_matrix.shape[1]):
                 plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

    # Classification Report
        plt.subplot(1, 2, 2)
        plt.text(0.01, 0.5, class_report, {'fontsize': 12}, fontfamily='monospace')
        plt.title('Classification Report',fontsize= 15)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate',fontsize = 15)
        plt.ylabel('True Positive Rate',fontsize = 15)
        plt.title('Receiver Operating Characteristic (ROC) Curve',fontsize = 20)
        plt.legend(loc='lower right',fontsize = 15)
        plt.show()

        # Extract feature importances
        feature_importances = model.feature_importances_

        # Create a DataFrame to display feature importances
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to display most important features at the top
        plt.show()
        
    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Script running time {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))




