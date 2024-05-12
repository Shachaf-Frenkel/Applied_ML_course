# from pipelines import *
from classes import *

import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold,StratifiedGroupKFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier,RUSBoostClassifier
import optuna
from sklearn.model_selection import cross_validate,GridSearchCV, learning_curve, LearningCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')
X_train = pd.read_pickle('data_cv_gan_boosting_models.pkl')
# df_features[['admission_type_id','discharge_disposition_id','admission_source_id']] = df_features[['admission_type_id','discharge_disposition_id','admission_source_id']].astype('object')

y_train = pd.read_pickle('labels_cv.pkl')
# df_features = df_features.drop(columns = 'readmitted')
folds_idx = pd.read_pickle('folds_idx.pickle')
folds_idx_synt = pd.read_pickle('folds_idx_synt.pickle')
print("done")
n_over_samp = 97
n_under_samp = 54446


folds_idx_synt_flat = [item for array in folds_idx_synt for item in array]
folds_idx_synt_samp = [np.random.choice(idx,n_over_samp,replace = False) for idx in folds_idx_synt]
folds_idx_synt_flat_samp = [item for array in folds_idx_synt_samp for item in array]
idx_drop_synt = list(set(folds_idx_synt_flat)-set(folds_idx_synt_flat_samp))

from imblearn.under_sampling import EditedNearestNeighbours

majority_class = 'NO'
fold = folds_idx.copy()
for i,tup in enumerate(folds_idx) :
    def custom_strategy(y):
    # Custom function to specify the sampling strategy
        return {majority_class: n_under_samp}  # Sample 100 samples for class1 and 150 samples for class2

    # display(tup)
    # enn = EditedNearestNeighbours(sampling_strategy = custom_strategy)

    undersample = RandomUnderSampler(sampling_strategy={majority_class:n_under_samp},random_state=42)
    y = y_train[tup[0]]
    X = X_train.iloc[tup[0],:]
    undersample.fit_resample(X,y)
    idx = undersample.sample_indices_ + tup[0][0]
    idx_drop = list(set(list(set(y[y == majority_class].index) - set(idx)) + idx_drop_synt))
    fold[i] = (np.array(list(set(tup[0]) - set(idx_drop))),tup[1])
    print(f"fold {i}")
    print('n majority:')
    print(sum(y_train[fold[i][0]] == 'NO'))
    print('n minority')
    print(sum(y_train[fold[i][0]]=='<30'))

with_gan = True

indices = []
for c in X_train.columns:
    col_type = X_train[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X_train[c] = X_train[c].astype('category')
# for c in X_train.columns:
#     col_type = X_train[c].dtype
#     if col_type == 'object' or col_type.name == 'category':
#         X_new = X_train.drop([c],axis=1)
#         X_train= X_new

#'nateglinide', 'chlorpropamide', 'tolbutamide', 'acarbose', 'miglitol','troglitazone','tolazamide','glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone'
# categorical_features=['race', 'gender', 'medical_specialty',
#        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin',
#        'repaglinide','glimepiride',
#        'acetohexamide', 'glipizide', 'glyburide',
#        'pioglitazone', 'rosiglitazone','insulin','change', 'diabetesMed','admission_source_id','discharge_disposition_id']
categorical_features = list(X_train.select_dtypes(include=['category', 'object']))
for column_name in categorical_features:
    index = X_train.columns.get_loc(column_name)
    indices.append(index)
#0.627988 n_estimators=1000,categorical_feature=indices,boosting_type='dart',num_iterations= 1000,objective="binary",num_leaves=24,max_depth=6,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True ,min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7,reg_alpha=2.6*(10**(-2)),reg_lambda=5.04*(10**(-2)),cat_l2 = 2.6*(10**(-2)))
#max_depth=8,num_leaves=20 0.625,0.6719
#lgbm = lgb.LGBMClassifier(n_estimators=1000,categorical_feature=indices,boosting_type='gbdt',num_iterations= 1000,objective="binary",num_leaves=24,max_depth=6,learning_rate=0.01,min_data_in_leaf=24,reg_alpha=2.6*(10**(-2)),reg_lambda=5.04*(10**(-2)),cat_l2 = 2.6*(10**(-2)) ,min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7,is_unbalance=True) 0.6278
#(boosting_type='gbdt',num_iterations= 1000,objective="binary",num_leaves=9,max_depth=4,learning_rate=0.01,min_data_in_leaf=48,reg_alpha=2.6*(10**(0)),reg_lambda=5.04*(10**(0)),min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7,scale_pos_weight=2.132*(10**-1)) 0.62628
#lgbm = lgb.LGBMClassifier(max_depth=8,num_leaves=20,n_estimators=1000,categorical_feature=indices,is_unbalance=True,num_iterations= 1000,learning_rate=0.01) 0.62628
#n_estimators=1000,categorical_feature=indices,boosting_type='dart',num_iterations= 1000,objective="binary",num_leaves=24,max_depth=6,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True 0.6263
#n_estimators=1000,categorical_feature=indices,boosting_type='dart',num_iterations= 1000,objective="binary",num_leaves=24,max_depth=6,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True,reg_alpha=2.6*(10**(-2)),reg_lambda=5.04*(10**(-2)),cat_l2 = 2.6*(10**(-2)) ,min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7,   0.6277
lgbm = lgb.LGBMClassifier(categorical_feature=indices,objective="binary",is_unbalance=True,num_leaves=10)
#lgbm = lgb.LGBMClassifier(max_bin=242,n_estimators=217,categorical_feature=indices,boosting_type='dart',objective="binary",num_leaves=53,max_depth=9,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True , feature_fraction = 0.5992184316846285,min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7,reg_alpha=2.6*(10**(-2)),reg_lambda=5.04*(10**(-2)),cat_l2 = 2.6*(10**(-2)),num_iterations=1000)

#tree = BalancedRandomForestClassifier(random_state=42)
if with_gan:
    cv = fold
    results = cross_validate(lgbm, X_train, y_train, cv=cv, return_train_score=True, scoring='balanced_accuracy')
    # train_size_abs, train_scores, test_scores = learning_curve(lgbm, X_train, y_train, cv=cv, scoring='balanced_accuracy',error_score=np.nan)
    # display = LearningCurveDisplay(train_sizes=train_size_abs,train_scores = train_scores, test_scores = test_scores, score_name = "balanced_accuracy")
    # display.plot()
    # plt.show()

else:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(lgbm, X_train, y_train, cv=cv, return_train_score=True, scoring='balanced_accuracy',
                                 groups=patient_nbr)
#lgbm = lgb.LGBMClassifier(feature_fraction=feature_fraction,max_bin = max_bin,n_estimators=n_estimators,categorical_feature=indices,boosting_type='dart',num_iterations= 1000,objective="binary",num_leaves=num_leaves,max_depth=max_depth,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True,reg_alpha=2.6*(10**(-2)),reg_lambda=5.04*(10**(-2)),cat_l2 = 2.6*(10**(-2)) ,min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7)
#[I 2024-05-12 01:46:31,231] Trial 60 finished with value: 0.6294308055356377 and parameters: {'under_samp': 54446, 'n_over_samp': 97, 'max_bin': 242, 'feature_fraction': 0.5992184316846285, 'num_leaves': 53, 'max_depth': 9, 'n_estimators': 217
#Trial 49 finished with value: 0.6269003924690637 and parameters: {'under_samp': 67056, 'n_over_samp': 153, 'max_bin': 416, 'feature_fraction': 0.46096105692005374, 'num_leaves': 14, 'max_depth': 8, 'n_estimators': 1042}. Best is trial 44 with value: 0.6280553853666783.
# print classifier accuracy
print('train mean accuracy :')
print(results['train_score'].mean())
print('test mean accuracy :')
print(results['test_score'].mean())
# test_scores.append(results)

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

# # for grid search:
model = lgb.LGBMClassifier(categorical_feature=indices,is_unbalance=True)


num_leaves_max_depth = {'num_leaves': range(5, 100, 5),'max_depth': range(0,20,1) }
gsearch = GridSearchCV(param_grid=num_leaves_max_depth, estimator=model,
                           scoring='balanced_accuracy', cv=fold)
gsearch.fit(X_train, y_train)
print(gsearch.best_params_)
print('balanced accuracy - num_leaves_max_depth', gsearch.best_score_)

from imblearn.under_sampling import EditedNearestNeighbours

# 1. Define an objective function to be maximized.
def objective_1(trial,X_train,y_train,folds_idx,folds_idx_synt):
    # 2. Suggest values for the hyperparameters using a trial object.
    count_classes = pd.DataFrame(y_train).value_counts()
    majority_class = count_classes.idxmax()[0]
    # n_under_samp = trial.suggest_int('under_samp', 10000, 70000)
    # n_over_samp = trial.suggest_int('n_over_samp', 0, 1000)
    # max_bin = trial.suggest_int('max_bin',100,500)
    # feature_fraction = trial.suggest_float('feature_fraction', 0.4, 0.9)
    # num_leaves = trial.suggest_int('num_leaves',2,80)
    # max_depth = trial.suggest_int('max_depth',1,10)
    # n_estimators = trial.suggest_int('n_estimators', 1, 1500)

    n_under_samp = 54446
    n_over_samp = 97
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.5)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5,500)
    reg_alpha = trial.suggest_float('reg_alpha',10 ** (-4),10**(1))
    reg_lambda = trial.suggest_float('reg_lambda', 10 ** (-4), 10 ** (1))
    cat_l2 = trial.suggest_float('cat_l2', 10 ** (-4), 10 ** (1))
    min_split_gain = trial.suggest_float('min_split_gain',10 ** (-4), 10 ** (1))
    bagging_fraction = trial.suggest_float('bagging_fraction',10 ** (-4), 0.999)
    bagging_freq = trial.suggest_int('min_data_in_leaf', 1,100)
    num_iterations = trial.suggest_int('num_iterations', 100, 10000)


    folds_idx_synt_flat = [item for array in folds_idx_synt for item in array]
    folds_idx_synt_samp = [np.random.choice(idx,n_over_samp,replace = False) for idx in folds_idx_synt]
    folds_idx_synt_flat_samp = [item for array in folds_idx_synt_samp for item in array]
    idx_drop_synt = list(set(folds_idx_synt_flat)-set(folds_idx_synt_flat_samp))


    majority_class = 'NO'
    fold = folds_idx.copy()
    for i,tup in enumerate(folds_idx) :

    # display(tup)
        # enn = EditedNearestNeighbours(n_neighbors = n_neighbors)

        undersample = RandomUnderSampler(sampling_strategy={majority_class:n_under_samp},random_state=42)
        y = y_train[tup[0]]
        X = X_train.iloc[tup[0],:]
        undersample.fit_resample(X,y)
        idx = undersample.sample_indices_ + tup[0][0]
        idx_drop = list(set(list(set(y[y == majority_class].index) - set(idx)) + idx_drop_synt))
        fold[i] = (np.array(list(set(tup[0]) - set(idx_drop))),tup[1])
    # over_samp = trial.suggest_float('over_samp', 0, 0.5,step = 0.01)
        # undersample = RandomUnderSampler(sampling_strategy={majority_class:under_samp},random_state=42)
        # X_train, y_train = undersample.fit_resample(X_train,y_train)
        # patient_nbr = patient_nbr[X_train.index].reset_index(drop=True)
        # X_train.reset_index(drop=True,inplace=True)
    # max_depth = trial.suggest_int('max_depth', 10, 20)
    # n_estimators = trial.suggest_int('n_estimators', 150, 300)
    #max_bin=242,n_estimators=217,categorical_feature=indices,boosting_type='dart',objective="binary",num_leaves=53,max_depth=9,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True , feature_fraction = 0.5992184316846285
    # best:     lgbm = lgb.LGBMClassifier(feature_fraction=0.5992184316846285,max_bin = 242,n_estimators=217,categorical_feature=indices,boosting_type='dart',objective="binary",num_leaves=53,max_depth=9,learning_rate=0.01,min_data_in_leaf=24,is_unbalance=True,reg_alpha=2.6*(10**(-2)),reg_lambda=5.04*(10**(-2)),cat_l2 = 2.6*(10**(-2)) ,min_split_gain=1.246*(10**-4),bagging_fraction=9.428*(10**(-1)),bagging_freq=7)
    lgbm = lgb.LGBMClassifier(num_iterations = num_iterations, feature_fraction=0.5992184316846285,max_bin = 242,n_estimators=217,categorical_feature=indices,boosting_type='dart',objective="binary",num_leaves=53,max_depth=9,learning_rate=learning_rate,min_data_in_leaf=min_data_in_leaf,is_unbalance=True,reg_alpha=reg_alpha,reg_lambda=reg_lambda,cat_l2 = cat_l2 ,min_split_gain=min_split_gain,bagging_fraction=bagging_fraction,bagging_freq=bagging_freq)
    #cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    results = cross_validate(lgbm,X_train,y_train,cv = fold,return_train_score = True,scoring = 'balanced_accuracy')

    # score = score_model(model, X_train, y_train, X_train_SMOTE,y_train_SMOTE,patient_nbr,over_sample_factor=over_samp)
    accuracy = results['test_score'].mean()
    return accuracy

n_trials=50
optuna.logging.set_verbosity(optuna.logging.INFO)

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
objective_with_args = lambda trial: objective_1(trial, X_train, y_train,folds_idx,folds_idx_synt)

study.optimize(objective_with_args, n_trials=n_trials)

