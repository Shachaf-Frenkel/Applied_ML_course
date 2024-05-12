from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn import set_config
from classes import *  # Import custom classes

# Configure sklearn to transform output to pandas DataFrame
set_config(transform_output="pandas")

preprocess_for_gan_smote = False
preprocess_for_random_forest = False
preprocess_for_boosting_models = True

#load and orgenize train set
df_train = pd.read_csv('train_set_new_seed.csv',index_col=0).copy().drop(columns=['encounter_id','encounter_id','payer_code','weight'])

#split target and features and unique patient number
df_features_train = df_train.copy().drop(columns =  ['readmitted','patient_nbr']).reset_index(drop = True)
labels = df_train['readmitted'].reset_index(drop = True)
labels[labels == '>30'] = 'NO'
patient_nbr = df_train['patient_nbr'].reset_index(drop = True)

#load and orgenize test set
df_test = pd.read_csv('test_set_new_seed.csv',index_col=0).copy().drop(columns=['encounter_id','encounter_id','payer_code','weight'])
#split target and features
df_features_test = df_test.copy().drop(columns =  ['readmitted','patient_nbr']).reset_index(drop = True)
labels_test = df_test['readmitted'].reset_index(drop = True)
labels_test[labels_test == '>30'] = 'NO'


    # pre-process the raw data in order to implemant GAN and SMOTE and 
    # create cross-validation balanced folds with synthetic data samples
preprocessing_pipe_gan = Pipeline(steps=[
    ('manual_transform', ManualTranform()),
    ('drop_catFet_by_variance', DropFeaturesByVariance(th = 99)),
    ('Drop_children_patiant', DropYoung())
])
df_gan = preprocessing_pipe_gan.fit_transform(df_features_train)
patient_nbr = patient_nbr[df_gan.index]
labels = labels[df_gan.index]




numerical_attribs  = list(df_gan.select_dtypes(include=['int64', 'float64']))
scailing = ColumnTransformer(transformers=[('standart_scaler', StandardScaler(), numerical_attribs)],
                             verbose_feature_names_out=False,
                             remainder='passthrough')  


df_smote = scailing.fit_transform(df_gan)

if preprocess_for_gan_smote:

    # save data for GAN
    df_gan.to_pickle('df_gan.pkl')
    labels.to_pickle('labels.pkl')
    patient_nbr.to_pickle('patient_nbr.pkl')
    # save data for SMOTE
    df_smote.to_pickle('df_smote.pkl')

# save features and lables dataframes for hyper-parameters tuning of the diffrent models
data_cv_GAN = pd.read_csv('data_cv_balanced_nfolds_10.csv')
data_cv_SMOTE = pd.read_csv('data_cv_balanced_nfolds_10_SMOTE.csv')

data_cv_GAN[['admission_type_id','discharge_disposition_id','admission_source_id']] = data_cv_GAN[['admission_type_id','discharge_disposition_id','admission_source_id']].astype('object')
data_cv_SMOTE[['admission_type_id','discharge_disposition_id','admission_source_id']] = data_cv_SMOTE[['admission_type_id','discharge_disposition_id','admission_source_id']].astype('object')

labels_cv_GAN = data_cv_GAN['readmitted']
data_cv_GAN = data_cv_GAN.drop(columns = 'readmitted')
labels_cv_SMOTE = data_cv_SMOTE['readmitted']
data_cv_SMOTE = data_cv_SMOTE.drop(columns = 'readmitted')


numerical_attribs  = list(data_cv_GAN.select_dtypes(include=['int64', 'float64']))
categorical_attribs =list(data_cv_GAN.select_dtypes(include='object'))



one_hot_encoder = ColumnTransformer(transformers=[("one_hot_encoder", OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore'), categorical_attribs)],
                                    verbose_feature_names_out=False,
                                    remainder='passthrough')


scailing = ColumnTransformer(transformers=[('standart_scaler', StandardScaler(), numerical_attribs)],
                             verbose_feature_names_out=False,
                             remainder='passthrough')  # min-max - Tryed - lower accuracy in random forest

# do the remain step for the bossting models (e.g xgb, lgbm, catBoost)
data_cv_boosting_models_gan = scailing.fit_transform(data_cv_GAN)

# do the remain step for random forests


data_cv_rf = one_hot_encoder.fit_transform(data_cv_boosting_models_gan)
data_cv_rf_smote = one_hot_encoder.fit_transform(data_cv_SMOTE)




# prepare the data for training on the whole dataset and evaluate the relults via the test set
preprocessing_pipe_full_rf = Pipeline(steps=[
    ('manual_transform', ManualTranform()),
    ('feature_adder', FeatureAdder()),
    ('drop_catFet_by_variance', DropFeaturesByVariance(th = 99)),
    ('Drop_children_patiant', DropYoung()),
    ('one_hot_encoder', one_hot_encoder),
    ('standart_scaler', scailing)    
])

if preprocess_for_random_forest :
    df_train = preprocessing_pipe_full_rf.fit_transform(df_features_train)
    df_test = preprocessing_pipe_full_rf.transform(df_features_test)
    labels_test = labels_test[df_test.index]
    df_train.to_pickle('train_set_after_preprocessing_rf.pkl')
    df_test.to_pickle('test_set_after_preprocessing_rf.pkl')
    labels.to_pickle('labels_train_set.pkl')
    labels_test.to_pickle('labels_test_set.pkl')
    data_cv_rf.to_pickle('data_cv_gan_rf.pkl')
    data_cv_rf_smote.to_pickle('data_cv_smote_rf.pkl')
    labels_cv_SMOTE.to_pickle('labels_cv_smote.pkl')
    labels_cv_GAN.to_pickle('labels_cv_gan.pkl')

preprocessing_pipe_full_boosting = Pipeline(verbose = True , steps=[
    ('manual_transform', ManualTranform()),
    ('Drop_children_patiant', DropYoung()),
    ('feature_adder', FeatureAdder()),
    ('standart_scaler', scailing),
    ('drop_catFet_by_variance', DropFeaturesByVariance(th = 99)),
])

if preprocess_for_boosting_models :
    df_train =  preprocessing_pipe_full_boosting.fit_transform(df_features_train)
    df_test = preprocessing_pipe_full_boosting.transform(df_features_test)
    labels_test = labels_test[df_test.index]
    df_train.to_pickle('train_set_after_preprocessing_boosting_models.pkl')
    df_test.to_pickle('test_set_after_preprocessing_boosting_models.pkl')
    labels.to_pickle('labels_train_set.pkl')
    labels_test.to_pickle('labels_test_set.pkl')
    data_cv_boosting_models_gan.to_pickle('data_cv_gan_boosting_models.pkl')
    data_cv_SMOTE.to_pickle('data_cv_smote_boosting_models.pkl')
    labels_cv_SMOTE.to_pickle('labels_cv_smote.pkl')
    labels_cv_GAN.to_pickle('labels_cv_gan.pkl')


