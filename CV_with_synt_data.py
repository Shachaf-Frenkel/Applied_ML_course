import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import  StratifiedGroupKFold
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
from imblearn.over_sampling import SMOTENC
import numpy as np
from sklearn import set_config
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def split_data_for_cv_GAN(X_train,y_train,patient_nbr,nfolds = 10) :
            data_train = pd.concat((X_train, y_train),axis = 1)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data_train)
            metadata.update_column(
                column_name='change',
                sdtype='categorical')
            metadata.update_column(
                column_name='age',
                sdtype='numerical')
            metadata.update_column(
                column_name='num_procedures',
                sdtype='numerical')
            metadata_dict = metadata.to_dict()
            # make sure each colomn metadata data type is correct if not then you need to  change it using the update_column method https://docs.sdv.dev/sdv/single-table-data/data-preparation/single-table-metadata-api
            # print("metadata_dict", metadata_dict)
            metadata.validate()
            metadata.validate_data(data=data_train)
            synthesizer = CopulaGANSynthesizer(metadata,
                                        # enforce_min_max_values=True, enforce_rounding=False,
                                        # numerical_distributions={'amenities_fee': 'beta', 'checkin_date': 'uniform'},
                                        epochs=500,
                                        verbose=True,
                                        #cuda=cuda
                            )
            # Remember in case you using more than one sample for some instances (e.g. multiple records per patients) then you need to perform stratified split by groups e.g., using StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
            skf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=42)
            folds_idx = []
            folds_idx_synt = []
            start = 0
            X_train[X_train.columns[X_train.dtypes == 'category']] = X_train[X_train.columns[X_train.dtypes == 'category']].astype('object')
            # for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
            for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train,groups=patient_nbr)):

                X_train_cv = X_train.iloc[train_index]
                y_train_cv = y_train.iloc[train_index]
                X_test_cv = X_train.iloc[test_index]
                y_test_cv = y_train.iloc[test_index]
                data_train_cv = pd.concat((X_train_cv, y_train_cv), axis =1)
                data_test_cv = pd.concat((X_test_cv, y_test_cv), axis =1)
                count_classes = pd.DataFrame(y_train).value_counts()
                minor_class = count_classes.idxmin()[0]
                minority_data_train_cv = data_train_cv.loc[data_train_cv['readmitted'] == minor_class]


                unique, counts = np.unique(y_train_cv, return_counts=True)

                print('difference between minority and majority', max(counts)-min(counts))
                synthesizer.fit(minority_data_train_cv)
                # Note you might not want to make the dataset balanced but only reduce the ratio between the majority and minority classes or even oversample both minority and majority to reduce overfitting
                synthetic_data = synthesizer.sample(max(counts)-min(counts))
                # print(synthetic_data.shape)

                if fold == 0:
                    print('fold', fold)
                    data_cv_balanced = data_train_cv
                else:
                     print('fold', fold)
                     data_cv_balanced = pd.concat((data_cv_balanced, data_train_cv), ignore_index = True)
                     print('fold', fold)
                data_cv_balanced = pd.concat((data_cv_balanced, synthetic_data), ignore_index = True)
                folds_idx_synt.append(np.array(range(len(data_cv_balanced)-len(synthetic_data),len(data_cv_balanced))))

                folds_idx.append((np.arange(start, start + data_train_cv.shape[0] + synthetic_data.shape[0]), np.arange(start + data_train_cv.shape[0] + synthetic_data.shape[0], start + data_train_cv.shape[0] ++ synthetic_data.shape[0] + data_test_cv.shape[0])))

                data_cv_balanced = pd.concat((data_cv_balanced, data_test_cv), ignore_index = True)
                start =+ data_cv_balanced.shape[0]

            data_cv_balanced['severity_of_disease'] = (2*data_cv_balanced['time_in_hospital'] + data_cv_balanced['num_procedures'] + data_cv_balanced['num_medications'] + data_cv_balanced['num_lab_procedures'])
            data_cv_balanced['sick_index'] = ( (data_cv_balanced['number_emergency'] + data_cv_balanced['number_inpatient'] + data_cv_balanced['number_outpatient']) )

            data_cv_balanced.to_csv(f"data_cv_balanced_nfolds_{nfolds}.csv", index=False)



            with open('folds_idx.pickle', 'wb') as output_file:
                pickle.dump(folds_idx, output_file)

            with open('folds_idx_synt.pickle', 'wb') as output_file:
                pickle.dump(folds_idx_synt, output_file)


# Make CV file with Gan synthetic data:
X_train = pd.read_pickle('X_train_gan.pkl')
y_train = pd.read_pickle('y_train_gan.pkl')
patient_nbr = pd.read_pickle('patient_nbr.pkl')
split_data_for_cv_GAN(X_train,pd.DataFrame(y_train),patient_nbr)




def split_data_for_cv_SMOTE(X_train,y_train,patient_nbr,nfolds = 10) :
            set_config(transform_output="default")
            # numerical_attribs  = list(X_train.select_dtypes(include=['int64', 'float64']))
            # scailing = ColumnTransformer(transformers=[('standart_scaler', StandardScaler(), numerical_attribs)],
            #                  verbose_feature_names_out=False,
            #                  remainder='passthrough')
            # X_train = scailing.fit_transform(X_train)
            # print(X_train)
            # # Convert X_train to a DataFrame if it's not already
            # if not isinstance(X_train, pd.DataFrame):
            #     X_train = pd.DataFrame(X_train)

            data_train = pd.concat((X_train, y_train),axis = 1)
            categorical_features = ["race",	"gender",	"medical_specialty",	"diag_1",	"diag_2",	"diag_3",	"max_glu_serum",	"A1Cresult",	"metformin",	"repaglinide",	"nateglinide",	"chlorpropamide",	"glimepiride","acetohexamide",	"glipizide",	"glyburide",	"tolbutamide",	"pioglitazone",	"rosiglitazone",	"acarbose",	"miglitol",	"troglitazone",	"tolazamide",	"insulin",	"glyburide-metformin",	"glipizide-metformin",	"metformin-rosiglitazone",	"change",	"diabetesMed",	"discharge_disposition_id","admission_type_id",	"admission_source_id"]
            categorical_features  = list(X_train.select_dtypes(include=['object', 'category']))
            oversample = SMOTENC(sampling_strategy={'<30':50000},random_state=42,categorical_features=categorical_features)
            skf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=42)
            folds_idx = []
            folds_idx_synt = []
            start = 0
            X_train[X_train.columns[X_train.dtypes == 'category']] = X_train[X_train.columns[X_train.dtypes == 'category']].astype('object')
            # for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
            for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train,groups=patient_nbr)):

                X_train_cv = X_train.iloc[train_index]
                y_train_cv = y_train.iloc[train_index]
                X_test_cv = X_train.iloc[test_index]
                y_test_cv = y_train.iloc[test_index]
                data_train_cv = pd.concat((X_train_cv, y_train_cv), axis =1)
                data_test_cv = pd.concat((X_test_cv, y_test_cv), axis =1)
                count_classes = pd.DataFrame(y_train).value_counts()
                minor_class = count_classes.idxmin()[0]
                minority_data_train_cv = data_train_cv.loc[data_train_cv['readmitted'] == minor_class]


                unique, counts = np.unique(y_train_cv, return_counts=True)

                print('difference between minority and majority', max(counts)-min(counts))

                X, Y = oversample.fit_resample(X_train_cv, y_train_cv)
                # Note you might not want to make the dataset balanced but only reduce the ratio between the majority and minority classes or even oversample both minority and majority to reduce overfitting
                synthetic_data_X = X.iloc[-50000:]
                synthetic_data_y = Y.iloc[-50000:]
                synthetic_data = pd.concat((synthetic_data_X,synthetic_data_y), axis =1)
                print(synthetic_data.shape)

                if fold == 0:
                    print('fold', fold)
                    data_cv_balanced = data_train_cv
                else:
                     print('fold', fold)
                     data_cv_balanced = pd.concat((data_cv_balanced, data_train_cv), ignore_index = True)
                     print('fold', fold)
                data_cv_balanced = pd.concat((data_cv_balanced, synthetic_data), ignore_index = True)
                folds_idx_synt.append(np.array(range(len(data_cv_balanced)-len(synthetic_data),len(data_cv_balanced))))

                folds_idx.append((np.arange(start, start + data_train_cv.shape[0] + synthetic_data.shape[0]), np.arange(start + data_train_cv.shape[0] + synthetic_data.shape[0], start + data_train_cv.shape[0] ++ synthetic_data.shape[0] + data_test_cv.shape[0])))

                data_cv_balanced = pd.concat((data_cv_balanced, data_test_cv), ignore_index = True)
                start =+ data_cv_balanced.shape[0]

            data_cv_balanced['severity_of_disease'] = (2*data_cv_balanced['time_in_hospital'] + data_cv_balanced['num_procedures'] + data_cv_balanced['num_medications'] + data_cv_balanced['num_lab_procedures'])
            data_cv_balanced['sick_index'] = ( (data_cv_balanced['number_emergency'] + data_cv_balanced['number_inpatient'] + data_cv_balanced['number_outpatient']) )

            data_cv_balanced.to_csv(f"data_cv_balanced_nfolds_{nfolds}_SMOTE.csv", index=False)



            with open('folds_idx_SMOTE.pickle', 'wb') as output_file:
                pickle.dump(folds_idx, output_file)

            with open('folds_idx_synt_SMOTE.pickle', 'wb') as output_file:
                pickle.dump(folds_idx_synt, output_file)


X_train = pd.read_pickle('X_train_gan.pkl')
y_train = pd.read_pickle('y_train_gan.pkl')
patient_nbr = pd.read_pickle('patient_nbr.pkl')
from sklearn import set_config

# Configure sklearn to transform output to pandas DataFrame
set_config(transform_output="pandas")
numerical_attribs  = list(X_train.select_dtypes(include=['int64', 'float64']))
scailing = ColumnTransformer(transformers=[('standart_scaler', StandardScaler(), numerical_attribs)],
                             verbose_feature_names_out=False,
                             remainder='passthrough')
X_train = scailing.fit_transform(X_train)
print(X_train)
# Convert X_train to a DataFrame if it's not already
if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
split_data_for_cv_SMOTE(X_train,pd.DataFrame(y_train),patient_nbr)
