import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn import set_config

# Configure sklearn to transform output to pandas DataFrame
set_config(transform_output="pandas")

# Custom transformer to drop features based on variance
class DropFeaturesByVariance(BaseEstimator, TransformerMixin):
    def __init__(self, th):
        self.th = th

    def fit(self, X, y=None):
        return self  

    def transform(self, X, y=None):
        df = X.copy()

        # Function to find category with max occurrence and its percentage for each column
        def find_max_occurrence_and_percentage(column):
            value_counts = column.value_counts()
            max_category = value_counts.idxmax()
            max_count = value_counts.max()
            total_count = column.count()
            percentage = (max_count / total_count) * 100
            return max_category, percentage

        # Iterate over columns and find category with max occurrence and percentage
        result = {}
        for column in df.columns:
            max_category, percentage = find_max_occurrence_and_percentage(df[column])
            result[column] = percentage

        # Filter columns by threshold
        filtered_keys = [key for key, value in result.items() if value < self.th]
        return df[filtered_keys]


# Custom transformer for manual transformations
class ManualTranform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X,y = None):
        df = X.copy()

        # Function to convert string to float
        def floatConvert(str):
            try:
                float(str)
                return float(str)
            except ValueError:
                return 0
        df[['admission_type_id','discharge_disposition_id','admission_source_id']] = df[['admission_type_id','discharge_disposition_id','admission_source_id']].astype('object')


        # Dictionary for replacing age categories
        replaceDict_age = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}

        low_frequency = ['Surgery-PlasticwithinHeadandNeck', 'Psychiatry-Addictive', 'Proctology', 'Dermatology', 'SportsMedicine', 'Speech', 'Perinatology',
                         'Neurophysiology', 'Resident', 'Pediatrics-Hematology-Oncology', 'Pediatrics-EmergencyMedicine', 'Dentistry', 'DCPTEAM', 'Psychiatry-Child/Adolescent',
                         'Pediatrics-Pulmonology', 'Surgery-Pediatric', 'AllergyandImmunology', 'Pediatrics-Neurology', 'Anesthesiology', 'Pathology', 'Cardiology-Pediatric',
                         'Endocrinology-Metabolism', 'PhysicianNotFound', 'Surgery-Colon&Rectal', 'OutreachServices', 'Surgery-Maxillofacial', 'Rheumatology', 'Anesthesiology-Pediatric',
                         'Obstetrics', 'Obsterics&Gynecology-GynecologicOnco']

        pediatrics = ['Pediatrics', 'Pediatrics-CriticalCare', 'Pediatrics-EmergencyMedicine', 'Pediatrics-Endocrinology', 'Pediatrics-Hematology-Oncology',
                      'Pediatrics-Neurology', 'Pediatrics-Pulmonology', 'Anesthesiology-Pediatric', 'Cardiology-Pediatric', 'Surgery-Pediatric']

        psychic = ['Psychiatry-Addictive', 'Psychology', 'Psychiatry', 'Psychiatry-Child/Adolescent', 'PhysicalMedicineandRehabilitation', 'Osteopath']

        neurology = ['Neurology', 'Surgery-Neuro', 'Pediatrics-Neurology', 'Neurophysiology']

        surgery = ['Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&Rectal', 'Surgery-General', 'Surgery-Maxillofacial',
                   'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck', 'Surgery-Thoracic', 'Surgery-Vascular', 'SurgicalSpecialty', 'Podiatry']

        # Map medical specialty to broader categories
        df['medical_specialty'] = df['medical_specialty'].apply(lambda x: 'low_frequency' if x in low_frequency
                                                                    else ('surgery' if x in surgery
                                                                          else ('neurology' if x in neurology
                                                                                else ('psychic' if x in psychic
                                                                                      else ('pediatrics' if x in pediatrics
                                                                                            else x)))))

            # Convert diag_1, diag_2, diag_3 to numerical values and categorize them
        df['diag_1'] = df['diag_1'].apply(lambda x: 0 if (str(x).lower() == 'nan') or (x == None) else floatConvert(x))
        df['diag_2'] = df['diag_2'].apply(lambda x: 0 if (str(x).lower() == 'nan') or (x == None) else floatConvert(x))
        df['diag_3'] = df['diag_3'].apply(lambda x: 0 if (str(x).lower() == 'nan') or (x == None) else floatConvert(x))

        for diag in ['diag_1', 'diag_2', 'diag_3']:
                df[diag] = df[diag].apply(lambda x: 'circulatory' if int(x) in [1, 2]
                else ('respiratory' if (int(x) in range(460, 520)) or (int(x) == 786)
                else ('digestive' if (int(x) in range(520, 580)) or (int(x) == 787)
                else ('diabities_without_complications' if (str(x).startswith('250.0')) or (abs(float(x) - 50) < 10e-10)
                else ('diabities_with_complications' if (str(x).startswith('250.')) and (not str(x).startswith('250.0'))
                else ('injury' if int(x) in range(800, 1000)
                else ('musculoskeletal' if int(x) in range(710, 740)
                else ('genitourinary' if (int(x) in range(580, 630)) or (int(x) == 788)
                else ('neoplasms' if int(x) in range(140, 240)
                else ('pregnecy' if int(x) in range(630, 680)
                else 'other'))))))))))

            # Replace age categories with numerical values
        df['age'] = df['age'].apply(lambda x: replaceDict_age[x] if x in replaceDict_age.keys()
                                        else np.nan)
        df['change'] = df['change'].apply(lambda x : 1 if x == 'Ch'
                                                 else -1)


        return df


# Custom transformer to add features
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  

    def transform(self, X,y = None):
        df = X.copy()
        # Add a new features 
        df['severity_of_disease'] = (2*df['time_in_hospital'] + df['num_procedures'] + df['num_medications'] + df['num_lab_procedures'])
        df['sick_index'] = ( (df['number_emergency'] + df['number_inpatient'] + df['number_outpatient']) )


        return df

class DropYoung(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X,y=None):
            return self

        def transform(self,X,y=None):
            df = X.copy()
            df = df[(df['age'] > 1)]
            return df
        # Add a new features 

# Custom transformer for debugging
class Debugger(BaseEstimator, TransformerMixin):
    def transform(self, data):
        # Print shape of pre-processed data
        data.to_csv('debug_df.csv')
        print(data)
        print("Shape of Pre-processed Data:", data.shape)
        return pd.DataFrame(data.reset_index(drop=True))

    def fit(self, data, y=None, **fit_params):
        return self  # No need to fit anything

class DropOUtliers(BaseEstimator, TransformerMixin):
        def __init__(self,cat = 0.01, num=0.02):
            self.cat = cat
            self.num = num

        def fit(self, X, y=None):
            return self  # nothing else to do

        def transform(self, X):
            df = X.copy()
            numerical = list(df.select_dtypes(include=['int64', 'float64']))
            categorical = list(df.select_dtypes(include=['object']))
 

            for col in categorical:

                # Calculate the percentage of unique values in the column
                unique_values_percentage = (df[col].value_counts(dropna=True))
                unique_values_percentage = unique_values_percentage/ len(df[col])

                # Filter out unique values with a percentage lower than 2%
                filtered_unique_values = unique_values_percentage[unique_values_percentage < self.cat].index.tolist()

                # Update the DataFrame by keeping only rows where the column value is in the filtered unique values
                df.loc[df[col].isin(filtered_unique_values) == True,col] =  np.nan

                # Define a function to remove outliers using the IQR method for a specific column
            for col in numerical:
            # Calculate the 10th and 90th percentiles
                lower_bound = df[col].quantile(self.num)
                upper_bound = df[col].quantile(1-self.num)
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                df.loc[outliers.index,col] = np.nan
            return df
        
