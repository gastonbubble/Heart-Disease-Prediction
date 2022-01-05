#Importing Libraries
import warnings
warnings.filterwarnings('ignore')

# data pre-processing
import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split




# machine learning algorithms

from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import MinMaxScaler

from scipy import stats
#loading Data
dt = pd.read_csv('heart_final.csv')

# renaming features to proper name
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope','target']

       # converting features to categorical features 

dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'



dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'



dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 2] = 'flat'
dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'

dt["sex"] = dt.sex.apply(lambda  x:'male' if x==1 else 'female')

#Bch ndropiw el Ligne li féha st_slope =0
dt.drop(dt[dt.st_slope ==0].index, inplace=True)
## 6. Outlier Detection & Removal <a id='data-out'></a>

# filtering numeric features as age , resting bp, cholestrol and max heart rate achieved has outliers as per EDA

dt_numeric = dt[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved']]
# calculating zscore of numeric columns in the dataset
z = np.abs(stats.zscore(dt_numeric))
# Defining threshold for filtering outliers 
threshold = 3
dt = dt[(z < 3).all(axis=1)]
## encoding categorical variables:
# RQ : houni Hatina drop_first=True bch na9sou mel Nbr de columns générés, par Exemple sexe fih Female w male
# kén maamlnéch drop_first = true ,bch tjina sex_male w sex_female donc bch na9sou ala rwé7na les features 
dt = pd.get_dummies(dt, drop_first=True)
# Lazm n9assmou datasey lel X and Y  Features and Label :
X = dt.drop(['target'],axis=1)
y = dt['target']

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)

#Feature Normalization
scaler = MinMaxScaler()
X_train[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.fit_transform(X_train[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']])
X_test[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.transform(X_test[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']])


#Building Model
rf_ent = RandomForestClassifier(criterion='entropy',n_estimators=100)
rf_ent.fit(X_train, y_train)
y_pred_rfe = rf_ent.predict(X_test)
