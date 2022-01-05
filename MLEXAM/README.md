# Tanabo2 Maradh l9alb , Msadek Ghaith 



## Project Overview

Fel Projet hédha bch Nepredictiw Par rapport a certaines Features , une personne aandha  risk anha Tkoun Mridha bel 9alb wale
Bch Nadhfou Dataset mté3na  na7iw outliars w missing Values Nchoufou ahsen Model w n7awlou nhassnou fel metrics mté3ou .
El projet hédha bch yet9asem al les etapes hédhom :

1. [Data description]
2. [Importing Libraries & setting up environment]
3. [Loading dataset]
4. [Data Cleaning & Preprocessing]
5. [Exploratory Data Analysis]
6. [OUtlier Detection & Removal]
7. [Training & Test Split]
8. [Cross Validation]
9. [Model Building]
10. [Model evaluation & comparison]<br>
11. [Conclusion]

## About Data 

Dataset Fih  11 features W 1 target variable: fIH 6 categoricals(nominal) variables w 5 numeric variables.
Les details mtaa3 el Features;

1. Age: Patients Age in years (Numeric)
2. Sex: Gender of patient (Male - 1, Female - 0) (Nominal)
3. Chest Pain Type: Naaw3 el ChestPain : 1 typical(wji3a causé par le coeur), 2 atypical angina(wji3a par forcément mel coeur tnjm tkoun mel respiration etc), 3 non-anginal pain(maandha hata da5l bel 9alb), 4 asymptomatic(silent killer) (Nominal)
4. resting bp s: Niveau mtaa3 l'blood pressure wel partient merté7(resting) in mm/HG (Numerical)
5. cholestrol: Serum cholestrol in mg/dl (Numeric)
6. fasting blood sugar: Taux de sucre dans le sang à jeun > 120 mg/dl represents as 1 in case of true and 0 as false (Nominal)
7. resting ecg: Result of electrocardiogram while at rest are represented in 3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy (Nominal)
8. max heart rate: Max mtaa3 da9at l9alb (Numeric)
9. exercise angina: Angina pain mba3 ma taa3ml SPORT : 0 = NO ,1 = Yes (Nominal)
10. oldpeak: Exercise induced ST-depression in comparison with the state of rest (Numeric)
11. ST slope: ST segment measured in terms of slope during peak exercise 0: Normal 1: Upsloping 2: Flat 3: Downsloping (Nominal)

Target variable
12. target: Label eli bch nprédictiweh 1 Maantha patient aandou risk mtaa3 9alb and 0 Maantha Patient normal.

## Installations :
Lazmk  Python 3.x wel Librairie Python hédhom lazm ikounou installées:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [xgboost](https://xgboost.readthedocs.io/en/latest/build.html)

Nansa7kom Tinstaliw  Anaconda, Eli féha All the  libraries and software for this project weli zeda  jupyter notebook to run and execute

## Code :
Actual code bch tebda le project hédha tal9ah fil File ```heart-disease-exam.ipynb```
W zeda Pour la Data Mining Fama file Hédha ```heart-disease-exam.py```

## Run :
Fel terminal , CD  to project directory / (Eli fih el  README) w Run el commande hédhi :

```ipython notebook heart-disease-exam.ipynb```
Ou bien :

```jupyter notebook heart-disease-exam.ipynb```

comande hédhi bch t7éélk Jupyter Notebook wel project hédha fel 

 browser.

## Model Evaluation :

1. [Cross Validation Score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
2. [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
3. [Sensitivity and Specitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
4. [Classification Error](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
5. [Log Loss](https://www.kaggle.com/dansbecker/what-is-log-loss#:~:text=Log%20Loss%20is%20a%20slight,by%20understanding%20the%20likelihood%20function.)
6. [Mathew Correlation coefficient](https://www.kaggle.com/dansbecker/what-is-log-loss#:~:text=Log%20Loss%20is%20a%20slight,by%20understanding%20the%20likelihood%20function.)

