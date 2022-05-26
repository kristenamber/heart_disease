# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from initial_info import DataInfo
from feature_distribution import FeatureDist
from feature_generation import FeatureGen
from model import Model




#read in the data
df = pd.read_csv('heart_2020_cleaned.csv')




############## use initial_info to explore aspects of the data ######################
my_info = DataInfo(df)

# get info on feature types and number of non null values
print(my_info.get_feature_types_nuls())


# we see that there is a mixture of features with object types and float types

# explore unique categorical elements of a specific feature

#print(my_info.get_unique("HeartDisease"))
#print(my_info.get_unique("GenHealth"))


# explore min/ max range of features with float type.

#min_sleep_time, max_sleep_time = my_info.get_min_max("SleepTime")
#print("The minimum amount of sleep present is {0}, the maximum amount is {1}".format(min_sleep_time, max_sleep_time))


#Explore feature distributions
my_dists = FeatureDist(df)
#my_dists.get_boxplot('SleepTime')
#my_dists.get_hist("SkinCancer")


# Explore Heart disease
#print(my_info.get_unique("HeartDisease"))
# This is binary we can change change to 0's and 1's
heart_disease_params = {'No': 0, 'Yes':1}


#Eplore Smoking
#print("smoking")
#print(my_info.get_unique('Smoking'))
#This is binary, we can change to 0's and 1's
smoking_params = {'Yes':1, 'No':0}


#Explore AlcoholDrinking
#print('Alcohol')
#print(my_info.get_unique('AlcoholDrinking'))
#This is binary, we can change to 0's and 1's
alcohol_params = {'Yes':1, 'No':0}

#Explore Stroke
#print('stroke')
#print(my_info.get_unique('Stroke'))
#This is binary, we can change to 0's and 1's
stroke_params = {'Yes':1, 'No':0}

# Explore DiffWalking
#print('difficulty walking')
#print(my_info.get_unique('DiffWalking'))
# This binary, we can change to 0's and 1's
diffwalking_params = {'Yes':1, 'No':0}

# Explore Sex
#print('Sex')
#print(my_info.get_unique('Sex'))
# This is a nominal categorical variable Since we are doing
# A decision tree classifier we will stick with buckets instead of
# one-hot encoding. Need to do one-hot encoding for something like SVM
# or logistic regression
sex_params = {'Female':1, 'Male': 0}

#Eplore Age Category
#print('Age Category')
#print(my_info.get_unique('AgeCategory'))
#These are ordinal categorical. We can use buckets
# for both the tree classifier and a classifier such
# as svm or logistic regression
age_cat_params = {'18-24':0, '25-29':1, '30-34': 2, '35-39': 3, '40-44':4,
                  '45-49':5, '50-54':6, '55-59':7, '60-64':8, '65-69': 9, '70-74': 10,
                  '75-79':11, '80 or older': 12}

####Explore Race
#print("race")
#print(my_info.get_unique('Race'))
#This is a nominal categorical feature. For using the Tree Classifier we can use
# buckets but we would need to use one-hot encoding for something like SVM
# or logistic regression
race_params = {"White":0, 'Black': 1, 'Asian': 2, 'American Indian/Alaskan Native':3, 'Other': 4, 'Hispanic':5}

#### Explore Diabetic
#print('Diabetic')
#print(my_info.get_unique('Diabetic'))
# This nominal categorical
diabetic_params = {'Yes':0, 'No':1, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3}


###### Explore Physical Activity
#print('Physical Activity')
#print(my_info.get_unique('PhysicalActivity'))
phys_act_params = {'Yes':1, 'No': 0}

#### Explore GenHealth
#print('GenHealth')
#print(my_info.get_unique('GenHealth'))
gen_health_params = {'Poor':0, 'Fair':1, 'Good':2, 'Very good':3, 'Excellent': 4}

#### Explore Asthma
#print("Asthma")
#print(my_info.get_unique('Asthma'))
asthma_params = {'Yes':1, 'No':0}

######Explore Kidney Disease
#print('Kidney Disease')
#print(my_info.get_unique('KidneyDisease'))
kidney_params = {'Yes':1, 'No':0}

##### Explore Skin Cancer
#print('Skin Cancer')
#print(my_info.get_unique('SkinCancer'))
skin_cancer_params = {'Yes':1, 'No':0}






###########################################  Get new features  ####################################################3


feature_list = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
                'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',
                'Asthma', 'KidneyDisease', 'SkinCancer']

feature_params = []
feature_params.append(heart_disease_params)
feature_params.append(smoking_params)
feature_params.append(alcohol_params)
feature_params.append(stroke_params)
feature_params.append(diffwalking_params)
feature_params.append(sex_params)
feature_params.append(age_cat_params)
feature_params.append(race_params)
feature_params.append(diabetic_params)
feature_params.append(phys_act_params)
feature_params.append(gen_health_params)
feature_params.append(asthma_params)
feature_params.append(kidney_params)
feature_params.append(skin_cancer_params)

get_features = FeatureGen(df)
new_df = get_features.ordinal_cat(feature_list, feature_params)

my_new_info = DataInfo(new_df)
print(my_new_info.get_feature_types_nuls())







######################################train a Random Forest Classifier#########################################
labels = new_df['HeartDisease']
data = new_df.drop(columns = 'HeartDisease')

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=3)

my_model = Model(RandomForestClassifier(n_estimators=10, max_depth=5))
my_model.fit(X_train, y_train)

confusion_array = my_model.get_confusion(X_train,y_train)
print("the confusion matrix is")
print(confusion_array)


predictions = my_model.predict(X_train)




true_negative = confusion_array[0][0]
false_negative = confusion_array[0][1]

true_positive = confusion_array[1][1]
false_positive = confusion_array[1][0]

accuracy = (true_negative + true_positive)/ (true_negative + true_positive + false_positive + false_negative)
recall = (true_positive)/(true_positive + false_negative)
precision = (true_positive)/(false_positive+ true_positive)

print("The accuracy of the model is {0}".format(accuracy))
print("The recall of the model is {0}".format(recall))
print("The precision of the model is {0}".format(precision))



#Interpretation
# When dealing with heart disease we want to make sure our recall is good. We want to make sure we're catching
# people that actually have heart disease.

#It's okay to have a lower precision. Its okay to flag someone has possibly having heart disease that might not.



# create Fast API around model






# Interpretability
