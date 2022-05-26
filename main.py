# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from initial_info import DataInfo
from feature_distribution import FeatureDist




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





#expirement with different models




# choose best model






# create Fast API around model






# Interpretability
