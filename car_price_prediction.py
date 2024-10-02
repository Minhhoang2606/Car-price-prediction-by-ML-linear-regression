import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# Data preprocessing
## Load the data
car_df = pd.read_csv(R'data\car data.csv')

## Basic information
car_df.head()
car_df.shape
car_df.info()

## Check for missing values
car_df.isnull().sum()
'''
There is no missing values in the dataset.
'''

## Categorical data analysis
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=car_df, ax=ax[i])
    ax[i].set_title('Countplot of {}'.format(col))
    ax[i].set_xlabel(col)
    ax[i].set_ylabel('Count')
plt.show()



