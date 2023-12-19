#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:03:20 2023

@author: mayankgrover
"""
import os
os.getcwd()
os.chdir('/Users/mayankgrover/Documents/AIT-580/project')
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Reading data from the cleaned dataset
data = pd.read_csv('data_CT.csv', sep=',')

# creating a copy of the original data
df = data.sample(frac=1)


df.head(10)
df.describe()
df.info()

len(df['Cause.of.Death'].unique())
len(df['Description.of.Injury'].unique())
df['Description.of.Injury'].unique()
                                            
df['index_col'] = df.index

melted_df = pd.melt(df,id_vars=['index_col','Description.of.Injury'],value_vars=['Cocaine', 'Fentanyl','Fentanyl.Analogue', 'Oxycodone', 'Oxymorphone', 'Ethanol','Hydrocodone', 'Benzodiazepine', 'Methadone', 'Meth.Amphetamine','Amphet', 'Tramad', 'Hydromorphone', 'Morphine..Not.Heroin.','Xylazine', 'Gabapentin', 'Opiate.NOS', 'Heroin.Morph.Codeine','Other.Opioid', 'Any.Opioid', 'Other'],var_name='Substance',value_name='Indicator')
melted_df['Indicator'].isna()
df2 = melted_df.dropna()

df2 = df2.copy()
df2['Description.of.Injury'] = df2['Description.of.Injury'].str.lower()
df2['Substance'] = df2['Substance'].str.lower()


df3 = df2.groupby(['Description.of.Injury','Substance']).size().reset_index().rename(columns={0:'count'})

df3.head(20)
df3['Description.of.Injury'].value_counts()
df3['Substance'].value_counts()
df3.columns

all_substances = df3['Substance'].unique()
all_substances
 

for s in all_substances:
    df_sub = df3[(df3['Substance'] == s) & (df3['count'] > 5)]
    sns.barplot(df_sub,x='Description.of.Injury',y='count')
    plt.title('Major reasons of deaths caused by ' +str(s))
    plt.xticks(rotation=90)
    plt.show()
    


df4 = df[['Death.City','Age']]
df4.head(20)

death_avg_age_df = df4.groupby('Death.City')['Age'].mean().reset_index()
death_avg_age_df.info()

death_avg_age = death_avg_age_df[(death_avg_age_df['Age']>20) & (death_avg_age_df['Age']<40)]

fig,ax= plt.subplots(figsize=(20,10))
sns.barplot(x='Death.City', y='Age', data=death_avg_age,ax=ax)
plt.title('Average age of people dying in various cities')
plt.ylabel('Average Age')
plt.xticks(rotation=90)
plt.show()


# splitting the records into 5
death_1 = death_avg_age_df[:45]
death_2 = death_avg_age_df[45:90]
death_3 = death_avg_age_df[90:135]
death_4 = death_avg_age_df[135:180]
death_5 = death_avg_age_df[180:225]

fig,ax= plt.subplots(figsize=(20,10))
sns.barplot(x='Death.City', y='Age', data=death_1,ax=ax)
plt.title('Average age of people dying in various cities')
plt.ylabel('Average Age')
plt.xticks(rotation=90)
plt.show()

fig,ax= plt.subplots(figsize=(20,10))
sns.barplot(x='Death.City', y='Age', data=death_2,ax=ax)
plt.title('Average age of people dying in various cities')
plt.xticks(rotation=90)
plt.ylabel('Average Age')
plt.show()

fig,ax= plt.subplots(figsize=(20,10))
sns.barplot(x='Death.City', y='Age', data=death_3,ax=ax)
plt.title('Average age of people dying in various cities')
plt.xticks(rotation=90)
plt.ylabel('Average Age')
plt.show()

fig,ax= plt.subplots(figsize=(20,10))
sns.barplot(x='Death.City', y='Age', data=death_4,ax=ax)
plt.title('Average age of people dying in various cities')
plt.xticks(rotation=90)
plt.ylabel('Average Age')
plt.show()

fig,ax= plt.subplots(figsize=(20,10))
sns.barplot(x='Death.City', y='Age', data=death_5,ax=ax)
plt.title('Average age of people dying in various cities')
plt.xticks(rotation=90)
plt.ylabel('Average Age')
plt.show()


df['Year'].value_counts()

year_race_df = df.groupby('Year')['Race'].value_counts().unstack().reset_index()
year_race_df = year_race_df.fillna(value=0)

races = year_race_df.columns[1:]

plt.figure(figsize=(10, 6))

for race in races:
    plt.plot(year_race_df['Year'], year_race_df[race], label=race)

plt.xlabel('Year')
plt.ylabel('Death Count')
plt.title('Year wise Death counts for Each Race')
plt.legend(title='Race', loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
    

#####################
# Feature Engineering
df['Gender'] = df['Sex'].map({'Male': 0, 'Female': 1})
races_list = list(races)
df['race_category'] = pd.Categorical(df['Race']).codes
df['race_category'].value_counts()
df['Race'].value_counts()
df.info()

melted_df2 = pd.melt(df,id_vars=['race_category','Gender','Age','Year','index_col','Residence.City'],value_vars=['Cocaine', 'Fentanyl','Fentanyl.Analogue', 'Oxycodone', 'Oxymorphone', 'Ethanol','Hydrocodone', 'Benzodiazepine', 'Methadone', 'Meth.Amphetamine','Amphet', 'Tramad', 'Hydromorphone', 'Morphine..Not.Heroin.','Xylazine', 'Gabapentin', 'Opiate.NOS', 'Heroin.Morph.Codeine','Other.Opioid', 'Any.Opioid', 'Other'],var_name='Substance',value_name='Indicator')
melted_df2 =melted_df2.dropna()


melted_df2['Substance_Category'] = pd.Categorical(melted_df2['Substance']).codes
category_code_dict = dict(zip(melted_df2['Substance'], pd.Categorical(melted_df2['Substance']).codes))

melted_df2['Residence.City.Category'] = pd.Categorical(melted_df2['Residence.City']).codes
residence_code_dict = dict(zip(melted_df2['Residence.City'], pd.Categorical(melted_df2['Residence.City']).codes))

melted_df2 =melted_df2.drop(['Residence.City','Substance','Indicator'],axis=1)

##########################
# Generating a heatmap to study the correlation of variables linearly.
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(melted_df2.corr(),annot=True)
plt.show()

# It is observed that there is almost negligible linear relationship between the variables, hence
# we cannot use linear regression with this data.


# Proceeding forward with the random forest technique to predict the substance category used
# if the input parameters passed are race_category, Gender, Age, Year and the Residence City Category


x_data = melted_df2[['race_category','Gender','Age','Residence.City.Category']]
y_data = melted_df2[['Substance_Category']]
y_data = y_data.values.ravel()
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=30)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(x_train, y_train)


y_pred = rf_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}\n")

# Visualize Feature Importances
features = ['race_category', 'Gender', 'Age','Residence.City.Category']
feature_importances = pd.Series(rf_model.feature_importances_, index=features)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()


