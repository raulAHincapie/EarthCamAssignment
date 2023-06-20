#!/usr/bin/env python
# coding: utf-8

# # EarthCam Data Science Assignment - Raul Hincapie's Code
# #### Raul Hincapie
# #### 6-20-2023

# ## Packages/Libraries

# In[1]:


import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from numpy.linalg import norm
from pathlib import Path
from tqdm import tqdm

from statistics import mean
import pandas as pd
import numpy as np
import itertools
import calendar
import datetime
import sklearn
import string
import scipy
import glob
import math
import time
import ast
import csv
import os

# for the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# ## Files within folder

# In[2]:


# Accessing all files within folder. Helps to see naming convention of file to convert to dataframe
#currentListOfExcels = list(glob.glob("**/*.xlsx", recursive = True))
#print(currentListOfExcels)


# ## Accessing the data

# In[3]:


# Extracting data from data excel workbook
dataDF = pd.read_excel('Data.xlsx', sheet_name = 'Data')
display(dataDF)


# ## Columns being kept

# In[4]:


# Determined the following columns were necessary for the assignment. Extracted using Excel
# These columns contain necessary data for the model and could give better insight
# when predicting 2024 weather data for any specific day and month.

keepingColumns = pd.read_excel('Data.xlsx', sheet_name = 'Columns to keep')
keepingColumns = list(keepingColumns['Columns'])
print(keepingColumns)


# ## DataFrame containing ONLY columns being kept

# In[5]:


# Created a new dataframe to only contain these columns
newDataDF = dataDF[keepingColumns]

# Removed any duplicate days from the "ObservedAt_DateTime" column
# since the goal is to maintain uniqueness for the model
newDataDF.drop_duplicates(subset = 'ObservedAt_DateTime', inplace = True)

display(newDataDF)


# ## Converting string dictionary columns into actual dictionaries

# In[6]:


# Certain columns above contain string dictionaries, so they're extracted into a list below
dictionaryColumns = [keepingColumns[i] for i in [1,2,3,4,5,6,7,8,13]]
print(dictionaryColumns)


# In[7]:


# To convert them to be dictionaries, "ast.literal_eval" is used on each column
for eachDictCol in dictionaryColumns:
    newDataDF[eachDictCol] = newDataDF[eachDictCol].apply(ast.literal_eval)
    
    # One column contained a dictionary within a list, so it was extracted
    if type(newDataDF[eachDictCol][0]) == list:
        for eachIndex in list(newDataDF.index):
            newDataDF[eachDictCol][eachIndex] = newDataDF[eachDictCol][eachIndex][0] 


# ## Extracting all dictionary data

# In[8]:


# Created a deep copy of "newDataDF" so the original DataFrame isn't lost
newDataDFcopy = newDataDF.copy(deep = True)


# In[9]:


# Created new columns to be able to extract the data efficiently

# Going through each dictionary column and extracting keys
for eachDictCol in dictionaryColumns:
    
    # Current keys of current dictionary column
    keysFromCurrentColDict = list(newDataDFcopy[eachDictCol][0].keys())

    # Going through each key and creating a new empty column
    for eachKey in keysFromCurrentColDict:
        
        
        
        # New column name based on dictionary column name, a dash, then the key's name
        newColName = eachDictCol + " - " + eachKey
        
        print(newColName)
        
        # New empty column based on newColName
        newDataDFcopy[newColName] = ''
        
        
        
        # Extracting the data then dumping it into the new column above
        for eachIndex in list(newDataDFcopy.index):
            
            
            
            # Current dictionary based on dictionary column then its index
            currentDictionary = newDataDFcopy[eachDictCol][eachIndex]
            
            # Current key value based on dictionary above
            currentDictionaryKeyValue = currentDictionary[eachKey]  
            
            # Putting the value in its new corresponding column and its index
            newDataDFcopy[newColName][eachIndex] = currentDictionaryKeyValue
        
        
        
    newDataDFcopy.drop(eachDictCol, axis = 1, inplace = True)


# ## Columns with embedded dictionaries

# In[10]:


for eachColumn in list(newDataDFcopy.columns):
    
    # If dictionary values still exist within a column,
    # extract the data once again. Tried to implement this
    # in the code above but it was too computationally intensive
    
    if type(newDataDFcopy[eachColumn][0]) == dict:
        
        # Inner keys of dictionary
        innerDictionaryKeys = list(newDataDFcopy[eachColumn][0].keys())
        
        # Looping through inner keys and create columns and extract data
        for eachInnerKey in innerDictionaryKeys:
    
    
    
            # New column name for inner dictionary values:
            newInnerColumnName = eachColumn + " - " + eachInnerKey
            
            
            print(newInnerColumnName)
            
            
            
            # Placing it within the dataframe
            newDataDFcopy[newInnerColumnName] = ''
            
    
    
            # Extracting the data then dumping it into new column above
            for eachIndex in list(newDataDFcopy.index):



                # Current inner dictionary based on dictionary name, its index, and its outer key
                currentInnerDictionaryKeyValue = newDataDFcopy[eachColumn][eachIndex][eachInnerKey]

                # Putting the inner value in its new corresponding column and its index                
                newDataDFcopy[newInnerColumnName][eachIndex] = currentInnerDictionaryKeyValue
                
        newDataDFcopy.drop(eachColumn, axis = 1, inplace = True)


# ## Extracting datetime data

# In[11]:


# Extracting the date, hour, seconds, and milliseconds from the datetime column
newDataDFcopy['Date'] = newDataDFcopy['ObservedAt_DateTime'].dt.strftime('%m-%d-%Y')
newDataDFcopy['Hour'] = newDataDFcopy['ObservedAt_DateTime'].dt.hour
newDataDFcopy['Minute'] = newDataDFcopy['ObservedAt_DateTime'].dt.minute
newDataDFcopy['Second'] = newDataDFcopy['ObservedAt_DateTime'].dt.second

# Dropped the original datetime column to avoid confusion
newDataDFcopy.drop('ObservedAt_DateTime', axis = 1, inplace = True)

# Shifting the new columns to the front of the dataframe
newDataDFcopy = pd.concat([newDataDFcopy[['Date', 'Hour', 'Minute', 'Second']], newDataDFcopy.drop(['Date', 'Hour', 'Minute', 'Second'], axis = 1)], axis = 1)

display(newDataDFcopy)


# ## Extracting dates that do not have four weather instances

# In[12]:


uniqueDates = list(set(newDataDFcopy['Date']))

doesNotHaveFourInstances = []

for eachUniqueDate in uniqueDates:
    if len(newDataDFcopy[newDataDFcopy['Date'] == eachUniqueDate]) != 4:
        doesNotHaveFourInstances.append(eachUniqueDate)


# In[13]:


doesNotHaveFourInstancesDF = newDataDFcopy[newDataDFcopy['Date'].isin(doesNotHaveFourInstances)].copy(deep = True)
doesNotHaveFourInstancesDF.reset_index(drop = True, inplace = True)
display(doesNotHaveFourInstancesDF)


# ## Removing days without four weather instances from main copied DF

# In[14]:


newDataDFcopy.drop(newDataDFcopy[newDataDFcopy['Date'].isin(doesNotHaveFourInstances)].index, inplace = True)
newDataDFcopy.reset_index(drop = True, inplace = True)
display(newDataDFcopy)


# In[15]:


# Columns being used for the model
numericalDataColumns = list(newDataDFcopy.columns)[4:]


# In[16]:


# Excel created from dataframe above in order to extract unique dates
# from the 'Date' column
#newDataDFcopy.to_excel('Only Four Instances.xlsx', sheet_name = 'Data')


# ## Finding averages/modes of each numerical column

# #### Extracting unique dates from data with only four instances

# In[17]:


# Dataframe containing only unique dates with only four weather instances. 
# This has been extracted via Excel using the UNIQUE function, then placed in "Unique Dates" Excel worksheet
onlyFourInstancesUniqueDatesDF = pd.read_excel('Only Four Instances.xlsx', sheet_name = 'Unique Dates')

# Creating new dataframe to contain unique dates and same columns from numericalDataColumns list
numericalDataColumnsDF = pd.DataFrame({header: [''] * len(onlyFourInstancesUniqueDatesDF) for header in numericalDataColumns})

# Concat unique dates DF and the empty column DF
onlyFourInstancesUniqueDatesDF = pd.concat([onlyFourInstancesUniqueDatesDF, numericalDataColumnsDF], axis = 1)

# Setting 'Date' column as the index
onlyFourInstancesUniqueDatesDF.set_index('Date', inplace = True)


# #### Finding the averages based on unique dates

# In[18]:


# Iterates through each unique date and column
for eachUniqueDate in list(onlyFourInstancesUniqueDatesDF.index):
    for eachColumnName in list(onlyFourInstancesUniqueDatesDF.columns):
        
        
        # Acquires unique values per each date and also column name
        uniqueElementsInColumn = list(set(newDataDFcopy[newDataDFcopy['Date'] == eachUniqueDate][eachColumnName]))
        

        # If an element in the list above is string, only input a unique list of values
        if any(isinstance(element, str) for element in uniqueElementsInColumn):
            
            onlyFourInstancesUniqueDatesDF[eachColumnName][eachUniqueDate] = list(set(newDataDFcopy[newDataDFcopy['Date'] == eachUniqueDate][eachColumnName]))
        
        # If an element is of type None exists in uniqueElementsInColumn, remove all Nones and find average
        # If only contains None, only input None
        elif any(isinstance(element, (str, type(None))) for element in uniqueElementsInColumn):
            
            currentList = list(set(newDataDFcopy[newDataDFcopy['Date'] == eachUniqueDate][eachColumnName]))
            filteredList = [num for num in currentList if num is not None and isinstance(num, float)]
            
            if len(filteredList) == 0:
                onlyFourInstancesUniqueDatesDF[eachColumnName][eachUniqueDate] = None
            else:
                onlyFourInstancesUniqueDatesDF[eachColumnName][eachUniqueDate] = mean(filteredList)
            
        # This condition handles all numerical data so it acquires the necessary mean
        else:
            
            onlyFourInstancesUniqueDatesDF[eachColumnName][eachUniqueDate] = mean(list(set(newDataDFcopy[newDataDFcopy['Date'] == eachUniqueDate][eachColumnName])))


# #### Removing columns with only None values

# In[19]:


for eachCol in onlyFourInstancesUniqueDatesDF.columns:
    if onlyFourInstancesUniqueDatesDF[eachCol].isnull().all():
        print(f"Column '{eachCol}' has only None values.")
        onlyFourInstancesUniqueDatesDF.drop([eachCol], axis = 1, inplace = True)


# ## Exploratory Data Analysis

# #### Average temperatures per date

# In[20]:


# Convert the index values to be datetime for the plot
xAxisDates = pd.to_datetime(onlyFourInstancesUniqueDatesDF.index, format = '%m-%d-%Y')

# Adjusting the plot size
plt.figure(figsize = (15, 5))

# Plotting the average Fahrenheit values based on unique dates
plt.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Temperature - Fahrenheit'], label = 'Fahrenheit')

# x and y axis label, its title, and the legend
plt.xlabel('Unique dates')
plt.ylabel('Average temperature (in degrees)')
plt.title('Average temperature per unique date - May 2018 to May 2023')
plt.legend()

#Showing the final graph
plt.show()


# #### Average temperature & average relative humidity per date

# In[21]:


# Adjusting the plot size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 10))



# Plotting the average Fahrenheit values based on unique dates in subplot 1
ax1.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Temperature - Fahrenheit'], label = 'Fahrenheit')
ax1.set_ylabel('Average temperature (in degrees)')
ax1.set_title('Average temperature per unique date - May 2018 to May 2023')



# Plotting the average relative humidity values based on unique dates in subplot 2
ax2.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['RelativeHumidity'], color = 'Green', label = 'Relative humidity')
ax2.set_xlabel('Unique dates')
ax2.set_ylabel('Average relative humidity')
ax2.set_title('Average relative humidity per unique date - May 2018 to May 2023')


# Plotting legend within each subplot
ax1.legend()
ax2.legend()

# Showing the final graph
plt.show()


# #### Average temperature & average pressure in inches of mercury

# In[22]:


# Adjusting the plot size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 10))

# Plotting the average Fahrenheit values based on unique dates in subplot 1
ax1.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Temperature - Fahrenheit'], label = 'Fahrenheit')
ax1.set_ylabel('Average temperature (in degrees)')
ax1.set_title('Average temperature per unique date - May 2018 to May 2023')

# Plotting the average pressure in inches of mercury values based on unique dates in subplot 2
ax2.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Pressure - InchesOfMercury'], color = 'Red', label = 'Pressure')
ax2.set_xlabel('Unique dates')
ax2.set_ylabel('Average pressure in inches of mercury')
ax2.set_title('Average pressure in inches of mercury per unique date - May 2018 to May 2023')


# Plotting legend within each subplot
ax1.legend()
ax2.legend()

# Showing the final graph
plt.show()


# #### Average temperature & average visibility in miles

# In[23]:


# Adjusting the plot size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 10))

# Plotting the average Fahrenheit values based on unique dates in subplot 1
ax1.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Temperature - Fahrenheit'], label = 'Fahrenheit')
ax1.set_ylabel('Average temperature (in degrees)')
ax1.set_title('Average temperature per unique date - May 2018 to May 2023')


# Plotting the average visibility in miles values based on unique dates in subplot 2
ax2.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Visibility - Miles'], color = 'Orange', label = 'Visibility')
ax2.set_xlabel('Unique dates')
ax2.set_ylabel('Average visibility in miles')
ax2.set_title('Average visibility in miles per unique date - May 2018 to May 2023')



# Plotting legend within each subplot
ax1.legend()
ax2.legend()

# Showing the final graph
plt.show()


# #### Average temperature & average wind speed in miles per hour

# In[24]:


# Convert the index values to be datetime for the plot
xAxisDates = pd.to_datetime(onlyFourInstancesUniqueDatesDF.index, format = '%m-%d-%Y')

# Adjusting the plot size
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 10))


# Plotting the average Fahrenheit values based on unique dates in subplot 1
ax1.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Temperature - Fahrenheit'], label = 'Fahrenheit')
ax1.set_ylabel('Average temperature (in degrees)')
ax1.set_title('Average temperature per unique date - May 2018 to May 2023')


# Plotting the average wind speed in miles per hour values based on unique dates in subplot 2
ax2.plot(xAxisDates, onlyFourInstancesUniqueDatesDF['Wind - Speed - MilesPerHour'], color = 'violet', label = 'Wind speed')
ax2.set_xlabel('Unique dates')
ax2.set_ylabel('Average wind speed in miles per hour')
ax2.set_title('Average wind speed in miles per hour per unique date - May 2018 to May 2023')



# Plotting legend within each subplot
ax1.legend()
ax2.legend()

# Showing the final graph
plt.show()


# ## Linear Regression Model

# In[25]:


# New DF as copy from DF above
modelDF = onlyFourInstancesUniqueDatesDF.copy(deep = True)

# Converting index into datetime format
modelDF.index = pd.to_datetime(modelDF.index, format = '%m-%d-%Y')

# Day, Month, and dayOfYear columns created based on index data
modelDF['Day'] = modelDF.index.day
modelDF['Month'] = modelDF.index.month
modelDF['dayOfYear'] = modelDF.index.dayofyear
modelDF['Year'] = modelDF.index.year


# In[26]:


# Creating training and testing sets
X = modelDF[['Day', 'Month', 'dayOfYear']]
yFahrenheit = modelDF['Temperature - Fahrenheit']
yCelsius = modelDF['Temperature - Celsius']

X_train, X_test, y_fTrain, y_fTest, y_cTrain, y_cTest = train_test_split(X, yFahrenheit, yCelsius, test_size = 0.2, random_state = 42)



# In[27]:


# Training model for Fahrenheit
modelF = LinearRegression()
modelF.fit(X_train, y_fTrain)

# Training the model for Celsius
modelC = LinearRegression()
modelC.fit(X_train, y_cTrain)


# In[28]:


# Prediction function
def predictTemperature(day, month):
    dayOfYear = pd.to_datetime(f'{month}-{day}', format = '%m-%d').dayofyear
    inputData = [[day, month, dayOfYear]]
    
    temperaturePredF = modelF.predict(inputData)[0]
    temperaturePredC = modelC.predict(inputData)[0]
    
    return temperaturePredF, temperaturePredC


# In[45]:


# user input
while True:
    try:
        day = int(input("Please input a day: "))
        month = int(input("Please input a month: "))
        print()
        
        temperaturePredF, temperaturePredC = predictTemperature(day,month)
        
        if temperaturePredF is not None and temperaturePredC is not None:
            print("2024 predicted temperature for Joliet, IL in Fahrenheit:", temperaturePredF.round(2))
            print("2024 predicted temperature for Joliet, IL in Celsius:", temperaturePredC.round(2))
            print()
            
        break
    except ValueError:
        print("Invalid input, please try again.")
        print()

# displays prior yearly data for input above
userInputDF = modelDF.copy(deep = True)
userInputDF = userInputDF[(userInputDF['Month'] == month) & (userInputDF['Day'] == day)]
userInputDF.drop(['Day', 'Month', 'dayOfYear', 'Year'], axis = 1, inplace = True)
userInputDF.index = pd.to_datetime(userInputDF.index).year
userInputDF = userInputDF.rename_axis('Year', axis = 'index')

print()
print('\033[1m' + 'Yearly Data for ' + str(month) + "-" + str(day) + '\033[0m')

display(userInputDF.T)

