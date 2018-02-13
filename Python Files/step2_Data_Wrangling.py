import numpy as np
import pandas as pd 
#In this notebook we are going through the basics of data wrangling using pandas employing the same dataset as the one of the first step. 
#In order to import the dataset, we use read_csv again, with the only difference that this time, we add the header while importing the dataset

url = url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
givenHeader = ["symboling","normalized-losses","make","fuel-type","aspiration",
"num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length",
"width","height","curb-weight","engine-type","num-of-cylinder","engine-size",
"fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm",
"city-mpg","highway-mpg","price"]


df = pd.read_csv(url, names =givenHeader )

# we can now have a look on the first rows of it.

df.head()


# Now we would like to see the ways we can deal with the missing data
# First, we should see how to identify the missing data

# by having a look on the dataset we can see that there are several cells in which we have
# the symbol "?" which represents the missing value, so the first step is replacing "?" with nan. which is Python's default missing value marker
#In order to so we can do it either for a certain column,or for the whole dataset. 
#In order to do the latter for a certain column


#In order to do the latter for a certain column, we can use list comprehension: So first let's find the boolean vector of the "normalized-losses" column being equal to "?"
df["normalized-losses"]=="?"

#we can now replace those with np.nan usin

df["normalized-losses"][df["normalized-losses"]=="?"] = np.nan

# An easier approach clearly is using repalce function:
 
df.replace("?",np.nan,inplace=True)
df.head()

# Evaluating for Missing Data
# The next step is to identify the missing values. In order to do so, we can use .isnull and .notnull the ouput of which are boolean values

missingValues = df.isnull()
missingValues.head()

# Count missing values in each column
#now that we have the boolean equivalent we can use the .value_counts() in order to count the number of True values for each column
#so for example for the second  columsn  

missingValues["normalized-losses"].value_counts()

result = missingValues["normalized-losses"].value_counts()

# So we can now write a for loop that find the number of missing values of each column
missing_value_count=pd.Series()
for column in missingValues.columns: 
    if True in missingValues[column].value_counts().index:
        missing_value_count[column]= missingValues[column].value_counts()[True]
    else:
        missing_value_count[column] = 0

        
# How to deal with missing values
# whiel dealing with missing values, we have several options 
# 1 droping the data which can be droping the whole row or dropping the whole column:
# droping the whole column only make sense in case most of the entries in that column are missing which is not the case for our dataset.

# 2 the second apprach is replacing the data: whcih can involve replacing it by the mean of that column, replacing it with the most frequent item of  of the colmn or replacing using other functions
# each of these approaches can be used based on the properties of the data given in each column 

"""
 We will apply each method to many different columns:

Replace by mean:
"normalized-losses": 41 missing data, replace them with mean
"stroke": 4 missing data, replace them with mean
"bore": 4 missing data, replace them with mean
"horsepower": 2 missing data, replace them with mean
"peak-rpm": 2 missing data, replace them with mean

Replace by frequency:

"num-of-doors": 2 missing data, replace them with "four". 
    * Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to 

Drop the whole row:

"price": 4 missing data, simply delete the whole row
    * Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore they are not useful to us
    
    """
# So first let's first calculate the avarage of normalized losses

avg_normalized_losses = df["normalized-losses"].astype("float").mean()
# Now we should use replace to replace the nan values with the determiend average
df["normalized-losses"].replace(np.nan, avg_normalized_losses,inplace=True)

# We can do the same for the stroke, bore, peak-rpm    
    
avg_stroke = df["stroke"].astype("float").mean()
df["stroke"].replace(np.nan, avg_stroke,inplace=True)

avg_bore = df["bore"].astype("float").mean()
df["bore"].replace(np.nan, avg_bore,inplace=True)

avg_horsepower= df["horsepower"].astype("float").mean()
df["horsepower"].replace(np.nan, avg_horsepower,inplace=True)

avg_horsepower= df["horsepower"].astype("float").mean()
df["horsepower"].replace(np.nan, avg_horsepower,inplace=True)

avg_peak_rpm= df["peak-rpm"].astype("float").mean()
df["peak-rpm"].replace(np.nan, avg_peak_rpm,inplace=True)




# Now let's see what we can do with the missing values of categorical variables: number of doors
# To see number of each values present in a paarticular column as we saw we can use .value_counts()
df["num-of-doors"].value_counts()
MostFrquent_num_of_doors = df["num-of-doors"].value_counts().idxmax()

df["num-of-doors"].replace(np.nan, MostFrquent_num_of_doors, inplace=True)

# For the case of the price, we simply drop the whole rows for which we do not have the value
df.dropna(subset=["price"],axis=0, inplace=True)

# We have to reset the index because we dropped some rows
df.reset_index(drop=True, inplace=True)

df.head()


# to make sure that we have correctly dealt with all of the values, we can see the following
missingValues = df.isnull() 
missing_value_count_toCheck=pd.Series()
for column in missingValues.columns: 
    if True in missingValues[column].value_counts().index:
        missing_value_count_toCheck[column]= missingValues[column].value_counts()[True]
    else:
        missing_value_count_toCheck[column] = 0
missing_value_count_toCheck

# We can see that no missing value has remained in the dataset
    
# Before proceeding, as we previously saw the data format of some rows should be changed. First , let's have a look
df.dtypes

# So we can see that the format of bore, stroke, price, peak-rpm, should be converted to "float"
# the foramt of normalized-losses instead should be con eerted to int  so:
df[["bore","stroke","price", "peak-rpm"]] =  df[["bore","stroke","price", "peak-rpm"]].astype("float")
df["normalized-losses"] =  df["normalized-losses"].astype("int")
df.dtypes

# NOw let's standardize some of the columns, we would like to convert city-mpg and highway-mpg into L/100km units, to do so:
df["city-L/100km"] = 235/df["city-mpg"]
df["highway-L/100km"] = 235/df["highway-mpg"]

df.head()

# In the next stepl we can go through some data normalization procedure in which we convert the height, width and length to values between 0 and 1.
# To do so, we simply divide the values of each column by the maximum value of that column

df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

"""
##Binning

Binning is a process of transforming continuous numerical variables into discrete categorical 'bins'.
An example, in our dataset, we can categorize the cars which have high, medium or low horsepower,
so we have to convert the horsepower variable which is a continuos number into three categorical variables.

"""
 # To do so we should first calculate the width of a bin
 
df["horsepower"]=df["horsepower"].astype(float, copy=True)
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
bins = np.arange(min(df["horsepower"]),max(df["horsepower"]),binwidth)

groupNames = ["Low","Medium", "High"]

df["horsepower-binned"]= pd.cut(df["horsepower"],bins, labels=groupNames, include_lowest= True)
df[["horsepower","horsepower-binned"]].head(30)

# Another way of showing the numbers in a binned way, is employing histogram

import matplotlib.pyplot as plt

# draw historgram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")


# Converting unique values into dummy(indicator) variables
# In order to use the unquie values of some columns as categorical variables which are later used in the regression analysis, we have to first convert them into dummy variables
# dummy variables are numerical variables that represent categories.
#Example, the column "fuel-type" has two unique values, "gas" or "diesel". 
#Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.
dummy_variables_fuelType= pd.get_dummies(df["fuel-type"])
dummy_variables_fuelType.head(5)
dummy_variables_fuelType.rename(columns={"gas":'fuel-type-diesel',"diesel":'fuel-type-diesel'}, inplace=True)

df= pd.concat([df, dummy_variables_fuelType],axis=1)
df.head()

# We can finally save this formatted dataset
import os
os.chdir("C:/Users/behzad/Dropbox/6 EduMaterial_PYTHON_DataAnalysis/Data-Analysis-with-Python-Pipeline/DataSet")

df.to_csv("car_dataset_formatted_after_step2.csv")