import numpy as np
import pandas as pd 
#In this notebook we are going through the basics of data wrangling using pandas employing the same dataset as the one of the first step. 
#In order to import the dataset, we use read_csv again, with the only difference that this time, we add the header while importing the dataset

url = url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
givenHeader = ["symboling","normalized-losses","make","fuel-type","aspiration",
"num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length",
"width","height","curb-weight","engine-type","num-of-cylinder","engine-size",
"fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm",
"city-mpg"," highway-mpg","price"]


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
    
    
    
# Before proceeding, as we previously saw the data format of some rows should be changed. First , let's have a look
df.dtypes

# So we can see that the format of bore, stroke, price, peak-rpm, should be converted to "float"
# the foramt of normalized-losses instead should be con eerted to int  so:
df[["bore","stroke","price", "peak-rpm"]] =  df[["bore","stroke","price", "peak-rpm"]].astype("float")
df["normalized-losses"] =  df["normalized-losses"].astype("int")
df.dtypes

