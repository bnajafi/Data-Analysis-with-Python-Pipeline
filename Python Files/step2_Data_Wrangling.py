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

df.tail(100)

# Now we would like to see the ways we can deal with the missing data
# First, we should see how to identify the missing data

# by having a look on the dataset we can see that there are several cells in which we have
# the symbol "?" which represents the missing value, so the first step is replacing "?" with nan.
#In order to so we can do it either for a certain column,or for the whole dataset. 
#In order to do the latter for a certain column, we can use list comprehension
# so for example for column 

#In order to do the latter for a certain column, we can use list comprehension: So first let's find the boolean vector of the "normalized-losses" column being equal to "?"

df["normalized-losses"][df["normalized-losses"]=="?"] = np.nan

