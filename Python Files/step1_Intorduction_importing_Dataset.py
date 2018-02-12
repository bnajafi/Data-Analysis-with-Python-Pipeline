import numpy as np
import pandas as pd

url = url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url,header=None)

df.head(10)
df.tail()

newHeader = ["symboling","normalized-losses","make","fuel-type","aspiration",
"num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length",
"width","height","curb-weight","engine-type","num-of-cylinder","engine-size",
"fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm",
"city-mpg"," highway-mpg","price"]

df.columns = newHeader


df.head()
df.columns


# Now let's save the updates dataset as a new csv file

df.to_csv("uCarDataset.csv")

# How to check the data type of different columsn

df.dtypes

# so there are plenty of objects whose data type should be changed, we are going to go through this task in the next step

# Let's see a statistical description of the dataset

df.describe()

# in case we would like to add more parameters specifically the ones corresponding to the categorical variable we should use include="all" argument

df.describe(include="all")

# finally in order to have a concise summary of the dataset we can use df.info()

df.info()