# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



DataSetFolder = "C:/Users/behzad/Dropbox/6 EduMaterial_PYTHON_DataAnalysis/Data-Analysis-with-Python-Pipeline/DataSet"
Formatted_file_name_after_Step2= "car_dataset_formatted_after_step2.csv"
file_path="/".join([DataSetFolder,Formatted_file_name_after_Step2])
df = pd.read_csv(file_path)

df.head()

# finding the correaltion between different parameters
# Let' first review the format of different files

df.dtypes

# Then for the numeric columns we can use .corr to find the correlation 
df.corr()

#We should pay attention that in the results all of the string (object) columns were automatically removed.

#we can also find the correlation matrix only for some parameters, having a look on the correaltion matrix, 
#we can observe that engine-size, highway-L/100km , and horsepower are notably correlated with the price, let's have a look on the correlation of these variables with the price
df[["engine-size","highway-L/100km", "horsepower","price"]].corr()


#ScatterPlot: Visualising correlation between numeric variables¶

#In order to visualise the correaltion between two numeric variables we can use scatter plots.



#In order to visualise the correaltion between two numeric variables we can use scatter plots.
plt.scatter(df["engine-size"],df["price"])
plt.xlabel("engine size")
plt.ylabel("price")
plt.title("Price Vs. Engine-size")

#An alternative appraoch is using seaborn's regplot which also adds plot a regression line
sns.regplot(data=df,x="engine-size",y="price")

#Let's crete a similar plot to see the correlation of price and city-mpg
sns.regplot(data=df,x="city-mpg",y="price")


#We can clearly observe a negative correlation, we can verify it by calculating the corresponding correlation

df[["city-mpg","price"]].corr()
#Finally,we can observe the correlation between peak-rpm, which are poorly correlated, the fact that it is also verified by calculating their correaltion coefficient.

sns.regplot(data=df,x="peak-rpm",y="price")
df[["peak-rpm","price"]].corr()


## Finding the correlation between categorical variables  and numeric variables
#categorical variables are the ones that describe a property of a data unitand can have the type "object" or "int64". A promising way to visualize categorical variables is by using boxplots.

sns.boxplot(x="body-style", y="price", data=df)
