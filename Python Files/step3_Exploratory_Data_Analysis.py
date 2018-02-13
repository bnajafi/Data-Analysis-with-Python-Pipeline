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



"""
 sometimes we would like to know the significant of the correlation estimate.

P-value: What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.

By convention, when the

    p-value is < 0.001 we say there is strong evidence that the correlation is significant,
    the p-value is < 0.05; there is moderate evidence that the correlation is significant,
    the p-value is < 0.1; there is weak evidence that the correlation is significant, and
    the p-value is > 0.1; there is no evidence that the correlation is significant.

"""
from scipy import stats

#Let't's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'. 

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient of wheel-based Vs. price is", pearson_coef, " with a P-value of P =", p_value)

#Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585)


pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient of horsepower Vs. price is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient of length Vs. price is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient of width Vs. price  is", pearson_coef, " with a P-value of P =", p_value )

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient of curb-weight Vs. price is", pearson_coef, " with a P-value of P =", p_value)  


pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient of engine-size Vs. price is", pearson_coef, " with a P-value of P =", p_value) 

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient of bore Vs. price is", pearson_coef, " with a P-value of P =", p_value ) 

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient of city-mpg Vs. price is", pearson_coef, " with a P-value of P =", p_value) 

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient of highway-mpg Vs. price is", pearson_coef, " with a P-value of P =", p_value ) 

## Finding the correlation between categorical variables  and numeric variables
#categorical variables are the ones that describe a property of a data unitand can have the type "object" or "int64". A promising way to visualize categorical variables is by using boxplots.
plt.figure()
sns.boxplot(x="body-style", y="price", data=df)
plt.figure()
sns.boxplot(data=df,x="engine-location",y="price")
plt.figure()
sns.boxplot(data=df,x="drive-wheels",y="price")
plt.figure()
sns.boxplot(x="horsepower-binned",y="price",data=df)


#Descriptive statistics for categorical variables

#We saw that we can use .describe() to have a statistical summary of the dataset


df.describe()

# However, .describe() ignores every variable which is an object, to include them we use (include=["object"])

df.describe(include=["object"])

#Value counts¶

#as we saw value counts is a promising way of describing categorical variables
df["drive-wheels"].value_counts()
# We can convert it into a dataframe
drive_wheels_count = df["drive-wheels"].value_counts().to_frame()

# We can also rename the main column to value counts and rename the index name to drive wheels
drive_wheels_count.rename({"drive-wheels":"value counts"},inplace=True)
drive_wheels_count.index.name = "drive-wheels"

drive_wheels_count

# we can next do the same procedure for engine location
# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

