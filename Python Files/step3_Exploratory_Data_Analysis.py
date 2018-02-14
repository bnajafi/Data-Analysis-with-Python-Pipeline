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


# Grouping : Group by
# The "groupby" method groups data by different categories. The data is grouped based on one or several variables and analysis is performed on the individual groups.
#or example, let's group by the variable "drive-wheels". We see that there are 3 different categories of drive wheels.

df["drive-wheels"].unique()

#f we want to know, on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them.

#we can select the columns 'drive-wheels','body-style' and 'price' , then assign it to the variable "df_group_one".


df_three_columns = df[["drive-wheels","body-style","price"]]

df_three_columns_groupby_drive_wheels = df_three_columns.groupby(["drive-wheels"],as_index=  False).mean()

df_three_columns_groupby_drive_wheels

df_groupby_driveWheels_and_bodyStyle = df_three_columns.groupby(["drive-wheels", "body-style"],as_index=  False).mean()

df_groupby_driveWheels_and_bodyStyle_pivot= df_groupby_driveWheels_and_bodyStyle.pivot(index= "drive-wheels", columns = "body-style")

# in order to visualize the created pivot table in a better way, we can use a heat map.
plt.figure()
plt.pcolor(df_groupby_driveWheels_and_bodyStyle_pivot,cmap="RdBu")
plt.colorbar()
plt.show()


# in order to improve this presentation, we can do the following:
fig, ax=plt.subplots()
im=ax.pcolor(df_groupby_driveWheels_and_bodyStyle_pivot, cmap='RdBu')
#label names
row_labels=df_groupby_driveWheels_and_bodyStyle_pivot.columns.levels[1]
col_labels=df_groupby_driveWheels_and_bodyStyle_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(df_groupby_driveWheels_and_bodyStyle_pivot.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(df_groupby_driveWheels_and_bodyStyle_pivot.shape[0])+0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
"""

Anova (Analysis of Variance): Determining the correaltion of a categorical variable with a numerical target

The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups. 
ANOVA returns two parameters: F-test score: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, 
and reports it as the F-test score. A larger score means there is a larger difference between the means. P-value: P-value tells how statistically significant is our calculated score value 
If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value.
"""
#Extracting the groups using groupby()

#The first step is extracting a group using groupby(): to do so we can use the get_group() method.


df_driveWheels_price = df[["drive-wheels","price"]]
df_price_groupedby_drive_wheels = df_driveWheels_price.groupby(["drive-wheels"])
df_price_groupedby_drive_wheels.head(2)



df_price_groupedby_drive_wheels.get_group("4wd")["price"]


# implement anova for three groups

f_val, p_val = stats.f_oneway(df_price_groupedby_drive_wheels.get_group('fwd')['price'], df_price_groupedby_drive_wheels.get_group('rwd')['price'], df_price_groupedby_drive_wheels.get_group('4wd')['price'])  
 
print( "ANOVA results for fwd, rwd, and 4wd : F=", f_val, ", P =", p_val)   

# Now conducting Anova for the groups one by one 
f_val, p_val = stats.f_oneway(df_price_groupedby_drive_wheels.get_group('fwd')['price'],  df_price_groupedby_drive_wheels.get_group('4wd')['price'])  
 
print( "ANOVA results for fwd and 4wd: F=", f_val, ", P =", p_val) 

f_val, p_val = stats.f_oneway(df_price_groupedby_drive_wheels.get_group('fwd')['price'], df_price_groupedby_drive_wheels.get_group('rwd')['price'])  
 
print( "ANOVA results for fwd and rwd: F=", f_val, ", P =", p_val)  
 
f_val, p_val = stats.f_oneway(df_price_groupedby_drive_wheels.get_group('rwd')['price'], df_price_groupedby_drive_wheels.get_group('4wd')['price'])  
 
print( "ANOVA results for rwd, and 4wd : F=", f_val, ", P =", p_val) 

