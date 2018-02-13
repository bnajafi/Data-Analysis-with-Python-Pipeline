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