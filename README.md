# ODD2023-Datascience-Ex06
## EX-06 FEATURE TRANSFORMATION
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.
### EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### ALGORITHM:
-  Step1: Read the given Data.
-  Step2: Clean the Data Set using Data Cleaning Process.
-  Step3: Apply Feature Transformation techniques to all the features of the data set.
-  Step4: Print the transformed features.
## PROGRAM:
```
Developed by : Sabitha P
Register No : 212222040137
```
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

df.skew()

df.head()

df.isnull().sum()

df.info()
df.describe()

np.log(df["Highly Positive Skew"])
np.reciprocal(df["Highly Positive Skew"])
np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df1=df.copy()
sm.qqplot(df["Moderate Negative Skew"],fit=True,line='45')
plt.show()

sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],fit=True,line='45')
plt.show()

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
## OUTPUT:
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/f41763b2-529b-4a5a-812f-1598cb0d98a4)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/643f7cd0-8bfc-4617-8e76-ecbd5ef4498f)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/c4a60ef7-5d74-45e7-824c-a3e9ce69bc33)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/1201350c-1404-4bbf-9f4f-3e938573e33d)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/111052a0-ad4f-40c7-a90c-14aa90fcce38)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/ad5281e7-fc91-4f6c-aef9-959fb51ac788)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/71e7895f-6384-4a1d-834b-eacfda734c8b)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/dd6ae654-1c1c-4a35-9b3a-dd636e9a0370)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/beb097b9-8d55-4d80-87f3-3126a8d74fd8)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/a4db3b77-403a-4612-942b-258c2c525010)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/70b86f7f-edca-4acd-be4c-8c650519ec5a)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/5b81722a-2481-4866-85d8-8dd99efd7421)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/0cc93dac-a7a3-424e-97ee-cf9d7b6fc217)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/dfba67cc-c761-4b3c-b12d-2c7bdde66aea)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/d3057b6e-258a-4ac5-af01-0ebbd8a59083)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/37c8df93-c8bd-4757-8131-fc2cc201de99)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/00fa14b6-1e25-4353-8217-8472ba3b0587)
![image](https://github.com/sabithapaulraj/ODD2023-Datascience-Ex06/assets/118343379/1721a7eb-b50a-418e-bb46-199e26cdec52)

# RESULT:

Thus,Feature Transformation is performed on the given dataset.



