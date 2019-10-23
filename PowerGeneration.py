# -*- coding: utf-8 -*-                                                   
"""

@author: Kavitha Shiva
"""
import pandas as pd
import seaborn as sns#predefined func
#vacumn,Air Temp ,Air Pressure, Roon humidity ,Pwr genrtn in Energy

data=pd.read_csv("PowerGen.csv") 
print(data.columns)#column names
print(data.dtypes)#column types
print(data.head)#to take first 5 rows
print(data.tail)#to take last 5 rows
desc=data.describe()#finding mean,std.... of all the indiv columns
print(desc)

sns.distplot(data["AP"])
sns.boxplot(data["AP"])

                    
sns.distplot(data["RH"])

sns.boxplot(data["RH"])

sns.distplot(data["AT"])
sns.boxplot(data["AT"])

sns.distplot(data["PE"])
sns.boxplot(data["PE"])

sns.distplot(data["V"])
sns.boxplot(data["V"])

correaktion = data.corr()#to find of each correlation

#sns.pointplot(data["RH"],data["PE"])
sns.regplot(data["RH"],data["PE"])#to find relation
#o rows 1 column
data.drop(["RH"],inplace=True,axis=1)#to drop the columnl as not rela
sns.regplot(data["AP"],data["PE"])
sns.regplot(data["AT"],data["PE"])
sns.regplot(data["V"],data["PE"])
#find triangular effect
data.drop(["V"],inplace=True,axis=1)#based on triangulaare effect as v and AT related ,and v is not correlated much as AT and has outliers as well see in correlation table
#while predicting

#to get the no of rows nd column
data.shape


y =  data["PE"]
x = data[["AP","AT"]]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

print(model.coef_)
print(model.intercept_)

pred = model.predict(x)

from sklearn.metrics  import r2_score , mean_squared_error
print(r2_score(y,pred))
print(mean_squared_error(y,pred))   
                      

pred_rand = model.predict([])                     


