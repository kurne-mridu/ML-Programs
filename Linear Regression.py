import pandas as pd
import numpy as np
import statsmodels.formula.api as sm


# import dataset
import os

os.chdir("D:\\DATA SCIENCE COURSE\\CSV FILES")

data = pd.read_csv('House_Price.csv')
pd.set_option('display.max_columns', None)#show all columns
data.head()
# boxplot to showoutliers
data.boxplot(column=["Price_house"])


# outliers with quantiles
data.Price_house.quantile([0,0.01,.1,.3,.5,.7,.9,.95,.99,.995,1])#quantiles



#  function to remove outliers
def outliers(data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["Price_house"])
data = outliers(data,"Price_house")
data


data.boxplot(column=["Taxi_dist"])
data = outliers(data,"Taxi_dist")


data.boxplot(column=["Carpet_area"])
data = outliers(data,"Carpet_area")

data.boxplot(column=["Builtup_area"])

data.boxplot(column=["Rainfall"])
data = outliers(data,"Rainfall")


#
data.info()#shows data rows
data.isnull().sum()#missing values per variable
data = data.dropna()# deletes all rows with missing values


#segregating categorical variables
cat = data.loc[:,["City_type","Parking_type"]]
cat.head()

#dropping the original variables
data = data.drop(["City_type","Parking_type"],axis=1)



# creating dummy varaibles
dum = pd.get_dummies(cat.astype(str),drop_first=True)
dum.head()

# concatnating the columns (cbind of R)
data = pd.concat([data,dum],axis=1)



rock=sm.ols(formula=
"Price_house~Taxi_dist+Market_dist+Hospital_dist+Carpet_area+Builtup_area + Rainfall+Q('City_type_CAT B')+Q('City_type_CAT C')+Q('Parking_type_No Parking')+Q('Parking_type_Not Provided')+Q('Parking_type_Open') ",
data=data).fit()
rock.summary()# shows total summary




rock=sm.ols(formula=
"Price_house~Market_dist+Hospital_dist+Carpet_area+Builtup_area + Rainfall+Q('City_type_CAT B')+Q('City_type_CAT C')+Q('Parking_type_No Parking')+Q('Parking_type_Not Provided')+Q('Parking_type_Open') ",
data=data).fit()
rock.summary()# shows total summary

rock=sm.ols(formula=
"Price_house~Hospital_dist+Carpet_area + Q('City_type_CAT B')+Q('City_type_CAT C')+Q('Parking_type_No Parking')+Q('Parking_type_Not Provided')+Q('Parking_type_Open') ",
data=data).fit()
rock.summary()# shows total summary

data["pred"] = rock.predict()
data.head()

