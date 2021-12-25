import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 

import os

os.chdir("D:\\DATA SCIENCE COURSE\\CSV FILES")
airline = pd.read_csv('data_arima.csv')# put the path and filename
airline = pd.read_csv('data_arima.csv', index_col ='month', parse_dates = True) 
airline.head()

result = seasonal_decompose(airline['air'],  model ='multiplicative') 
result.plot()

from pmdarima import auto_arima 
import warnings 
warnings.filterwarnings("ignore") 
#description of auto arima
#https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html

stepwise_fit = auto_arima(airline['air'], start_p = 1, start_q = 1, 
max_p = 8, max_q = 8, m = 12, start_P = 0, seasonal = True, 
 d = 1, D = 1, trace = True, error_action ='ignore',   
 suppress_warnings = True,   stepwise = True)         
  
# To print the summary 
stepwise_fit.summary() 



#total airline datarows
len(airline)

# Split data into train / test sets 
train = airline.iloc[:132] 
len(train)
test = airline.iloc[132:] 
len(test)

# Fit a SARIMAX(0, 1, 1)x(2, 1, [], 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['air'],  
                order = (0, 1, 1),  
                seasonal_order =(2, 1, [], 12)) 
  
result = model.fit() 
result.summary() 



# Predictions for one-year against the test set 

predictions = result.predict(132, 143, 
typ = 'levels').rename("Predictions") 
len(predictions)

#Plot
predictions.plot(legend = True) 
test['air'].plot(legend = True) 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(test['air'],predictions)#mape

# Train the model on the full dataset 
model = SARIMAX(airline['air'],  
                        order = (0, 1, 1),  
                        seasonal_order =(2, 1, [], 12)) 
result = model.fit() 
  
# Forecast for the next 3 years 
forecast = result.predict(start = len(airline),  
                          end = (len(airline)-1) + 3 * 12,  
                          typ = 'levels').rename('Forecast') 

  
# Plot the forecast values 
airline['air'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 
plt.savefig("result.png")

#export the forecasted values
forecast.to_csv("bhulbhal.csv")





















################ using log #####################






import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 

import os

os.chdir('C:\\Users\\USER\\Desktop\\Python\\Class 8 ARIMA')
airline = pd.read_csv('data.csv')# put the path and filename
airline = pd.read_csv('data.csv', index_col ='month', parse_dates = True) 
airline.head()

#log transform
airline["air"] = np.log10(airline["air"])


result = seasonal_decompose(airline['air'],  model ='multiplicative') 
result.plot()

from pmdarima import auto_arima 
import warnings 
warnings.filterwarnings("ignore") 
#description of auto arima
#https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html

stepwise_fit = auto_arima(airline['air'], start_p = 1, start_q = 1, 
max_p = 8, max_q = 8, m = 12, start_P = 0, seasonal = True, 
 d = 1, D = 1, trace = True, error_action ='ignore',   
 suppress_warnings = True,   stepwise = True)         
  
# To print the summary 
stepwise_fit.summary() 



#total airline datarows
len(airline)

# Split data into train / test sets 
train = airline.iloc[:132] 
len(train)
test = airline.iloc[132:] 
len(test)

# Fit a SARIMAX(0, 1, 1)x(0, 1, 1, 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['air'],  
                order = (0, 1, 1),  
                seasonal_order =(0, 1, 1, 12)) 
  
result = model.fit() 
result.summary() 



# Predictions for one-year against the test set 

predictions = result.predict(132, 143, 
typ = 'levels').rename("Predictions") 
len(predictions)

#Plot
predictions.plot(legend = True) 
test['air'].plot(legend = True) 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(test['air'],predictions)#mape

# Train the model on the full dataset 
model = SARIMAX(airline['air'],  
                        order = (0, 1, 1),  
                        seasonal_order =(0, 1, 1, 12)) 
result = model.fit() 
  
# Forecast for the next 3 years 
forecast = result.predict(start = len(airline),  
                          end = (len(airline)-1) + 3 * 12,  
                          typ = 'levels').rename('Forecast') 


#Anti log10 (10 to the power)
forecast = np.power(10,forecast)

#import the original data
airline = pd.read_csv('data.csv', index_col ='month', parse_dates = True) 

# Plot the forecast values 
airline['air'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 
plt.savefig("result.png")



import os

os.chdir('D:\\DATA SCIENCE COURSE\\EXCECUTED CSV FILES')

final.to_csv('Logistic regression.csv',index=False)#export data to local drive






#export the forecasted values
forecast.to_csv("arima.csv")










