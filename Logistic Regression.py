########### Logistic regression ########################


import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import os
import statsmodels.api as sm
import pylab as pl

os.chdir("D:\\DATA SCIENCE COURSE\\CSV FILES")

# import dataset
data = pd.read_csv('LogReg1.csv')
pd.set_option('display.max_columns', None)#show all columns
data.head()

## Information Value ##
def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

## calling the function
# data is the name of the dataset
# Churn is the dependent varaible
iv, woe = iv_woe(data = data, target = 'Churn', bins=10, show_woe = True)
print(iv)
print(woe)

iv = pd.DataFrame(iv)
iv.sort_values(["IV"],ascending=[0])





## IV ends here ########



def outlier (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["utilization"])#no outlier

data.boxplot(column=["Age"])
data = outlier(data,"Age")

data.boxplot(column=["MonthlyIncome"])
data = outlier(data,"MonthlyIncome")

data.describe()#monthly income of 1 has been removed

data.boxplot(column=["DebtRatio"])
data = outlier(data,"DebtRatio")
data






#missing value treatment
data.isnull().sum()#show is any variable has any missing value

data = data.dropna()#deletes rows with missing values



data.head()











train_cols = data.loc[:,["utilization","Age","Num_loans",
"Num_dependents","MonthlyIncome","Num_Savings_Acccts","DebtRatio"]]

train_cols.head()







#train_cols.head()
# model
logit = sm.Logit(data['Churn'], train_cols)
# fit the model
result = logit.fit()
result.summary()#all p values are significant

# may not run this code
var = pd.DataFrame(round(result.pvalues,3))# shows p value
var["coeff"] = result.params#coefficients
#rename columns
var.columns.values[[0,1]]= ["p value","coefficients"]
var

cov = result.cov_params()
std_err = np.sqrt(np.diag(cov))
var["z"]=result.params.values/std_err
var
# end of may not run this code

result.conf_int()# confidence interval

np.exp(result.params)#odds ratio (may igmore it for the time being)

### VIF Calculation
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


result=sm.ols(formula="Churn~utilization+Age+Num_loans+Num_dependents+MonthlyIncome+Num_Savings_Acccts+DebtRatio",
             data=data).fit()
result.summary()# shows total summary

#remove variable based on vif
#all vif values are under 2, hence no variable is removed

var = pd.DataFrame(round(result.pvalues,3))# shows p value
var["coeff"] = result.params#coefficients
variables = result.model.exog #.if I had saved data as rock
# this it would have looked like rock.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 
var["vif"] = vif
var

## final variables after vif

#change for vif
train_cols = data.loc[:,["utilization","Age","Num_loans",
"Num_dependents","MonthlyIncome","Num_Savings_Acccts","DebtRatio"]]

# upto this much correct
## AUC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


train_cols = data.loc[:,["utilization","Age","Num_loans",
"Num_dependents","MonthlyIncome","Num_Savings_Acccts","DebtRatio"]]



inputData=train_cols # ind var
outputData=data.loc[:,"Churn"] # dep var





#outputData.count()

logit1=LogisticRegression()
logit1.fit(inputData,outputData)
logit1.score(inputData,outputData)

y_pred = logit1.predict(train_cols)
prob = logit1.predict_proba(train_cols)
#prob.count()
#transform the probabilities into DataFrame
# it shows probability for both
prob = pd.DataFrame(prob)
prob = prob.iloc[:,1]#showing the probability of being 1
prob = prob.reset_index()
prob.head()



outputData = pd.DataFrame(outputData)
outputData.head()
outputData = outputData.reset_index()
outputData = outputData.iloc[:,1]
outputData.head()


rock = pd.concat([outputData,prob], axis=1)
rock = rock.iloc[:,[0,2]]#this line might give error, check the column index
rock.head()
df = rock.copy()
df.columns = ["y","p"]
df.head()



###Confusion matrix with sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(inputData),outputData)# this is experimental not required

##Computing false and true positive rates
fpr, tpr,_=roc_curve(logit1.predict(inputData),outputData,drop_intermediate=False)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
roc_auc_score(logit1.predict(inputData),outputData)


## ks stat
df = rock.copy()
df.columns = ["y","p"]
df.head()

new = df.copy()
new.columns = ["Churn","Prob"]
new.head()

new['decile'] = pd.qcut(new['Prob'],10,
labels=['1','2','3','4','5','6','7','8','9','10'])
new.head()

new.columns = ['Defaulter','Probability','Decile']
new.head()
new['Non-Defaulter'] = 1-new['Defaulter']
new.head()


boogieman = pd.pivot_table(data=new,index=['Decile'],
values=['Defaulter','Non-Defaulter','Probability'],
aggfunc={'Defaulter':[np.sum],'Non-Defaulter':[np.sum],
'Probability' : [np.min,np.max]})
boogieman.head()
boogieman.reset_index()

boogieman.columns = ['Defaulter_Count','Non-Defaulter_Count','max_score','min_score']
boogieman['Total_Cust'] = boogieman['Defaulter_Count']+boogieman['Non-Defaulter_Count']
boogieman

kane = boogieman.sort_values(by='min_score',ascending=False)
kane


kane['Default_Rate'] = (kane['Defaulter_Count'] / 
kane['Total_Cust']).apply('{0:.2%}'.format)
default_sum = kane['Defaulter_Count'].sum()
non_default_sum = kane['Non-Defaulter_Count'].sum()
kane['Default %'] = (kane['Defaulter_Count']/
default_sum).apply('{0:.2%}'.format)
kane['Non_Default %'] = (kane['Non-Defaulter_Count']/
non_default_sum).apply('{0:.2%}'.format)
kane

kane['ks_stats'] = np.round(((kane['Defaulter_Count'] / 
kane['Defaulter_Count'].sum()).cumsum() -
(kane['Non-Defaulter_Count'] / 
kane['Non-Defaulter_Count'].sum()).cumsum()), 4) * 100
kane

flag = lambda x: '*****' if x == kane['ks_stats'].max() else ''
kane['max_ks'] = kane['ks_stats'].apply(flag)
kane


## ks stat 2nd method

df = rock.copy()
df.columns = ["y","p"]
df.head()



def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)


mydf = ks(data=df,target="y", prob="p")


## merging data and probability score

df = rock.copy()
df.columns = ["y","p"]
df.reset_index(inplace=True)
df.head()

kane = data.copy()
kane.reset_index(inplace=True)
kane.head()

final = pd.concat([df,kane],axis=1)
final.head()

# export data onto local drive
import os

os.chdir('D:\\DATA SCIENCE COURSE\\EXCECUTED CSV FILES')

final.to_csv('Logistic regression.csv',index=False)#export data to local drive














