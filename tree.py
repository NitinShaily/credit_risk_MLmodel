#%%
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#%%
os.chdir('c:\\Users\\HP\\Desktop\\ML')
#%%
df=pd.read_csv("credit_risk.csv")
#%%
df.isnull().sum()
#%%
# filtering out all null values in different coloums

def  fill_Depen(df):
    df.Dependents.fillna(0,inplace=True)
    return df

def fill_grn(df):   
    df.Gender.fillna('Male',inplace=True)
    return df

def fill_married(df):
    df.Married.fillna('Yes',inplace=True)
    return df

def fill_SelfEmp(df):
    df.Self_Employed.fillna('No',inplace=True)
    return df

def fill_LoanAmount(df):
    mean=146.412
    df.LoanAmount.fillna(mean,inplace=True)
    return df
def fill_Loan_Amount_Term(df):
    df.Loan_Amount_Term.fillna(360,inplace=True)
    return df

def fill_CH(df):
    df.Credit_History.fillna(1.0,inplace=True)
    return df

#%%
def label_encode(df):    
    from sklearn.preprocessing import LabelEncoder
    label=LabelEncoder()
    df['Married']=label.fit_transform(df['Married'])
    #df['Dependents']=label.fit_transform(df['Dependents'])
    df["Gender"]=label.fit_transform(df["Gender"])
    df['Self_Employed']=label.fit_transform(df['Self_Employed'])
    return df
#%%
def label_feature(df):
    
    #df=fill_Depen(df)
    df=fill_CH(df)
    df=fill_Loan_Amount_Term(df)
    df=fill_LoanAmount(df)
    df=fill_married(df)
    df=fill_grn(df)
    df=fill_SelfEmp(df)
    df=label_encode(df)
    return df
#%%
train,test=train_test_split(df,test_size=0.2,random_state=12)
#%%
train=label_feature(df)
test=label_feature(df)
#%%
def x_and_y(df):
    x=df.drop(['Education','Loan_ID','Loan_Status','Married','Dependents','Property_Area'],axis=1)
    y=df.Loan_Status
    return x,y
x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)

#%%
y_train.head()

#%%
x_train.head()

#%%
log_model=DecisionTreeClassifier(criterion="entropy")
log_model.fit(x_train,y_train)
prediction=log_model.predict(x_test)
score=accuracy_score(y_test,prediction)
print(score)

#%%
