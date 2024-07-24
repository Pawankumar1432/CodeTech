
# Data preprocessing 

# Importing libraries
import numpy as np
import pandas as pd

# Importing datasets
data=pd.read_csv('Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

# Finding Missing Data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

# Encoding Categorical Data
# Encoding independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=ct.fit_transform(x)
print(x)
# Encoding dependent variables
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit_transform(y)

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

# Feature scaling
#Standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain[:,3:]=sc.fit_transform(xtrain[:,3:])
xtest[:,3:]=sc.fit_transform(xtest[:,3:])

# Normalization
from sklearn.preprocessing import MinMaxScaler
n=MinMaxScaler()
xtrain[:,3:]=n.fit_transform(xtrain[:,3:])
xtest[:,3:]=n.fit_transform(xtest[:,3:])