import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics

"""LOADING THE DATASET"""

data = pd.read_csv("C:\\Users\\sastr\\Downloads\\insurance.csv")


"""ANALYSIS OF TH SEX COLUMN OF THE DATASET"""

sns.countplot(x = 'sex', data = data)
plt.title('Sex Distribution')


'''ANALYSIS OF THE BMI COLUMN'''
sns.set()
sns.distplot(data['bmi'])
plt.show()


"""ANALYSIS OF THE CHILDREN COLUMN"""
sns.set()
sns.countplot(x = 'smoker',data = data)
plt.title("Smoker Distribution")
plt.show()


"""ANALYSIS OF THE CHARGES COLUMN"""
sns.set()
sns.distplot(data['charges'])
plt.title("Distribution of charges")
plt.show()


"""CONVERTING THE NON-NUMERICAL VALUES TO NUMERIC"""

data.replace({'sex':{'male':0,'female':1}},inplace = True)
data.replace({'smoker':{'yes':0,'no':1}},inplace = True)
data.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace = True)
data.head()


"""SPLITTNG THE DATA INTO TRAINING AND TESTING"""
x = data.drop(columns = 'charges',axis = 1)
y = data['charges']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=2)


"""MODEL TRAINING"""

model = LinearRegression()
model.fit(x_train,y_train)

"""PREDICTING INSURANCE COSTS"""

prediction = model.predict(x_test)


# input_data = ()----> Enter the values of a particular row except for the charges column
# arr = np.array(input_data)
# arr = arr.reshape(1,-1)
# predicts = model.predict(arr)
# predicts
