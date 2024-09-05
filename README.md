# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by human brain neurons). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps establish a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries that we will use Import the dataset and check the types of the columns Now build your training and test set from the dataset Here we are making the neural network 2 hidden layers with 1 output and input layer and an activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model
![image](https://github.com/user-attachments/assets/ab02d054-fa9c-4d93-8bd4-a838f944f0b8)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: MIRUDHULA D
### Register Number: 212221230060
```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('clinical').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'actual':'float'})
dataset1 = dataset1.astype({'predicted':'float'})
dataset1.head()
X = dataset1[['actual']].values
y = dataset1[['predicted']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(4,activation = 'relu',input_shape=[1]),
    Dense(2,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 4000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[10]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information

![image](https://github.com/user-attachments/assets/9162eaf1-2214-4cf4-9e15-8bfbec375e25)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/045fa2dd-ac2e-4256-b97f-959c3372265c)

### Test Data Root Mean Squared Error
![image](https://github.com/user-attachments/assets/e15f4c86-f457-47f7-9902-ae9e50beda31)

### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/9846c207-6462-49eb-a098-93390b2ec332)

## RESULT

Thus, Neural network for Regression model is successfully Implemented.
