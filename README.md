# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression models learn complex relationships between input variables and continuous outputs through interconnected layers of neurons. By iteratively adjusting parameters via forward and backpropagation, they minimize prediction errors. Their effectiveness hinges on architecture design, regularization, and hyperparameter tuning to prevent overfitting and optimize performance.

## Neural Network Model

![image](https://github.com/user-attachments/assets/47cbffc1-868e-430a-bba3-e8ab34c5136c)

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
# Importing the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Data from sheets
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Deep-1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

# Data Visualisation
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()
x=df[['INPUT']].values
y=df[['OUTPUT']].values

# Spliting and Preprocessing the data
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

# Building and compiling the model
ai_brain=Sequential([
    Dense(units=4,input_shape=[1]),
    Dense(units=2,activation='relu'),
    Dense(units=1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 3000)

# Loss Calculation
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

# Analysing the performance
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[57]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/user-attachments/assets/840bca93-3cfb-4d62-baa3-9834ba5ab986)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/e662e494-125f-4291-b3c5-d1f4183e0d45)

### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/e0a910df-80bc-4304-aed1-7dfd892392aa)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/0890f182-6bd9-4279-a19f-44ff78c1ddae)

## RESULT
Thus a basic neural network regression model for the given dataset is written and executed successfully.
