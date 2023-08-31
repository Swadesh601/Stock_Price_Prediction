import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import streamlit as st
import yfinance as yf
yf.pdr_override()
from tensorflow.python import tf2
from tensorflow.python.keras.models import load_model
print(""" Dataset is taken from Yahoo finance
    If ticker of Indian Stock Market, add ".NS" at last
    For example -- "RELIANCE.NS" for RIL(Reliance Industries Limited)
                -- "TATAMOTORS.NS" for Tata Motors
                     for GOOGLE use "GOOG"
                     for TESLA use "TSLA"
                     for APPLE use "AAPL"
""")
st.title("Stock Price Prediction")
company =st.text_input("Enter Stock ticker symbol ")
df_train=pdr.DataReader(company, data_source='Yahoo', period='5y', interval='1d')
print(company)
#Describe The Data
st.subheader("Data from 2018-2023".format(company))

st.write(df_train)
#st.write(df_train.describe())

# Visualisation plotting the graph in pandas data frame
st.subheader("Open stock Price Growth".format(company))
opn = df_train.Open     # opn is a variable which will store the open values of a company and plot the graph in pandas data frame
fig=plt.figure(figsize=(12,6))
plt.plot(opn)
plt.legend(['Open'])
st.pyplot(fig)


#feature scaling
from sklearn.preprocessing import MinMaxScaler

len(df_train)

#Type of data
type(df_train)

df_train.isna().any()
df_train.info()

# Convert into numpy array 
ds = opn.values # converted into numpy array using .values gives the data in array
print(ds)
st.subheader("Open stock Price Growth in numpy Array".format(company))
fig2=plt.figure(figsize=(12,6))
plt.plot(ds)
plt.legend(['Open'])
st.pyplot(fig2)

#feature scaling
from sklearn.preprocessing import MinMaxScaler # preprocessing of data by rescaling of data(data preprocessing) using scikit learn.
#Using MinMaxScaler for Transforming or Normalising data between 0 & 1. MinMaxScalar scales all the data to be in the region of 0 and 1.
normalizer = MinMaxScaler(feature_range=(0,1))
#reshaping of ds variable to be in 2D array, for fitting model requires 2D array(reshape)
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

print(ds_scaled)
print("Length of DS_Scaled ",len(ds_scaled))
print("Length of DS ", len(ds))

#Defining test and train data sizes train=70% and test=30&
train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

print("Train_Size Value And Test_size Value ",train_size,test_size)

#Splitting data between train and test
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

print("Size of ds_train and Size of ds_test ",len(ds_train),len(ds_test))

print(ds_train)
#creating dataset in time series for LSTM model 
# taking 100 data set means by taking current 100 days stock price dataset as one record and next 101 day price which is your Y we predict the price at 101 day. 
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1): # prediction is been doing
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

# please combine line 81 with line 91 . in line 92 and 93 function is called where (ds_train and ds_test)=dataset variable and steps variable = timestamp=100
#Taking 100 days price as one record for training for X it will predict the next 101 day price as value for y 
time_stamp = 100
x_train, y_train = create_ds(ds_train,time_stamp)
x_test, y_test = create_ds(ds_test,time_stamp)    

print("x_train.shape ",x_train.shape)
print("y_train.shape ",y_train.shape)
print("x_test.shape ",x_test.shape)
print("y_test.shape ",y_test.shape)

#Reshaping data to fit into LSTM model
#why we need to reshape is because we need to give (x_train.shape[1],1) value it as input for giving it to LSTM
#(x_train.shape[0]) it is the no of records for x_train=763
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

# LSTM Model Implementaion
# Import Keras Libraries
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dropout  # Dropout for adding dropout layers that prevent overfitting

#Creating LSTM model using keras
# if don't write true for return_sequences=True, return_sequences=True which determines whether to return the last output in the output sequence.
model = Sequential()  # sequential jobs is to analyse sequence of data and dense,lstm and dropout to measure dropout rate 
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1))) # lstm=50 units which is the dimensionality of the output space.
model.add(LSTM(units=50,return_sequences=True))
model.add(LSTM(units=50)) 
model.add(Dense(units=1,activation='linear')) # Dense for adding a densely connected neural network layer
model.summary() # after execution total  50851 parameter required to be trained

#Training model with adam optimizer and mean squared error loss function
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=64)

# Visualisation
st.subheader("Loss",company)
loss = model.history.history['loss']
fig3=plt.figure(figsize=(12,6))
plt.plot(loss)
plt.legend(['LOSS'])
st.pyplot(fig3)


#Predicitng on train and test data
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

#Inverse transform is to get actual value of that transform value.
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(y_test,test_predict))

#Comparing using visuals
st.subheader("predicted Test and Train Value Vs Actual Value".format(company))
fig4=plt.figure(figsize=(12,6))
plt.plot((normalizer.inverse_transform(ds_scaled)),color="Blue")
plt.plot(train_predict,color="Red")
plt.plot(test_predict,color="Green")
plt.legend(['Actual Value', 'Predicted Train Value', 'Predicted Test Value'])
st.pyplot(fig4)

type(train_predict)

test = np.vstack((train_predict,test_predict))

# Visualisation
#Combining the predited data to create uniform data visualization
st.subheader(" Combination of predicted (Test and Train) Value Vs Actual Value".format(company))
fig5=plt.figure(figsize=(12,6))
plt.plot(normalizer.inverse_transform(ds_scaled),color="Blue")
plt.plot(test,color="Green")
# Predicted value = Addition of train and test predicted value
plt.legend(['Actual Value','Predicted Value'])
st.pyplot(fig5)

len(ds_test)
#Getting the last 100 days records
fut_inp = ds_test[270:] # after 270 data of test data remaining 100 data is actual will be used
fut_inp = fut_inp.reshape(1,-1) # First we are taking 100 values and then we are predicting the next value
tmp_inp = list(fut_inp)  # we are saving the predicited value to our current list
fut_inp.shape
# we will take the new predicted value in and again train it for 100 days data this is how it will predict for 30 days . for 30 times iteration loop.
#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

print(tmp_inp)

#Predicting next 30 days price using the current data
#It will predict in sliding window manner technique with stride 1
# stride by 1 means we are shifting value by 1 and we will have 101 value so we will remove the last value
# we will take the new predicted value in 
lst_output=[]
n_steps=(fut_inp.shape[1])
i=0
while(i<30):
    #when while loop runs it will enter in the else loop because in if condition tmp_inp=100, in else it will predict the first value at 101 th day 
    if(len(tmp_inp)>100):   # we will append the first predicted value with 100 existing data value once tmp_inp=101 , then we won't use first value
         #print(tmp_input)
        fut_inp = np.array(tmp_inp[1:]) # we will newly predicted value with other 99 values. this will go 30 times. 
        print("{} day input {}".format(i,fut_inp))
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        #print(fut_inp)
        yhat = model.predict(fut_inp, verbose=0)  # model is predicted here 
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        #print(tmp_inp)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)  # model is predicted here
        print(yhat[0])
        tmp_inp.extend(yhat[0].tolist())
        print(len(tmp_inp))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print("Output is ",lst_output)   # finally we will predict 30 new values. final 30 days prediction value

len(ds_scaled)

normalizer.inverse_transform(lst_output)
print("fut_inp.shape[0] is ",fut_inp.shape[0])
print("fut_inp.shape[1] is ",fut_inp.shape[1])

x=fut_inp.shape[1]
y=fut_inp.shape[0]
#Creating a dummy plane to plot graph one after another
plot_new=np.arange(1,x+1) # x=100 (1,101) is first 100 train data
plot_pred=np.arange(x+1,(x+30)+1) # (101,131) is predicted 30 days data

# Plot the Graph
st.subheader("dummy plane graph",company)
fig6=plt.figure(figsize=(12,6))
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[1136:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
st.pyplot(fig6)

ds_new = ds_scaled.tolist()
len(ds_new)

#Entends helps us to fill the missing value by taking approx value to show a smooth curve graph in visualisation
st.subheader("dummy graph",company)
fig7=plt.figure(figsize=(12,6))
ds_new.extend(lst_output)
plt.plot(ds_new[1200:])
st.pyplot(fig7)

#Creating final data for plotting
final_graph = normalizer.inverse_transform(ds_new).tolist()

#Plotting final results with predicted value after 30 Days
st.subheader("final results with predicted value after 30 Days ",company)
fig8=plt.figure(figsize=(12,6))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next month Open Price".format(company))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig8)