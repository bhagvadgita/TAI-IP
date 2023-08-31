# %%
# importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#importing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/SaiTamminana89/csv/main/TCS1.CSV')
df

# %%
#This dataframe contains 7 columns
#Date-- Date includes day and month and year
#open -- open value of the TCS stock price on particular day
#High -- high price value of TCS on particular day
#LOW -- Low Price Value Of ITC on particular day
#Close --Stock Price of TCS After Closing The Stock Market
#Volume -- Volume of TCS means sum of buy's and shares
#Adjclose --Adjusted close is the closing price after adjustments for all applicable splits and dividend distributions

# %% [markdown]
# Data Preprocessing

# %%
df.isnull().sum()
# checking whether it contains any null values

# %%
# replacing null values with the mean of the open column
df['Open'] = df['Open'].fillna(df['Open'].mean())

# %%
# replacing null values with the mean of the High column
df['High'] = df['High'].fillna(df['High'].mean())

# %%
# replacing null values with the mean of the Low column
df['Low'] = df['Low'].fillna(df['Low'].mean())

# %%
# replacing null values with the mean of the Close column
df['Close'] = df['Close'].fillna(df['Close'].mean())

# %%
# replacing null values with the mean of the AdjClose column
df['Adj Close'] = df['Adj Close'].fillna(df['Adj Close'].mean())

# %%
# replacing null values with the mean of the Volume column
df['Volume'] = df['Volume'].fillna(df['Volume'].mean())

# %%
df.isnull().sum()
# there is no missing values present in this dataset

# %%
# cheecking whtether it contains duplicates
df.duplicated().sum()

# %%
# our dataset looks cool

# %% [markdown]
# Data Visualization

# %%
corr_matrix = df.corr()
corr_matrix
# this gives the correlation between the variables

# %%
import seaborn as sns
# importing seaborn it is a visualization library
sns.heatmap(corr_matrix,cmap='BrBG')

# %%
import matplotlib.pyplot as plt

# %%
# Checking the relation between the input and target variables by data visulaization 
# our target variable is close price 

# %%
# Scatter Plot Between Open and Close
plt.scatter(df['Open'],df['Close'])
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('open and Close')
# open and close are highly correlated

# %%
# Scatter Plot Between High and Low
plt.scatter(df['High'],df['Close'])
plt.xlabel('High')
plt.ylabel('Close')
plt.title('High and Close')
# High and close are highly correlated

# %%
# Scatter Plot Between Volume and Low
plt.scatter(df['Volume'],df['Close'])
plt.xlabel('Volume')
plt.ylabel('Close')
plt.title('Volume and Close')
# Volume and close are not much correlated

# %% [markdown]
# Model Making

# %%
df.head(2)
# it returns up to 2 entries

# %%
x = df.iloc[:,[1,2,3,6]]
y = df.iloc[:,[4]]

# %%
x
# our input variables

# %%
y
# Our target Variable

# %%
# Model making using RandomForestRegressor

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)

# %%
x_train.shape
# to get the shape of the training data

# %%
y_train.shape

# %%
x_test.shape

# %%
y_test.shape

# %%
from sklearn.ensemble import RandomForestRegressor
# importing randomforest regressor from ensemble

# %%
model = RandomForestRegressor()
mymodel = model.fit(x_train,y_train)
# training the model

# %%
pred_y = mymodel.predict(x_test)
# testing using testing data

# %%
from sklearn.metrics import r2_score
r2_score(pred_y,y_test)
# r2_score is used to check the performance of the model

# %%
# Accuracy of our model is 0.9998685435063304


