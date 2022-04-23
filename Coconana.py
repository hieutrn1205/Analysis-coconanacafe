
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read the csv file
sales = pd.read_csv("sales-by-day-11-20-03-10.csv")
#sales.drop is removes rows/column in dataset
#index[-1] is the last row
sales = sales.drop(sales.index[-1])
sales


# In[3]:


#Create variable to store the rows that are duplicated
duplicated_rows = sales.duplicated()
#Filter out the duplicated rows
sales[duplicated_rows]


# In[4]:


#Remove duplicated rows if there is any
#In this case the duplicated rows are 15, 21
sales = sales.drop_duplicates()
#Display the dataset after cleaning
sales


# In[5]:


#sales["Date"] = pd.to_datetime(sales["Date"], infer_datetime_format = True, errors = 'coerce')
#sales.head()


# In[6]:


#Remove special characters to convert the value in the Net Total column 
#into the numeric value in order to graph
sales["Net Total"]= sales["Net Total"].str.replace("$", "")
sales["Net Total"]= sales["Net Total"].str.replace(",", "")
#Convert types of the column Net Total to float
sales["Net Total"] = sales["Net Total"].astype(float)
plt.scatter(x="Order Count", y ="Net Total", data = sales)
plt.title("Coconana's Order counts vs Net total")
plt.xlabel("# of Order")
plt.ylabel("Net Total($)")


# In[7]:


sales["Net Total"].hist(bins = 15)
plt.title("Net total overall of Coconana")
plt.xlabel("Net Total ($)")
plt.ylabel("Frequency")


# In[8]:


sales["Tips"] = sales["Tips"].str.replace("$", "")
sales["Tips"] = sales["Tips"].astype(float)
lm = smf.ols("Tips ~ Q('Order Count') + Q('Net Total')", data=sales).fit()
lm.summary()


# In[9]:


plt.scatter(x = sales["Tips"], y = lm.resid)
plt.title("Relationship between Net Total, Order Count and Tips")
plt.xlabel("Tips")
plt.ylabel("Residuals")


# In[10]:


sales.head()


# In[11]:


sales.dtypes


# In[12]:


sales["Gross Sales"] = sales["Gross Sales"].str.replace("$", "")
sales["Gross Sales"] = sales["Gross Sales"].str.replace(",", "")
sales["Gross Sales"] = sales["Gross Sales"].astype(float)
sales.head()


# In[13]:


x = sales[["Order Count", "Gross Sales", "Net Total"]]
y = sales["Tips"]


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[16]:


#lab_enc = preprocessing.LabelEncoder()
#y_train = lab_enc.fit_transform(y_train)


# In[17]:


knn5 = KNeighborsRegressor(n_neighbors = 5)
knn5.fit(x_train, y_train)


# In[18]:


y_test_pred = knn5.predict(x_test)


# In[19]:


mean_squared_error(y_test_pred, y_test)


# In[20]:


y_train_pred = knn5.predict(x_train)
mean_squared_error(y_train_pred, y_train)


# In[21]:


knn3 = KNeighborsRegressor(n_neighbors = 3)
knn3.fit(x_train, y_train)


# In[22]:


y_test_pred = knn3.predict(x_test)


# In[23]:


mean_squared_error(y_test_pred,  y_test)


# In[24]:


y_train_pred = knn3.predict(x_train)
mean_squared_error(y_train_pred, y_train)


# In[25]:


knn10 = KNeighborsRegressor(n_neighbors = 10)
knn10.fit(x_train, y_train)


# In[26]:


y_test_pred = knn10.predict(x_test)


# In[27]:


mean_squared_error(y_test_pred, y_test)


# In[28]:


y_train_pred = knn10.predict(x_train)
mean_squared_error(y_train_pred, y_train)


# In[29]:


train_mses=[]
for k in range(1, 10):
    print("Now computing the MSE for k = ", k)
    iknn = KNeighborsRegressor(n_neighbors = k)
    iknn.fit(x_train, y_train)
    iknn_y_pred = iknn.predict(x_train)
    mse = mean_squared_error(iknn_y_pred, y_train)
    train_mses.append(mse)


# In[30]:


mses=[]
for k in range(1, 10):
    print("Now computing the MSE for k = ", k)
    iknn = KNeighborsRegressor(n_neighbors = k)
    iknn.fit(x_train, y_train)
    iknn_y_pred = iknn.predict(x_test)
    mse = mean_squared_error(iknn_y_pred, y_test)
    mses.append(mse)


# In[31]:


mses


# In[32]:


plt.plot(mses)
plt.plot(train_mses)
plt.legend(["Test MSE", "Train MSE"])
plt.title("The MSE for the number of k")
plt.xlabel("Numbers of k")
plt.ylabel("MSE of train and test data")


# In[33]:


###The model is overfitting when k is at 0 to 3, after that the model becomes underfitting


# In[37]:


mses_tree =[]
train_mses_tree = []
for k in range(1,10):
    ireg = DecisionTreeRegressor(max_depth = k)
    ireg.fit(x_train, y_train)
    ireg_y_pred = ireg.predict(x_test)
    ireg_y_train = ireg.predict(x_train)
    mse_train = mean_squared_error(ireg_y_train, y_train)
    mse_test = mean_squared_error(ireg_y_pred, y_test)
    mses_tree.append(mse_test)
    train_mses_tree.append(mse_train)
mses_tree


# In[38]:


plt.plot(train_mses_tree)
plt.plot(mses_tree)
plt.legend(["Test MSE", "Train MSE"])
plt.title("The MSE for the number of k")
plt.xlabel("Numbers of k")
plt.ylabel("MSE of train and test data")


# In[ ]:


###The model is overfitting at k is 0 and 1, after that model becomes underfitting

