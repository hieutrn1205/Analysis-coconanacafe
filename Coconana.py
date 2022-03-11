
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


# In[6]:


sales["Net Total"].hist(bins = 15)
plt.title("Net total overall of Coconana")
plt.xlabel("Net Total ($)")
plt.ylabel("Frequency")


# In[7]:


sales["Tips"] = sales["Tips"].str.replace("$", "")
sales["Tips"] = sales["Tips"].astype(float)
lm = smf.ols("Tips ~ Q('Order Count') + Q('Net Total')", data=sales).fit()
lm.summary()


# In[8]:


plt.scatter(x = sales["Tips"], y = lm.resid)
plt.title("Relationship between Net Total, Order Count and Tips")
plt.xlabel("Tips")
plt.ylabel("Residuals")

