#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\batsh\OneDrive\Documents\College\Spring Semester 2025\Decision Support Systems\loan_approval_dataset.csv")
data.head(10)


# In[48]:


data.shape


# In[49]:


data.isnull().sum()


# In[50]:


data.duplicated().sum()


# In[51]:


data.info()


# In[52]:


data["loan_id"] = data["loan_id"].astype(str)


# In[53]:


data.info()


# In[54]:


data.describe()


# In[55]:


avg_loan_amount = data.groupby(" loan_status")[" loan_amount"].mean()
avg_loan_amount


# In[56]:


avg_loan_amount = data.groupby(" loan_status")["credit_score"].mean()
avg_loan_amount


# In[57]:


avg_loan_amount = data.groupby(" loan_status")[" no_of_dependents"].mean()
avg_loan_amount


# In[58]:


corr_income_amount = data["annual_income"].corr(data[" loan_amount"])
corr_income_amount


# In[59]:


print(data.columns)


# In[60]:


avg_loan_amount = data.groupby(" loan_status")[" loan_term_years"].mean()
avg_loan_amount


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[16]:


data[' loan_status'] = data[' loan_status'].replace({'Approved': 1, 'Rejected': 0})


# In[17]:


X = data[['credit_score', 'annual_income', ' loan_amount', ' loan_term_years']]
y = data[' loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  
model = LogisticRegression()
model.fit(X_train, y_train)


# In[18]:


y_pred = model.predict(X_test)


# In[19]:


print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# In[20]:


print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


# In[21]:


print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


# In[23]:


feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': abs(model.coef_[0])})
feature_importance.sort_values(by='Coefficient', ascending=False)


# In[ ]:





# In[ ]:




