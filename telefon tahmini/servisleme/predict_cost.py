#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib


# In[4]:


def predict(dataset):
    rf=joblib.load('rf_model.sav')
    return rf.predict(dataset)


# In[ ]:




