#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression


# In[2]:


st.markdown('## Telefon Fiyat Tahmini')
st.write('---------------------------------')


Arka_Kamera_Sayisi = st.number_input('Arka_Kamera_Sayisi (0 ve 10 arasında değer)', 0, 10)
Ön_Kamera_Sayisi = st.number_input('Ön_Kamera_Sayisi (0 ve 10 arasında değer)', 0, 10)
Dahili_Hafiza = st.number_input('Dahili_Hafiza (1000 ve 512000 arasında değer; - 128000=128GB olarak düşünün)', 1000, 512000)
Dokunmatik_Ekran = st.number_input('Dokunmatik_Ekran (1=Evet; 0=Hayır)', 0, 1)
Garanti_Süresi = st.number_input('Garanti_Süresi (0 ve 4 arasında değer)', 0, 4)
NFC = st.number_input('NFC (1=Evet; 0=Hayır)', 0, 1)
RAM_Kapasitesi = st.number_input('RAM_Kapasitesi (2 ve 16 arasında değer)', 2, 16)
Suya_Toza_Dayaniklilik = st.number_input('Suya_Toza_Dayaniklilik (1=Evet; 0=Hayır)', 0, 1)
Çift_Hat = st.number_input('Çift_Hat (1=Evet; 0=Hayır)', 0, 1)
İşletim_Sistemi = st.number_input('İşletim_Sistemi (1=iOS; 0=Android)', 0, 1)
Renk = st.number_input('Renk (1 ve 30 arasında bir değer girin)', 1, 30)
Cep_Telefonu_Modeli = st.number_input('Cep_Telefonu_Modeli (1 ve 100 arasında bir değer girin)', 1, 100)



if st.button('Telefon Fiyatını Tahmin Et'):
    features = np.array([[Arka_Kamera_Sayisi, Ön_Kamera_Sayisi, Dahili_Hafiza, Dokunmatik_Ekran,
                          Garanti_Süresi, NFC, RAM_Kapasitesi, Suya_Toza_Dayaniklilik, Çift_Hat,
                          İşletim_Sistemi,Renk,Cep_Telefonu_Modeli]])
    model = joblib.load('rf_model.sav')  # 'modelinizi_kaydettiğiniz_dosya.pkl' dosya adınızı gerçek dosya adınızla değiştirin
    Fiyat = model.predict(features)
    st.text(Fiyat[0])


# In[ ]:




