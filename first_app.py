# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:41:44 2021

@author: Elijah_Nkuah
"""
'''altair==4.1.0
catboost==0.24.3
lightgbm==3.1.1
matplotlib==3.0.3
mlxtend==0.18.0
numpy==1.19.4
pandas==0.24.2
pipreqs==0.4.10
scikit-image==0.14.2
scikit-learn==0.24.1
seaborn==0.9.0
statsmodels==0.9.0
streamlit==0.74.1
xgboost==1.3.0.post0
yfinance==0.1.55
'''
import streamlit as st
import numpy as np
import pandas as pd
import time

# Interactive Widget
st.write("Example of Streamlit interactive widget")
st.button("Click button bellow")
if st.button("Say Hello"):
    st.write('You clicked button 1')
else:
    st.write('Goodbye')
