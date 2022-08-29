# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 22:58:39 2021

@author: Elijah_Nkuah
"""
# Import packages
# CORE PKG
import streamlit as st
hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
    }
footer {
        visibility: visible;
        }

footer:before{
    content: 'Copyright @ Wondlyfe IT SYSTEMS';
    display: block;
    position: relative;
    color:tomato;
    }
</style>


"""
import pickle

# EDA PKG
import numpy as np
import pandas as pd
# Import Data
#from sklearn.datasets import fetch_openml
#mice = fetch_openml(name='miceprotein', version=4)
#iris = sklearn.datasets.load_iris(return_X_y=False, as_frame=False)

# UTILS
import os
import joblib
import hashlib # you can also use passlib and bcrypt

# VISUALISATION PKG
#%matplotlib inline 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss
from scipy import stats
#matplotlib.use('Agg')

#sns.pairplot(df)
# DATABASE
from manage_db_1 import *

#@st.cache
# Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Verify password
def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False

feature_variables = ("age", "anaemia", 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time')# 'DEATH_EVENT')
gender_dict = {"male":1, "female":2}
feature_dict = {"No":1, "Yes":2}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
                
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key        
        
def get_fvalue(val):
    feature_dict = {"No":1, "Yes":2}
    for key, value in feature_dict.items():
        if val == key:
            return value         

# LOAD ML models
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

    
   
# Interface

def main():
    """Mortality Prediction App"""
    st.title("Disease Mortality Prediction App")
    st.markdown(hide_menu, unsafe_allow_html=True)
    menu = ['Home', 'Login', 'Signup']
    submenu = ['Plot','Prediction', 'Metrics']
    
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.write("What is Hepatitis?")
    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username, verify_hashes(password, hashed_pswd))
            #if password =="12345":
            if result:
                st.success("Welcome {}".format(username))
                
                activity = st.selectbox("Activity", submenu)
                if activity == "Plot":
                    st.subheader("Data Visualisation Plot")
                    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
                    st.dataframe(df.head())
                    st.dataframe(df.describe())
                    #df['anaemia'].value_counts().plot(kind='bar', color='orange')
                    #st.pyplot(anaemia)
                    #fig, ax = plt.subplots(figsize=(7, 3))
                    anaemia_count = df['anaemia'].value_counts()
                    fig, ax = plt.subplots(figsize=(5, 2))
                    #styledict, or one of {darkgrid, whitegrid, dark, white, ticks}
                    ax = sns.set(style="darkgrid")
                    ax = sns.barplot(anaemia_count.index, anaemia_count.values, alpha=0.9)
                    #ax.set_xlim('No', 'Yes')
                    ax.set_xlabel('Anaemia Patient =1', fontsize=12)
                    ax.set_ylabel('Number of Occurrences', fontsize=12)
                    st.title('Frequency Distribution of Anaemia Patients')
                    #ax.plot.ylabel('Number of Occurrences', fontsize=12)
                    #ax.plot.xlabel('Yes or No', fontsize=12)
                    st.pyplot(fig)
                    
                    #death_count = df['DEATH EVENT'].value_counts()
                    #fig1, ax1 = plt.subplots(figsize=(7, 3))
                    #ax1 = sns.heatmap(df.drop(['platelets'], axis=1))                    
                    #ax1 = sns.heatmap(df.drop(['age', 'creatinine_phosphokinase', 'platelets', 'ejection_fraction','serum_creatinine','serum_sodium', 'time'], axis=1))
                    #st.pyplot(fig1)
                    #ax2 = sns.pairplot(df)
                    #st.pyplot(ax2)
                    if st.checkbox("Area Chart"):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect('Choose a Feature', all_columns)
                        new_df = df[feat_choices]
                        st.area_chart(new_df)
                    if st.checkbox("Line Chart"):
                        line_columns = df.columns.to_list()
                        line_choices = st.multiselect('Choose a Feature', line_columns, key = "<uniquevalueofsomesort>")
                        line_df = df[line_choices]
                        st.line_chart(line_df)
                        
                        
                elif activity == "Prediction":
                    st.subheader("Predictive Analytics")
                    #age_slider = st.slider("How old are you?", 0, 135, 25)
                    #age_range_slider = st.slider("How old are you?", 0, 135, (25, 75))
                    age = st.number_input("Age", 5,150)
                    anaemia = st.radio("Do you have Anaemia? 0:No, 1:Yes", tuple((0, 1)))
                    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 20,8000)
                    diabetes = st.radio("Do you have diabetes? 0:No, 1:Yes", tuple((0, 1)))
                    #diabetes = st.radio("Do you have diabetes? ", tuple(feature_dict.keys()))
                    ejection_fraction = st.number_input("Ejection Fraction", 10, 90)
                    high_blood_pressure = st.radio("Do have High Blood Pressure? 0:No, 1:Yes", tuple((0, 1)))
                    platelets = st.number_input("Platelets", 25000,900000)
                    serum_creatinine = st.number_input("Serum Creatinine", 0.1,10.0)
                    serum_sodium = st.number_input("Serum Sodium", 110,148)
                    sex = st.selectbox("Sex: 0 is Male, 1 is Female", tuple((0, 1)))
                    #sex = st.selectbox("Sex", tuple(gender_dict.keys()))
                    smoking = st.selectbox("Do you smoke? 0:No, 1:Yes",tuple((0,1)) )
                    time = st.number_input("Time", 3, 290)
                    feature_list = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]
                    # I will have used the functions if i had used them, converting male to 0, and female to 1
                    # eg. feature_list=[age, get_value(sex, gender_dict), get_fvalue(diabetes, feature_dict)]
                    #st.select_slider("Select a model of your choice", options=['ANN', 'LC', 'SVN', 'DT'])
                    pretty_Result ={'Age':age, 'anaemia':anaemia, 'creatinine_phosphokinase':creatinine_phosphokinase, 'diabetes':diabetes, 'ejection_fraction':ejection_fraction, 'high_blood_pressure':high_blood_pressure, 'Platelets':platelets, 'serum_creatinine':serum_creatinine, 'Serum Sodium':serum_sodium, 'Sex':sex, 'Smoking':smoking, 'Time':time }
                    st.json(pretty_Result)
                    simple_sample = np.array(feature_list).reshape(1,-1)
                    
                    # Machine Learning Model
                    model_choice = st.selectbox("Select Model", ["LR", "XGBOOST"])
                    if st.button("Predict"):
                        if model_choice =="XGBOOST":
                            loaded_model= load_model("heart_failure.py")
                            prediction = loaded_model.predict(simple_sample)
                        
                    
                elif activity == "Metrics":
                    st.subheader("Data Metrics")
                
            else:
                st.warning("Incorrect Username/Password")
            
    elif choice == "Signup":
        new_username = st.text_input("User Name")
        new_password = st.text_input("Password", type='password')
        confirmed_password = st.text_input("Confirm Password", type='password')
        if new_password == confirmed_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")
        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username, hashed_new_password)
            st.success("You have successfully created a new account")
            st.info("Login To Get Started")             
    
if __name__ == '__main__':
    main()
    
