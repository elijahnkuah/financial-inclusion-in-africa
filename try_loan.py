# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:00:38 2021

@author: Elijah_Nkuah
"""
# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from PIL import Image


# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

# DB
from database_acc import *
#from managed_db import *
# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False


feature_names_best = ['country', 'year', 'location_type',
       'cellphone_access', 'household_size', 'age_of_respondent',
       'gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level', 'job_type']


gender_dict = {"Male":1,"Female":2}
country_dict = {'Kenya':1, 'Rwanda':2, 'Tanzania':3, 'Uganda':4}
year_dict = {'2018':1, '2017':2, '2016':3}
feature_dict = {"No":0,"Yes":1}
location_dict = {'Rural':1, 'Urban':2}
relationship_dict = {'Head of Household':1, 'Spouse':2, 'Child':3, 'Parent':4, 'Other relative':5, 'Other non-relatives':6}
marital_dict = {'Married/Living together':1, 'Single/Never Married':2, 'Widowed':3,'Divorced/Seperated':4, 'Dont know':5}
education_dict = {'No formal education':0,'Primary education':1,'Secondary education':2,'Vocational/Specialised training':3,
                  'Tertiary education':4, 'Other/Dont know/RTA':5}
job_dict = {'Dont Know/Refuse to answer':0,'No Income':1,'Other Income':2,'Remittance Dependent':3,'Government Dependent':4,
            'Self employed':5,'Farming and Fishing':6,'Informally employed':7,'Formally employed Private':8, 
       'Formally employed Government':9}


def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key
def get_cvalue(val):
	country_dict = {'Kenya':1, 'Rwanda':2, 'Tanzania':3, 'Uganda':4}
	for key,value in year_dict.items():
		if val == key:
			return value
def get_yvalue(val):
	year = {'2018':1, '2017':2, '2016':3}
	for key,value in year.items():
		if val == key:
			return value
def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value
def get_lvalue(val):
    location_dict = {'Rural':1, 'Urban':2}
    for key,value in location_dict.items():
            if val == key:
                    return value
def get_rvalue(val):
	relationship_dict = {'Head of Household':1, 'Spouse':2, 'Child':3, 'Parent':4, 'Other relative':5, 'Other non-relatives':6}
	for key,value in relationship_dict.items():
		if val == key:
			return value
def get_mvalue(val):
	marital_dict = {'Married/Living together':1, 'Single/Never Married':2, 'Widowed':3,'Divorced/Seperated':4, 'Dont know':5}
	for key,value in marital_dict.items():
		if val == key:
			return value
def get_evalue(val):
	education_dict = {'No formal education':0,'Primary education':1,'Secondary education':2,'Vocational/Specialised training':3,
                  'Tertiary education':4, 'Other/Dont know/RTA':5}
	for key,value in education_dict.items():
		if val == key:
			return value
def get_jvalue(val):
	job_dict = {'Dont Know/Refuse to answer':0,'No Income':1,'Other Income':2,'Remittance Dependent':3,'Government Dependent':4,
            'Self employed':5,'Farming and Fishing':6,'Informally employed':7,'Formally employed Private':8, 
       'Formally employed Government':9}
	for key,value in job_dict.items():
		if val == key:
			return value

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# ML Interpretation
#pip install lime
import lime
import lime.lime_tabular


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Predicting those who may have Bank Account </h1>
		<h5 style="color:white;text-align:center;">BANK ACCOUNT </h5>
		</div>
		"""

# Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Bank Account Summary</h3>
		<p>It is clear to see that having a bank account in your name give you greater independence and 
        allows you to organise your money and access it easily. You can even save money that you would 
        typically spend on payments when you have a bank account that will process them for free.</p>
	</div>
	"""

@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
	return im
	
st.set_option('deprecation.showPyplotGlobalUse', False)

def change_avatar(sex):
	if sex == "male":
		avatar_img = 'img_avatar.png'
	else:
		avatar_img = 'img_avatar2.png'
	return avatar_img


def main():
	"""Prediction App for persons having bank account or not"""
	st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)

	menu = ["Home","Login","Signup"]
	submenu = ["Plot","Prediction","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		st.image(load_image('images/bank1.JPG'))


	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))
			# if password == "12345":
			if result:
				st.success("Welcome {} to Bank Account Prediction App".format(username))
				st.image(load_image('images/welcome.JPG'))
				activity = st.selectbox("Activity",submenu)
				if activity == "Plot":
					st.subheader("Data Visualisation page - Plots")
					df = pd.read_csv("data/Train.csv")
					st.dataframe(df)
					df['bank_account'].value_counts().plot(kind='bar', color="green")
					st.pyplot()
					#st.dataframe(df).shape
					# Freq Dist Plot
					#freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
					#st.bar_chart(freq_df['count'])

					if st.checkbox("Area Chart"):
						all_columns = df.columns.to_list()
						feat_choices = st.multiselect("Choose a Feature",all_columns)
						new_df = df[feat_choices]
						st.area_chart(new_df)
					if st.checkbox("Line Chart"):
						line_columns = df.columns.to_list()
						line_choices = st.multiselect('Choose a Feature', line_columns, key = "<uniquevalueofsomesort>")
						line_df = df[line_choices]
						st.line_chart(line_df)

				elif activity == "Prediction":
					st.subheader("Predictive Analytics")

					country = st.selectbox("Country of Residence",tuple(country_dict.keys()))
					year = st.radio("Year",year_dict.keys())
					location = st.radio("Location Type",tuple(location_dict.keys()))
					cellphone = st.selectbox("Do You have access to phone? ", tuple(feature_dict.keys()))
					household = st.number_input("How many people do you leave together as one household?",1,25)
					age = st.number_input("What is your Age?", 5,110)
					sex = st.radio("Sex",tuple(gender_dict.keys()))
					relationship = st.selectbox("What is your relationship to the head of your household?", tuple(relationship_dict.keys()))
					marital = st.selectbox("What is your Marital Status?", tuple(marital_dict.keys()))
					education = st.selectbox("What is your level of Education?", tuple(education_dict.keys()))
					job = st.selectbox("What is your Employment Status?", tuple(job_dict.keys()))

					feature_list = [get_cvalue(country),get_yvalue(year),get_lvalue(location),get_fvalue(cellphone),household,age,get_value(sex,gender_dict),get_rvalue(relationship),get_mvalue(marital), get_evalue(education),get_jvalue(job)]
					#feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]
					st.write("The Number of independent varaiables is {}".format(len(feature_list)))
					#st.write(feature_list)
					pretty_result = {"Country":country,"year":year,"location":location,"cellphone":cellphone,"household":household,"Age":age,"Sex":sex,"Relationship":relationship,"Marital Status":marital,"Education Level":education,"Job Type":job}
					#pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# ML
					model_choice = st.selectbox("Select Model",["Lightgbm","Catboost","Logistics Regression","Xgboost"])
					if st.button("Predict"):
						if model_choice == "Lightgbm":
							loaded_model = load_model("models/lgb_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "Catboost":
							loaded_model = load_model("models/cat_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice=="Logistics Regression":
							loaded_model = load_model("models/lr_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							loaded_model = load_model("models/xgb_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						
						if prediction == 1:
							st.success("Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						elif prediction == 2:
							st.success("Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						elif prediction == 3:
							st.success("Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
							
						else:
							st.warning("Not Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							
					if st.checkbox("Interpret"):
						if model_choice == "KNN":
							loaded_model = load_model("knn_hepB_model.pkl")
							
						elif model_choice == "DecisionTree":
							loaded_model = load_model("decision_tree_clf_hepB_model.pkl")
							
						else:
							loaded_model = load_model("logistic_regression_hepB_model.pkl")
							

							# loaded_model = load_model("models/logistic_regression_model.pkl")							
							# 1 Die and 2 Live
							df = pd.read_csv("data/clean_hepatitis_dataset.csv")
							x = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
							feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
							class_names = ['Die(1)','Live(2)']
							explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
							# The Explainer Instance
							exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=13, top_labels=1)
							exp.show_in_notebook(show_table=True, show_all=False)
							# exp.save_to_file('lime_oi.html')
							st.write(exp.as_list())
							new_exp = exp.as_list()
							label_limits = [i[0] for i in new_exp]
							# st.write(label_limits)
							label_scores = [i[1] for i in new_exp]
							plt.barh(label_limits,label_scores)
							st.pyplot()
							plt.figure(figsize=(20,10))
							fig = exp.as_pyplot_figure()
							st.pyplot()



					


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
