#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cat_
import seaborn as sns
import lightgbm as lgb
from scipy import stats


# In[143]:


# Importing datasets
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")
#train_dummy = pd.get_dummies(train_data.drop(['uniqueid'], axis=1))
#test_dummy = pd.get_dummies(test_data.drop(['uniqueid'], axis=1))


# In[144]:


# Merging the train and test data
ntrain = train_data.shape[0] 
ntest = test_data.shape[0]
dataset = pd.concat((train_data, test_data), sort=False).reset_index(drop=True)
dataset['year'].unique()
dataset.year.unique()


# # VISUALIZATION

# In[145]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
gender_count = dataset['gender_of_respondent'].value_counts()
sns.set(style="darkgrid")
sns.barplot(gender_count.index, gender_count.values, alpha=0.9)
plt.title('Frequency Distribution of Gender')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.show()


# In[146]:


y_train = train_data.bank_account.values
y_train.shape


# In[147]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
bank_account_count = train_data['bank_account'].value_counts()
sns.set(style="darkgrid")
sns.barplot(bank_account_count.index, bank_account_count.values, alpha=0.9)
plt.title('Frequency Distribution of Bank Accounts')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Bank Account', fontsize=12)
plt.show()


# In[148]:


train_data['bank_account'].value_counts()


# In[149]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
education_level_count = dataset['education_level'].value_counts()
sns.set(style="darkgrid")
sns.barplot(education_level_count.index, education_level_count.values, alpha=0.9)
plt.title('Frequency Distribution of Education Level')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Education Level', fontsize=12)
plt.show()


# In[150]:


dataset['education_level'].value_counts()


# # futher Analysis can be done in excel pivot table. comparing education levels to bank account created

# In[151]:


sns.heatmap(dataset[['age_of_respondent','household_size']])


# In[152]:


dataset[['age_of_respondent','household_size']].plot.scatter(x = 'age_of_respondent',y = 'household_size')


# In[153]:


#Pie Chart
labels = dataset['education_level'].astype('category').cat.categories.tolist()
counts = dataset['education_level'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[154]:


dataset.head()
#dataset = dataset.drop(['bank_account'], axis=1)
dataset.education_level.unique()
dataset.info()


# In[155]:


#dataset.head()


# ## Replacing some binary response with 1 or 0
# ##### There will be another model that will use encoder instead of replacing method

# In[156]:


#train_data['bank_account'] = train_data['bank_account'].str.replace('Yes','1')
#train_data['bank_account'] = train_data['bank_account'].str.replace('No','0')
y_train = train_data.bank_account.values


# In[157]:


y_train.shape


# In[158]:


#dataset['cellphone_access'] = dataset['cellphone_access'].str.replace('Yes','1')
#dataset['cellphone_access'] = dataset['cellphone_access'].str.replace('No','0')
#dataset['gender_of_respondent'] = dataset['gender_of_respondent'].str.replace('Male','1')
#dataset['gender_of_respondent'] = dataset['gender_of_respondent'].str.replace('Female','0')
#dataset['year'] = dataset['year'].replace(2018,1)
#dataset['year'] = dataset['year'].replace(2017,2)
#dataset['year'] = dataset['year'].replace(2016,3)


dataset.head()


# In[159]:


dataset = dataset.drop(['uniqueid'], axis=1)


# In[160]:


dataset.head()


# In[161]:


dataset = dataset.drop(['bank_account'], axis=1)


# In[162]:


data = pd.get_dummies(dataset)
data_1 = data #For checking distributions


# In[163]:


data_1.head()


# In[164]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()


# In[165]:


data.head()


# In[166]:


y_train = LE.fit_transform(y_train)


# In[167]:


y_train


# In[168]:


#data.head()


# In[169]:


#data.describe()


# # SCALLING OF SOME COLUMNS
# ### Before scaling, the type of distribution must be checked first. This will help you know the scale that must be used

# In[170]:


data_mean = np.mean(data_1[['household_size', 'age_of_respondent']])
data_median_hhsiz = np.median(data_1['household_size'])
data_median_age = np.median(data_1['age_of_respondent'])
data_mode = stats.mode(data_1[['household_size', 'age_of_respondent']])
print("The means for the various columns are: {} ".format(data_mean))
print("The median for the household_size is : {}".format(data_median_hhsiz))
print("The median for the Age_of_respondant is : {}".format(data_median_age))
print("The modes for the various columns are: {}".format(data_mode))


# ## Method 1.
# ### Min - Maximum Scale.
# ##### formular => Xnew = ((X-Xmin)/(Xmax - Xmin))
# This scale is sensitive to outliers and its been used when the data is not Gaussian(Normal distribution)
# To check normalisation, mean must be between mode & Median.
# From the distribution of the two columns, the mean is greater than the mode and median. this indcates that the data is not Gaussian. 

# In[171]:


from sklearn.preprocessing import MinMaxScaler
df = MinMaxScaler()
data[['year','household_size','age_of_respondent']] = df.fit_transform(data[['year','household_size','age_of_respondent']])


# In[172]:


data.year.unique()


# # 2. Standard Scaler:
# ### It assumes the data is normal distribution and its centered around 0, with a standard deviation of 1
# z = (x - u) / s
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# ## 3. Max Abs Scaler
# #### Scale each feature by its maximum absolute value. This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set is 1.0. It does not shift/center the data and thus does not destroy any sparsity.
# On positive-only data, this Scaler behaves similarly to Min Max Scaler and, therefore, also suffers from the presence of significant outliers.
# from sklearn.preprocessing import MaxAbsScaler
# scaler = MaxAbsScaler()

# ## 4) Robust Scaler
# #### As the name suggests, this Scaler is robust to outliers. If our data contains many outliers, scaling using the mean and standard deviation of the data won’t work well.
# This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile). The centering and scaling statistics of this Scaler are based on percentiles and are therefore not influenced by a few numbers of huge marginal outliers. Note that the outliers themselves are still present in the transformed data. If a separate outlier clipping is desirable, a non-linear transformation is required.
# ##### from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()

# ## 5) Quantile Transformer Scaler
# Transform features using quantiles information.
# This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is, therefore, a robust pre-processing scheme.
# The cumulative distribution function of a feature is used to project the original values. Note that this transform is non-linear and may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable. This is also sometimes called as Rank scaler.
# #### from sklearn.preprocessing import QuantileTransformer
# scaler = QuantileTransformer()

# ## 6) Power Transformer Scaler
# The power transformer is a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to the variability of a variable that is unequal across the range (heteroscedasticity) or situations where normality is desired.
# The power transform finds the optimal scaling factor in stabilizing variance and minimizing skewness through maximum likelihood estimation. Currently, Sklearn implementation of PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data.
# #### from sklearn.preprocessing import PowerTransformer
# scaler = PowerTransformer(method='yeo-johnson')

# In[173]:


#Train and test datasets
train = data[:ntrain].copy()
test = data[ntrain:].copy()
test = test.reset_index(drop=True)


# In[174]:


train.shape


# In[175]:


test.shape


# In[176]:


data.shape


# In[177]:


data.head()


# In[178]:


#pip install rpy2


# In[179]:


train.head()


# In[180]:


# lightgbm for classifier
#from numpy import mean
#from numpy import std
from lightgbm import LGBMClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot

# evaluate the model
#model = LGBMClassifier()
# fit the model on the whole dataset
model_lg = LGBMClassifier()
model_lg.fit(train, y_train)
y_pred_lgb = model_lg.predict_proba(test.values)[:,1]


# In[181]:


y_pred_lgb


# In[182]:


# catboost for regression
from catboost import CatBoostClassifier
from matplotlib import pyplot
# evaluate the model
model = CatBoostClassifier(verbose=0, n_estimators=100)
# fit the model on the whole dataset
model_cat = CatBoostClassifier(verbose=0, n_estimators=100)
model_cat.fit(train, y_train)
y_pred_cat1 = model_cat.predict_log_proba(test)
y_pred_cat = model_cat.predict_proba(test.values)[:,1]


# In[183]:


y_pred_cat


# In[184]:


# xgboost for regression
from xgboost import XGBClassifier
# fit the model on the whole dataset
model_xg = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='mlogloss')
model_xg.fit(train, y_train)
y_pred_xg = model_xg.predict_proba(test.values)[:,1]


# In[185]:


y_pred_xg


# In[186]:


from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingClassifier


# In[187]:


#conda install mlxtend


# In[188]:


# make a prediction with a stacking mlxtend
from sklearn.linear_model import LinearRegression
import mlxtend
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression

# define meta learner model
lr = LogisticRegression()
# define the stacking ensemble
model = StackingClassifier(classifiers=[model_lg, model_cat, model_xg], meta_classifier=lr)
# fit the model on all available data
model = model.fit(train, y_train)
#pred = list(pred.ravel())
#stack_result = list(stack_result.ravel())


# In[199]:


stack_result = model.predict(test)
#stack_result = model.predict_proba(test)[:,1]


# In[200]:


stack_result


# In[197]:


#submit = stack_result[['bank_account']].copy()
#submit['bank_account'] = stack_result


# In[198]:


Submission1 = pd.DataFrame(stack_result, columns=['bank_account']).to_csv('Submission1.csv')
#pd.DataFrame(stack_result).to_csv("bank_account.csv")


# In[121]:


Submission11 = pd.read_csv('Submission11.csv')
Submission11['bank_account'] = stack_result #model_lg.predict(test) 
file_name = 'submission11.csv'
stack_result.to_csv(file_name ,index=False)


# In[ ]:




