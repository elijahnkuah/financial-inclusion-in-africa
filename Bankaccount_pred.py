# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 22:28:21 2021

@author: Elijah_Nkuah
"""
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

train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")
train_dummy = pd.get_dummies(train_data.drop(['uniqueid'], axis=1))
pd.get_dummies(test_data)