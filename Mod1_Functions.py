#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for deleting python script to easily resave with same name
#!rm Mod1_Functions.py


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
import scipy.stats as stats


# In[ ]:


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[110]:


def clean_datafield(data, col_name, convert_type=None, val_to_replace=None, val_replacement=None, replacement_array=None):
    """Takes inputs of data as pandas.dataframe and column name as string
    Returns a pandas series of clean values
    """
    outval = pd.Series()
    if val_to_replace != None:
        if type(val_replacement) == list:
            # pass
            outval = np.where(data[col_name] == val_to_replace, sum(
                [data[val_replacement[i]]*replacement_array[i] for i in range(len(replacement_array))]), data[col_name])
        else:
            outval = data[col_name].copy().replace(
                to_replace=val_to_replace, value=val_replacement)
    else:
        outval = data[col_name].copy()

    if convert_type != None:
        outval = outval.astype(convert_type)
    return outval


# In[137]:


# values of adjustment dictionary in format of [convert_type, val_to_replace, val_replacement]

#data_adjustments = {'date': ['datetime64', None, None, None], 
#                    'bedrooms': [None, 33, 4, None], 
#                    'waterfront': [str, np.nan, 'missing', None], 
#                    'view': [str, np.nan, 0, None],
#                    'sqft_basement': [float, '?', ['sqft_living','sqft_above'], [1, -1]]
#                   }


# In[138]:


def clean_dataframe(data, data_adj={}):
    import numpy as np
    import pandas as pd
    out_df = pd.DataFrame()
    for column_name in data.columns:
        if column_name in data_adj.keys():
            dict_look = data_adj[column_name]
        else:
            dict_look = [None, None, None, None]
        out_df[column_name] = clean_datafield(
            data, column_name, dict_look[0], dict_look[1], dict_look[2], dict_look[3])
    return out_df


# In[1]:


def renovated_cat(series, n_years):
    if series == 0:
        return 'Never Renovated'
    elif (2015-n_years) <= series <= 2015:
        return 'Since {} inclusive'.format(2015-n_years)
    elif series < (2015-n_years):
        return 'Prior to {}'.format(2015-n_years)
    else:
        return 'missing'


# In[3]:


def set_to_categorical(df, listofcolumns):
    for col in listofcolumns:
        df[col] = df[col].astype('category')


# In[7]:


def scatter_y(df, y, ncols=3, figsize=(16, 20), wspace=0.2, hspace=0.5, alpha=0.05, color='b'):
    df_col_list = list(df.columns)
    if (len(df_col_list) % ncols > 0):
        nrows = len(df_col_list)//ncols + 1
    else:
        nrows = len(df_col_list)//ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    for i, xcol in enumerate(df_col_list):
        try:
            df.plot(kind='scatter', x=xcol, y=y,
                    ax=axes[i//ncols, i % ncols], label=xcol, alpha=alpha, color=color)
            plt.plot()
        except:
            print('warning: could not graph {} as numbers'.format(xcol))


# In[6]:


def create_dummyframe(df, listofcolumns):
    df_dummy = pd.DataFrame()
    for cat_col in listofcolumns:
        df_dummy = pd.concat([df_dummy, pd.get_dummies(df[cat_col], prefix=cat_col)], axis=1)
    return df_dummy


# In[4]:


def create_season(month):
    month = str(month)
    if month in ['12', '1', '2']:
        return 'Winter'
    elif month in ['3', '4', '5']:
        return 'Spring'
    elif month in ['6', '7', '8']:
        return 'Summer'
    elif month in ['9', '10', '11']:
        return 'Fall'


# In[ ]:


def findcorrpairs(df, corr_thresh=0.75, round=2):
    corr_mx = df.corr()
    pairs_list = []
    corr_list = []
    for col in corr_mx.columns:
        for col2 in corr_mx.columns:
            if col != col2:
                pair_corr = corr_mx[col][col2]
                if abs(pair_corr)>corr_thresh:
                    col_sort = [col, col2]
                    col_sort.sort()
                    pairs_list.append(col_sort)
                    corr_list.append(pair_corr.round(round))
    df_out = pd.DataFrame()
    df_out['Pairs'] = pairs_list
    df_out['PairsText'] = df_out['Pairs'].astype('str')
    df_out['Correlation'] = corr_list
    df_out.drop_duplicates(subset='PairsText', inplace = True)
    df_out.sort_values('Correlation', ascending=False, inplace=True)
    df_out.drop('PairsText',  axis=1, inplace=True)
    return df_out

