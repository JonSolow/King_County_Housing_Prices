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
import math
from math import radians, cos, sin, asin, sqrt

from sklearn.model_selection import train_test_split


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


# In[ ]:


def bathroom_bins(bathroom):
    return math.ceil(bathroom)


# In[ ]:



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# In[ ]:


def historic_home(age):
    if age>=80:
        return 1
    else:
        return 0


# In[6]:


def add_features_to_df(df):
    """Output is a new dataframe with our additional features added"""
    df_out = df.copy()
    try:
        df_out['month'] = df_out['date'].map(lambda x: x.month)
        df_out['season'] = df_out['month'].apply(create_season)
    except:
        print('Error: date was not found in dataframe.  Month and season could not be created')

    try:
        df_out['has_basement'] = df_out['sqft_basement'].apply(
            lambda x: 1 if x > 0 else 0)
    except:
        print('Error: sqft_basement was not found in dataframe.  has_basement could not be created')

    try:
        df_out['sqft_above_tophalf'] = df_out['sqft_above'].apply(
            lambda x: 1 if x > df_out['sqft_above'].mean() else 0)
    except:
        print('Error: sqft_above was not found in dataframe.  sqft_above_tophalf could not be created')

    try:
        df_out['zip_highprice'] = df_out['zipcode'].apply(
        lambda x: 1 if x in [98004, 98039, 98040, 98112] else 0)
    except:
        print('Error: zipcode was not found in dataframe.  zip_highprice could not be created')

# Set number of years to consider recent renovation
    try:
        n_years = 15
        df_out['yr_renovated_cat'] = df_out['yr_renovated'].apply(
            renovated_cat, n_years=n_years)
    except:
        print('Error: yr_renovated was not found in dataframe.  yr_renovated_cat could not be created')

    
    try:
        df_out['startdate'] = pd.Timestamp('19700101')
        df_out['date_num'] = (df_out['date'] - df_out['startdate']).dt.days
    except:
        print('Error: date was not found in dataframe.  date_num could not be created')

    try:
        # Calculated Distance from City Center
        df_out['dist_city_center'] = df_out.apply(
            lambda x: haversine(-122.3344, 47.6050, x.loc['long'], x.loc['lat']), axis=1)
    except:
        print('Error: long and/or lat was not found in dataframe.  dist_city_center could not be created')
        
        
    try:
        df_out['bathroom_bins'] = df_out.bathrooms.apply(bathroom_bins)
    except:
        print('Error: bathrooms was not found in dataframe.  bathroom_bins could not be created')
        
    try:
        df_out['age'] = df_out['date'].apply(lambda x: x.year) - df_out['yr_built']
        df_out['historic_home'] = df_out.age.apply(historic_home)
    except:
        print('Error: date and/or yr_built was not found in dataframe.  age and historic_home could not be created') 
        
    try:
        df_out['grade_bin'] = df_out.grade.apply(grade_bins)
    except:
        print('Error: grade was not found in dataframe.  grade_bin could not be created') 
    
    return df_out


# In[12]:


def filter_df_quantiles(df, filter_dict):
    df_filter = df.copy()

    for k, v in filter_dict.items():
        min_lim = df[k].quantile(v[0])
        max_lim = df[k].quantile(v[1])

        n_less = len(df_filter[df_filter[k] < min_lim])
        n_greater = len(df_filter[df_filter[k] > max_lim])

        filter1 = df_filter[k] >= min_lim
        filter2 = df_filter[k] <= max_lim

        print('filtered out {0} records with {1} less than: {2:.2f}'.format(n_less, k, min_lim))
        print('filtered out {0} records with {1} greater than: {2:.2f}'.format(n_greater,  k, max_lim))

        df_filter = df_filter[filter1 & filter2]

    print('{} total records removed'.format(len(df) - len(df_filter)))
        
    return df_filter


# In[5]:


def grade_bins(grade):
    if grade >= 11:
        return 11
    elif grade <= 5:
        return 5
    else:
        return grade


# In[ ]:




