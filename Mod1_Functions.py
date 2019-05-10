#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for deleting python script to easily resave with same name
#!rm Mod1_Functions.py


# In[11]:


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


sns.set()


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
    try:
        df_col_list.remove(y)
    except:
        pass
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


# In[12]:


df_out=pd.DataFrame


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
        df_out.drop(['startdate'], axis=1)
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


# In[4]:


def plot_mse_train_test(x, y, start_test_pct=5, test_pct_inc=5, end_test_pct=100, num_iter=50):   
    linreg = LinearRegression()

    mse_train_list = []
    mse_test_list = []
    s_list = []
    
    for s in range(start_test_pct, end_test_pct, test_pct_inc):
        mse_train_sum = 0
        mse_test_sum = 0
        for i in range(num_iter):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = s/100)
            linreg.fit(X_train, y_train)
            y_hat_train = linreg.predict(X_train)
            y_hat_test = linreg.predict(X_test)
            train_residuals = y_hat_train - y_train
            test_residuals = y_hat_test - y_test
            mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
            mse_test =np.sum((y_test-y_hat_test)**2)/len(y_test)
            mse_train_sum += mse_train
            mse_test_sum += mse_test 

        mse_train_list.append(mse_train_sum / num_iter)
        mse_test_list.append(mse_test_sum / num_iter)
        s_list.append(s)


    plt.figure(figsize=(10,6))
    sns.scatterplot(s_list, mse_train_list, label="Training Error")
    sns.scatterplot(s_list, mse_test_list, label="Testing Error")
    plt.xlabel('Testing Data Size as Percentage of Total Dataset')
    plt.ylabel('Mean Square Error (Average of {} trials)'.format(num_iter))
    plt.legend()
    plt.show()


# In[3]:


def residual_hist_qq(model):
    resid1 = model.resid
    resid1.hist()
    fig = sm.graphics.qqplot(resid1, dist=stats.norm, line='45', fit=True)
    fig.show()


# In[9]:


def plot_RFE_var_iter(X, Y, k_fold_n_splits=5, shuffle=True, scoring='neg_mean_squared_error'):


    n = len(X.columns)

    mse_list = []
    r2_list = []
    num_var = list(range(1, n+1))

    for i in range(n):
        linreg = LinearRegression()
        selector = RFE(linreg, n_features_to_select=i+1)
        selector = selector.fit(X, Y.squeeze())
        selected_X = X.columns[selector.support_]
        df_selected = X.loc[:, selected_X]
        df_selected = sm.add_constant(df_selected)
        model_kfold = KFold(n_splits=k_fold_n_splits, shuffle=shuffle)
        MSEs = cross_val_score(linreg, df_selected, Y,
                               scoring=scoring, cv=model_kfold)
        mean_MSE = -1 * np.mean(MSEs)
        model = sm.OLS(Y, df_selected).fit()

        r2_list.append(model.rsquared_adj)
        mse_list.append(mean_MSE)

    plt.figure(figsize=(10, 7))
    plt.plot(r2_list, label='Adjusted $r^{2}$')
    plt.xlabel(r"Number of X Variables Included in Model")
    plt.ylabel(r"Mean Squared Error / Adjusted r squared")
    plt.title("MSE and Adjusted $r^{2}$ versus Number of Model Variables")
    plt.legend()

    plt.plot(mse_list, label='Mean Squared Error')
    plt.legend()
    plt.show()


# In[3]:


def preprocess_data(df, categorical_columns=[], log_list=[],  min_max_list=[], std_scal_list=[], dropout_list=[], cat_drop_dict={}):
    """Custom preprocessing"""


# create temporary data frame to work with
    df_temp = df.copy()

# handle categorical columns
# set to categorical, then dummy out and store in df_dummy
    if len(categorical_columns) > 0:
        set_to_categorical(df_temp, categorical_columns)

        print("Categorical Variables:")
        print(df_temp.dtypes[df_temp.dtypes == 'category'])
        print("\n")

        df_dummy = create_dummyframe(df_temp, categorical_columns)

        drop_list = []

        for drop_col in list(df_temp.dtypes[df_temp.dtypes == 'category'].keys()):
            try:
                field_name = cat_drop_dict[drop_col]
            except:
                field_name = drop_col + '_' + str(df_temp[drop_col].cat.categories[0])

            drop_list.append(field_name)

        df_dummy.drop(drop_list, axis=1, inplace=True)
        print('To avoid multicollinearity, the following datafields were dropped: {}'.format(", ".join(drop_list)))
        df_temp.drop(categorical_columns, axis=1, inplace=True)
    else:
        print('no categorical dummy columns added')
        df_dummy = pd.DataFrame()

        

#  natural log of variables, note return of 0 for non-positive values
    if len(log_list) > 0:
        print('\n')
        for i in log_list:
            df_temp[i] = df_temp[i].apply(lambda x: np.log(x) if x > 0 else 0)
        print('Converted the following datafields to natural log: {}'.format(", ".join(log_list)))


# Min Max scaling
    print('\n')
    if len(min_max_list) > 0:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_min_max = pd.DataFrame(columns=min_max_list)
        for col_name in min_max_list:
            X_min_max[col_name] = df_temp[col_name]
            print('Converted {0} to scale min-max: [{1:.2f}, {2:.2f}] to [0,1]'.format(
                col_name, df_temp[col_name].min(), df_temp[col_name].max()))

        X_min_max = pd.DataFrame(data=min_max_scaler.fit_transform(
            X_min_max.values), columns=X_min_max.columns, index=X_min_max.index)
        df_temp.drop(min_max_list, axis=1, inplace=True)
    else:
        X_min_max = pd.DataFrame()
        print('No variables scaled with min-max scaler')


# Standard scaling
    print('\n')
    if len(std_scal_list) > 0:
        standard_scaler = preprocessing.StandardScaler()
        X_std_scal = pd.DataFrame(columns=std_scal_list)
        for col_name in std_scal_list:
            X_std_scal[col_name] = df_temp[col_name]
            print('Standardized {0} to scale with mean=0 and std=1 from: [{1:.3f}, {2:.3f}] to [0,1]'.format(
                col_name, df_temp[col_name].mean(), df_temp[col_name].std()))

        X_std_scal = pd.DataFrame(data=standard_scaler.fit_transform(
            X_std_scal.values), columns=X_std_scal.columns, index=X_min_max.index)
        df_temp.drop(std_scal_list, axis=1, inplace=True)
    else:
        X_std_scal = pd.DataFrame()
        print('No variables scaled with standard scaler')

# drop any columns in the drop list
    if len(dropout_list)>0:
        print('\n')
        for col in dropout_list:
            try:
                df_temp.drop(col, axis=1, inplace=True)
                print("Dropped {} from the output dataset".format(col))
            except:
                try:
                    df_dummy.drop(col, axis=1, inplace=True)
                    print("Dropped {} from the output dataset".format(col))
                except:
                    try:
                        X_min_max.drop(col, axis=1, inplace=True)
                        print("Dropped {} from the output dataset".format(col))
                    except:
                        try:
                            X_std_scal.drop(col, axis=1, inplace=True)
                            print("Dropped {} from the output dataset".format(col))
                        except:
                            print('Could not find {} to drop'.format(col))
                    
        
    
# concatenate all of these dataframes
    X_possible = pd.concat([X_min_max, X_std_scal, df_temp, df_dummy], axis=1)
    
    return X_possible


# In[20]:


def interpret_coef(model, df_original, categorical_columns=[], log_list=[],  min_max_list=[], std_scal_list=[]):

    var_coef = model.params[1:]
    adj_coef = []
    interpret = []

    for col, coef in var_coef.items():
        if col in min_max_list:
            adj_coef.append((np.exp(coef)-1) * 100 /
                            (df_original[col].max() - df_original[col].min()))

        elif col in std_scal_list:
            adj_coef.append((np.exp(coef)-1) * 100 / (df_original[col].std()))

        elif col in log_list:
            adj_coef.append(coef)
        else:
            adj_coef.append((np.exp(coef)-1) * 100)

    for i, col in enumerate(var_coef.keys()):
        if col in (min_max_list + std_scal_list):
            interpret.append("Price is expected to increase by {0:.4f}% for each unit increment beyond {1:.2f} in {2}".format(
                adj_coef[i], df_original[col].min(), col))

        elif col in log_list:
            interpret.append(
                "Price is expected to increase by {0:.4f}% for each 1% increase in {1}".format(adj_coef[i], col))
        elif str(col).split('_')[0] in categorical_columns:
            interpret.append(
                "Price is expected to increase by {0:.4f}% if condition met for {1}".format(adj_coef[i], col))
        else:
            interpret.append("Price is expected to increase by {0:.4f}% for each unit increment in {1}".format(
                adj_coef[i], col))
    return pd.DataFrame(list(zip(list(var_coef.keys()), list(var_coef.values),adj_coef,  interpret)), columns=['variable', 'coef', 'adj_coef', 'interpretation'])


# In[21]:


from IPython.display import HTML


# In[ ]:




