#!/usr/bin/env python
# coding: utf-8

# In[110]:


def clean_datafield(data, col_name, convert_type=None, val_to_replace=None, val_replacement=None, replacement_array=None):
    """Takes inputs of data as pandas.dataframe and column name as string
    Returns a pandas series of clean values
    """
    import numpy as np
    import pandas as pd
    outval = pd.Series()
    if val_to_replace != []:
        if type(val_replacement) == list:
            #pass
            outval = np.where(data[col_name]==val_to_replace, sum([data[val_replacement[i]]*replacement_array[i] for i in range(len(replacement_array))]), data[col_name])
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
        out_df[column_name] = clean_datafield(data, column_name, dict_look[0], dict_look[1], dict_look[2], dict_look[3])
    return out_df


# In[1]:


def renovated_cat(series, n_years):
    if series == 0:
        return 'Never Renovated'
    elif (2015-n_years) <= series <= 2015:
        return 'Since {} inclusive'.format(2015-n_years)
    elif  series < (2015-n_years):
        return 'Prior to {}'.format(2015-n_years)
    else:
        return 'missing'


# In[3]:


def set_to_categorical(df, listofcolumns):
    for col in listofcolumns:
        df[col] = df[col].astype('category')


# In[ ]:




