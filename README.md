
<div style="text-align:center; font-weight:bold; font-size:35px; line-height:1.0;">EDA: Sell Your House for More </div>

<figure style="margin-top:10px"><img src="https://www.racialequityalliance.org/wp-content/uploads/2016/10/assessors_social-1.jpg" />
<figcaption style="text-align:center; font-size:8px">source: https://www.racialequityalliance.org/jurisdictions/king-county-washington/assessors_social/</figcaption>
</figure>




# Project Goal

## How can we get more money for our house in King County?

We are selling our house and would like to know which characteristics of a home can help improve the sale price.

In order to help answer this question, we have been provided a dataset of home sales in King County which occurred during the period of September 9, 2014 through January 10, 2015.

We will attempt to model sale prices based on the other data fields and then determine which characteristics lead to an increase in sale price.

# Initial Imports and Data Cleaning

## Our Dataset

The data was provided in a csv format.  There are 21,597 sales records and 21 columns.

Column Names and descriptions:
* **id** - unique identified for a house
* **dateDate** - house was sold
* **pricePrice** -  is prediction target
* **bedroomsNumber** -  of Bedrooms/House
* **bathroomsNumber** -  of bathrooms/bedrooms
* **sqft_livingsquare** -  footage of the home
* **sqft_lotsquare** -  footage of the lot
* **floorsTotal** -  floors (levels) in house
* **waterfront** - House which has a view to a waterfront
* **view** - Number of quality (beautiful) views from the house
* **condition** - How good the condition is on a scale of 1-5
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house apart from basement
* **sqft_basement** - square footage of the basement
* **yr_built** - Built Year
* **yr_renovated** - Year when house was renovated
* **zipcode** - zip
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors

## Import Libraries and Custom Functions

The following code will import our custom functions from Mod1_Functions.py.

We also import pandas, numpy, matplotlib, seaborn, statsmodels, and scikit-learn.


```python
import sys
import os
sys.path.append("../src/")
```


```python
from Mod1_Functions import *
warnings.simplefilter("ignore")
```

## Import Raw Dataset

We import the csv data into a pandas dataframe by using `pandas.read_csv()`.


```python
df_raw = pd.read_csv('../data/kc_house_data.csv')
```

From below, we can see that `waterfront`, `view`, and `yr_renovated` are missing values.


# Variable Exploration

Let's take a look at some scatter plots of each potential X variable compared to our target `'price'`.

The custom function below accepts inputs of a dataframe and the name of the column that is the target.  Additional parameters can also be passed through to adjust the formatting.

Warnings are printed for any variables that cannot immediately be graphed as numbers, due to their datatypes.


```python
graph_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15', 'floors', 'grade', 'condition']
df_graph = df_raw.loc[:, graph_columns]
scatter_y(df_graph, 'price', ncols=2, figsize=(16,16))

```


![png](images/output_16_0.png)


Taking a look at a histogram of the data fields provides additional insights into how we may want to adjust certain variables.

For example, the data fields of sqft_living15 and sqft_above are right-skewed, with a large grouping of data points in the lower range.


```python
df_raw.iloc[:, 2:].hist(figsize  = [16, 10])
plt.subplots_adjust(wspace=0.5, hspace=0.6)
```


![png](images/output_18_0.png)



```python
missing = df_raw.shape[0] - df_raw.count()
print(missing[missing>0])
```

    waterfront      2376
    view              63
    yr_renovated    3842
    dtype: int64


## Clean data using custom function

Our `clean_dataframe()` function takes in inputs of dataframe and a dictionary of adjustments.

We first set our `data_adjustments` dictionary to contain the fields we want to change as keys and a list of adjustments as dictionary values.

The parameters in the list are: \[datatype, value to replace, value to replace with, replacement array\]

    - datatype: must be a valid data type
    - value to replace: can be a single value string, integer, or np.nan
    - value to replace with: can be a single value or can be a list with strings containg other column names in dataframe (see replacement array below)
    - replacement array: contains a list of floats or integers, which are multiplied by the associated data field in the "value to replace with" list
    
The list items should be set to None for any parameters you do not wish to use.

We found some issues with the following data fields and decided to make some adjustments.

- `'date'` : Currently the date is formatted as a string and we would like to change it into a date (datetime64) so that we can use it in our model.


- `'bedrooms'` : There was one unusually high value of 33 bedrooms for one record.  We decided to replace that value with 3, based on the properties nearest neighbors by `'price'` and `'sqft_living'`.  It is also reasonable to assume that that the 33 was an accidental double-keystroke.


- `'waterfront'`: There were 2,376 missing values in this field, which we have replaced with 'missing'.  We don't know whether these records are on the waterfront or not, but it is a significant enough proportion of records (11%), so we don't want to make an assumption about whether they are on the waterfront and also don't want to exclude them from the dataset.


```python
data_adjustments = {'date': ['datetime64', None, None, None],
                    'bedrooms': [None, 33, 3, None],
                    'waterfront': [str, np.nan, 'missing', None],
                    'view': [str, np.nan, 0, None],
                    'sqft_basement': [float, '?', ['sqft_living', 'sqft_above'], [1, -1]],
                    'floors': [None, 3.5, 3, None]
                   }
```


```python
df_clean = clean_dataframe(df_raw, data_adjustments)
```

## Add features calculated from other columns

We wondered whether there was any seasonality to home sale prices, so we needed to extract the month 

We created a `'month'` column in order to calculate a `'season'` column that will be used to categorize by season of the year.

A large proportion of houses do not have basements.  There may be a trend in home prices relative to basement size, specifically for homes that have a basement.  In order to classify those similarly, we have an added dummy variable which equals 1 if there is a basement.

We also create a custom binned variable `'yr_renovated_cat'` which categorizes whether the house has been renovated and whether that renovation was recent.

We have also add a dummy variable that classifies whether the house's square footage above ground is in the top 50% of the data set for square footage above ground.

After some work further on with the model, we determined that there were 4 zip codes where the average sales price was much higher than in other zipcodes.  Rather than consider every zipcode as a dummy variable, we decided to make a single variable for whether or not the property is in one of the 4 high priced zipcodes.

After running the function to create a new dataframe with these variables added, we show the information for each of the columns that have been added.


```python
df_feature_add = add_features_to_df(df_clean)

print('Added the following columns: {}'.format(", ".join(list(set(df_feature_add.columns) - set(df_clean.columns)))))
```

    Added the following columns: has_basement, startdate, month, dist_city_center, age, yr_renovated_cat, bathroom_bins, sqft_above_tophalf, zip_highprice, date_num, season, historic_home, grade_bin


## Target Scaling

Let's take a look at the distribution of prices.

The prices appear to follow a log relationship.


```python
df_feature_add.price.hist(figsize  = [16, 10]);
```


![png](images/output_30_0.png)


Let's consider evulating a log-linear relationship by adding a `'log_price'` column.


```python
df_feature_add['log_price'] = np.log(df_feature_add['price'])
```

Taking another look at the distribution of prices after taking the log reveals that we have much closer to a normal distribution of log_price.


```python
sns.distplot(df_feature_add.log_price);
```


![png](images/output_34_0.png)


## Analyze Correlation between our variables

We need evaluate the correlation matrix to determine whether any of our X variables are highly correlated, which would necessitate removing at least one of them to avoid multicollinearity problems.

To make things easier, we have a function that will find any pairs of variables in the matrix with an absolute value of correlation greater than the second parameter (default = 0.75).


```python
corr_pairs = findcorrpairs(df_feature_add, 0.7)
corr_pairs
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pairs</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>[grade, grade_bin]</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[bathroom_bins, bathrooms]</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[log_price, price]</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[sqft_above, sqft_living]</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[has_basement, sqft_basement]</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[sqft_above, sqft_above_tophalf]</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[bathrooms, sqft_living]</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[grade, sqft_living]</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[sqft_living, sqft_living15]</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[grade_bin, sqft_living]</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[grade, sqft_above]</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[grade_bin, sqft_above]</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[sqft_above, sqft_living15]</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[bathroom_bins, sqft_living]</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[sqft_lot, sqft_lot15]</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>35</th>
      <td>[age, historic_home]</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[grade_bin, sqft_living15]</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[grade, sqft_living15]</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[price, sqft_living]</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[grade, log_price]</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>42</th>
      <td>[grade_bin, log_price]</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[historic_home, yr_built]</td>
      <td>-0.71</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[age, yr_built]</td>
      <td>-1.00</td>
    </tr>
  </tbody>
</table>
</div>



# Our Initial Regression Model

## Set-up

From our intuition about the scatter plots above, along with removing a few variables that would be correlated, we take our first shot at linear regression.

Below is our list of initial X variables to try and model upon.

Some intuitive features you would see in the listing for the home include the number of bedrooms and bathrooms and the square footage of the home and property.


```python
x_list = ["bedrooms", 
          "bathrooms", 
          "sqft_above", 
          "sqft_basement", 
          "sqft_lot", 
          "sqft_living15", 
          "sqft_lot15"]
```

We first run the regression using stats model's OLS function.  Note that the dataframe of X variables must include a column of constants (1).  We have decided to use `'log_price'` as our target (Y).


```python
X = df_feature_add.loc[:, x_list]
X = sm.add_constant(X)
Y = df_feature_add['log_price']
```

## Initial Model Results

The code below fits our model and sets it equal to the variable `model_init`.  We then take a look at the results with the `.summary()` method


```python
model_init = sm.OLS(Y, X).fit()
model_init.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>log_price</td>    <th>  R-squared:         </th> <td>   0.516</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.516</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3287.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Aug 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>21:45:01</td>     <th>  Log-Likelihood:    </th> <td> -8957.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th> <td>1.793e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21589</td>      <th>  BIC:               </th> <td>1.799e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>         <td>   12.1662</td> <td>    0.011</td> <td> 1075.905</td> <td> 0.000</td> <td>   12.144</td> <td>   12.188</td>
</tr>
<tr>
  <th>bedrooms</th>      <td>   -0.0577</td> <td>    0.003</td> <td>  -16.510</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.051</td>
</tr>
<tr>
  <th>bathrooms</th>     <td>    0.0554</td> <td>    0.005</td> <td>   10.986</td> <td> 0.000</td> <td>    0.046</td> <td>    0.065</td>
</tr>
<tr>
  <th>sqft_above</th>    <td>    0.0003</td> <td> 6.01e-06</td> <td>   47.458</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>sqft_basement</th> <td>    0.0004</td> <td> 6.99e-06</td> <td>   51.438</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>sqft_lot</th>      <td> 2.408e-07</td> <td> 8.68e-08</td> <td>    2.773</td> <td> 0.006</td> <td> 7.06e-08</td> <td> 4.11e-07</td>
</tr>
<tr>
  <th>sqft_living15</th> <td>    0.0002</td> <td> 5.73e-06</td> <td>   31.259</td> <td> 0.000</td> <td>    0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>sqft_lot15</th>    <td>-1.175e-06</td> <td> 1.33e-07</td> <td>   -8.866</td> <td> 0.000</td> <td>-1.43e-06</td> <td>-9.15e-07</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>25.329</td> <th>  Durbin-Watson:     </th> <td>   1.977</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  22.437</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.031</td> <th>  Prob(JB):          </th> <td>1.34e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.855</td> <th>  Cond. No.          </th> <td>2.36e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.36e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



<font size="4">Not great...<br>
    Our $r^2$ of 0.516 isn't very good. <br><br> However, the silver lining is that our model appears to be meeting some of the assumptions necessary for linear regession.</font>


```python
residual_hist_qq(model_init)
```


![png](images/output_48_0.png)


# Refining our variable definitions

## Pruning outliers from key variables

We observed that there were some very large outliers in the fields of `'sqft_lot'` and `'sqft_lot15'`.

After experimenting with different thresholds, we settles on removing any outliers beyond the 99.9th percentile.

This allowed us to balance retaining the most information possible while also not letting outliers imbalance our model.  

The output below the code shows that 38 total outliers were removed.


```python
var_limits = {'sqft_lot': [0, .999], 
              'sqft_lot15': [0, .999]}
```


```python
df_filter = filter_df_quantiles(df_feature_add, var_limits)
```

    filtered out 0 records with sqft_lot less than: 520.00
    filtered out 22 records with sqft_lot greater than: 495972.95
    filtered out 0 records with sqft_lot15 less than: 651.00
    filtered out 16 records with sqft_lot15 greater than: 303191.60
    38 total records removed


## Data Preprocessing

### Categorical Dummy Variables

First we use our `set_to_categorical()` function to take the list of variables shown below and convert each to a category datatype in the dataframe.

We have made a custom function `create_dummyframe()` which generates a data frame of dummy variables for each of the variables listed in the second parameter from the original data frame.

Each of our categorical column groups needs at least one column removed to avoid multicollinearity issues.

### Logarithm of Selected Variables

Based on the scatterplot of possible x variables and `'log_price'` we assess that the following variables should have the log function applied in order for their relationships with `'log_price'` to be linear. 

### Scaling our Variables

To make sure no variables have any added or diminished effect simply due to the magnitude of the variable, we will standardize.

Some variables are standardized using the `MinMaxScaler()`, while others use the `StandardScaler()`.

We defaulted to using the `StandardScaler()` except for those variables that can be represented well by a specific domain.  

In this case, we thought that was appropriate for the `'date_num'`  data field because they can be thought of as positions of dates within a specific range of time.

### Applying these steps using our `preprocess_data()` function


```python
categorical_columns = ['waterfront', 'view', 'condition',
                       'yr_renovated_cat', 'season', 'grade_bin']

cat_drop_dict = {'condition': 'condition_3', 'grade_bin': 'grade_bin_7'}

log_list = ['sqft_above', 'sqft_basement',
            'sqft_living15', 'sqft_lot']


min_max_list = []


dropout_list = ['price', 'id', 'date', 'bathrooms', 'grade', 'bedrooms', 'yr_built',
                'lat', 'long', 'log_price', 'month', 'sqft_above_tophalf',
                'sqft_basement', 'startdate', 'sqft_living', 'age', 'sqft_lot15',
                'yr_renovated', 'yr_renovated_cat_missing', 'waterfront_missing', 
                'yr_renovated_cat_Prior to 2000', 'season_Summer', 'season_Winter',
                'floors', 'sqft_living15', 'date_num', 'zipcode']
```


```python
X_possible = preprocess_data(df_filter, categorical_columns=categorical_columns, log_list=log_list,
                             min_max_list=min_max_list, dropout_list=dropout_list, cat_drop_dict=cat_drop_dict)
Y = df_filter['log_price']
```

    Categorical Variables:
    waterfront          category
    view                category
    condition           category
    season              category
    yr_renovated_cat    category
    grade_bin           category
    dtype: object
    
    
    To avoid multicollinearity, the following datafields were dropped: waterfront_0.0, view_0.0, condition_3, season_Fall, yr_renovated_cat_Never Renovated, grade_bin_7
    
    
    Converted the following datafields to natural log: sqft_above, sqft_basement, sqft_living15, sqft_lot
    
    
    No variables scaled with min-max scaler
    
    
    No variables scaled with standard scaler
    
    
    Dropped price from the output dataset
    Dropped id from the output dataset
    Dropped date from the output dataset
    Dropped bathrooms from the output dataset
    Dropped grade from the output dataset
    Dropped bedrooms from the output dataset
    Dropped yr_built from the output dataset
    Dropped lat from the output dataset
    Dropped long from the output dataset
    Dropped log_price from the output dataset
    Dropped month from the output dataset
    Dropped sqft_above_tophalf from the output dataset
    Dropped sqft_basement from the output dataset
    Dropped startdate from the output dataset
    Dropped sqft_living from the output dataset
    Dropped age from the output dataset
    Dropped sqft_lot15 from the output dataset
    Dropped yr_renovated from the output dataset
    Dropped yr_renovated_cat_missing from the output dataset
    Dropped waterfront_missing from the output dataset
    Dropped yr_renovated_cat_Prior to 2000 from the output dataset
    Dropped season_Summer from the output dataset
    Dropped season_Winter from the output dataset
    Dropped floors from the output dataset
    Dropped sqft_living15 from the output dataset
    Dropped date_num from the output dataset
    Dropped zipcode from the output dataset


## Plug in our Scaled and Adjusted Variables to the Model


```python
X_model = sm.add_constant(X_possible)
```

We will split up our data so that we can train on one set of data and test it against the other set.


```python
X_train, X_test, y_train, y_test = train_test_split(X_model, Y, test_size = 0.2)
```


```python
model_smarter = sm.OLS(y_train, X_train).fit()
model_smarter.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>log_price</td>    <th>  R-squared:         </th> <td>   0.760</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.760</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2272.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Aug 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>21:45:08</td>     <th>  Log-Likelihood:    </th> <td> -1051.9</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 17247</td>      <th>  AIC:               </th> <td>   2154.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 17222</td>      <th>  BIC:               </th> <td>   2348.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    24</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                 <td>    9.3743</td> <td>    0.061</td> <td>  154.402</td> <td> 0.000</td> <td>    9.255</td> <td>    9.493</td>
</tr>
<tr>
  <th>sqft_lot</th>                              <td>    0.0685</td> <td>    0.003</td> <td>   24.876</td> <td> 0.000</td> <td>    0.063</td> <td>    0.074</td>
</tr>
<tr>
  <th>sqft_above</th>                            <td>    0.4152</td> <td>    0.009</td> <td>   44.717</td> <td> 0.000</td> <td>    0.397</td> <td>    0.433</td>
</tr>
<tr>
  <th>has_basement</th>                          <td>    0.1171</td> <td>    0.005</td> <td>   23.214</td> <td> 0.000</td> <td>    0.107</td> <td>    0.127</td>
</tr>
<tr>
  <th>zip_highprice</th>                         <td>    0.3325</td> <td>    0.010</td> <td>   32.177</td> <td> 0.000</td> <td>    0.312</td> <td>    0.353</td>
</tr>
<tr>
  <th>dist_city_center</th>                      <td>   -0.0188</td> <td>    0.000</td> <td>  -80.510</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.018</td>
</tr>
<tr>
  <th>bathroom_bins</th>                         <td>    0.0392</td> <td>    0.004</td> <td>   10.970</td> <td> 0.000</td> <td>    0.032</td> <td>    0.046</td>
</tr>
<tr>
  <th>historic_home</th>                         <td>    0.1581</td> <td>    0.007</td> <td>   24.223</td> <td> 0.000</td> <td>    0.145</td> <td>    0.171</td>
</tr>
<tr>
  <th>waterfront_1.0</th>                        <td>    0.3378</td> <td>    0.029</td> <td>   11.653</td> <td> 0.000</td> <td>    0.281</td> <td>    0.395</td>
</tr>
<tr>
  <th>view_1.0</th>                              <td>    0.1080</td> <td>    0.016</td> <td>    6.593</td> <td> 0.000</td> <td>    0.076</td> <td>    0.140</td>
</tr>
<tr>
  <th>view_2.0</th>                              <td>    0.1047</td> <td>    0.010</td> <td>   10.872</td> <td> 0.000</td> <td>    0.086</td> <td>    0.124</td>
</tr>
<tr>
  <th>view_3.0</th>                              <td>    0.1708</td> <td>    0.013</td> <td>   12.832</td> <td> 0.000</td> <td>    0.145</td> <td>    0.197</td>
</tr>
<tr>
  <th>view_4.0</th>                              <td>    0.3184</td> <td>    0.020</td> <td>   15.703</td> <td> 0.000</td> <td>    0.279</td> <td>    0.358</td>
</tr>
<tr>
  <th>condition_1</th>                           <td>   -0.2650</td> <td>    0.057</td> <td>   -4.678</td> <td> 0.000</td> <td>   -0.376</td> <td>   -0.154</td>
</tr>
<tr>
  <th>condition_2</th>                           <td>   -0.1764</td> <td>    0.023</td> <td>   -7.730</td> <td> 0.000</td> <td>   -0.221</td> <td>   -0.132</td>
</tr>
<tr>
  <th>condition_4</th>                           <td>    0.0442</td> <td>    0.005</td> <td>    9.235</td> <td> 0.000</td> <td>    0.035</td> <td>    0.054</td>
</tr>
<tr>
  <th>condition_5</th>                           <td>    0.1235</td> <td>    0.008</td> <td>   16.108</td> <td> 0.000</td> <td>    0.108</td> <td>    0.138</td>
</tr>
<tr>
  <th>yr_renovated_cat_Since 2000 inclusive</th> <td>    0.1076</td> <td>    0.015</td> <td>    7.208</td> <td> 0.000</td> <td>    0.078</td> <td>    0.137</td>
</tr>
<tr>
  <th>season_Spring</th>                         <td>    0.0494</td> <td>    0.004</td> <td>   11.533</td> <td> 0.000</td> <td>    0.041</td> <td>    0.058</td>
</tr>
<tr>
  <th>grade_bin_5</th>                           <td>   -0.2684</td> <td>    0.019</td> <td>  -14.198</td> <td> 0.000</td> <td>   -0.305</td> <td>   -0.231</td>
</tr>
<tr>
  <th>grade_bin_6</th>                           <td>   -0.1759</td> <td>    0.008</td> <td>  -22.903</td> <td> 0.000</td> <td>   -0.191</td> <td>   -0.161</td>
</tr>
<tr>
  <th>grade_bin_8</th>                           <td>    0.1602</td> <td>    0.005</td> <td>   29.535</td> <td> 0.000</td> <td>    0.150</td> <td>    0.171</td>
</tr>
<tr>
  <th>grade_bin_9</th>                           <td>    0.3391</td> <td>    0.008</td> <td>   42.369</td> <td> 0.000</td> <td>    0.323</td> <td>    0.355</td>
</tr>
<tr>
  <th>grade_bin_10</th>                          <td>    0.4725</td> <td>    0.011</td> <td>   42.145</td> <td> 0.000</td> <td>    0.451</td> <td>    0.494</td>
</tr>
<tr>
  <th>grade_bin_11</th>                          <td>    0.6406</td> <td>    0.016</td> <td>   39.479</td> <td> 0.000</td> <td>    0.609</td> <td>    0.672</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>293.123</td> <th>  Durbin-Watson:     </th> <td>   1.981</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 395.209</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.222</td>  <th>  Prob(JB):          </th> <td>1.52e-86</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.594</td>  <th>  Cond. No.          </th> <td>    759.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
residual_hist_qq(model_smarter)
```


![png](images/output_72_0.png)


<font size="4"> $r^{2}$ has improved since our initial model.  It's not too bad at about 0.9.<br><br> But we need to do a few more tests before we can feel good about it.  So far, all we know is that this model can do a decent job at predicting prices for the houses in the training data.</font> 

## Testing against our Test Data

The function below performs multiple train/test splits of the data.

Beginning with a test size of 5% of the total dataset, we increment by 5% up to 80% test data.  

The dots in blue show the mean squared error of the predicted values for the training data, while the testing data results are shown in orange.  Multiple trials are run at each percentage, with the average value being plotted.

As expected, the training data has lower mean squared error than the testing data in most cases, since that is the data that trains the model.  

As we increase the size of the testing data, the training data's error becomes lower, since the number of records in that data is decreasing.  The error in the testing data increases beause more values are being included in the testing data, while the model is becoming less intelligent as it trains on a smaller and smaller dataset.


```python
plot_mse_train_test(X_model, Y, start_test_pct=5, test_pct_inc=5, end_test_pct=82, num_iter=200)

```


![png](images/output_76_0.png)


# K-Fold Cross Validation

A related test we can use is called K-Fold Cross Validation.

K-Fold will separate our dataset into n segments containing the same number of rows.

The model will then be trained and tested n-times, with a different segment of the data being used as the testing data each time.

We would hope that the mean square errors will be small (close to zero) for each of the folds.  We also would like the errors to be similar in each fold, meaning that our model will be fairly consistent (though each fold will produce a slightly different model).

In our case, we see below that each of the 10 folds is of a similar value (about 0.03).


```python
linreg = LinearRegression()

model_kfold = KFold(n_splits=10, shuffle=True)

MSEs = -1 * cross_val_score(linreg, X_model, Y, scoring='neg_mean_squared_error', cv=model_kfold)

mean_MSE = np.mean(MSEs)


print(MSEs)
```

    [0.06556514 0.06806341 0.06669361 0.06807249 0.06250562 0.06293083
     0.06434775 0.06973193 0.06984482 0.06839193]


# Selecting our Most Significant Variables using Recursive Feature Elimination

<font size="3"> Our first function loops uses recursive feature elimination to generate models with different numbers of variables, ranging from 1 to all 34. <br> 
We then plot the values of $r^{2}$  and mean squared error against the number of variables.
</font>


```python
plot_RFE_var_iter(X_model, Y, k_fold_n_splits=5, shuffle=True, scoring='neg_mean_squared_error')
```


![png](images/output_82_0.png)



```python
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select = 24)
selector = selector.fit(X_model, Y.squeeze())
selected_X = X_model.columns[selector.support_]
print("Recursive Feature Elimination removed the following datafields: {}".format(", ".join(list(set(model_smarter.params.keys()[1:]) - set(selected_X)))))
df_selected = X_model.loc[:, selected_X]
df_selected = sm.add_constant(df_selected)

model_RFE = sm.OLS(Y, df_selected).fit()
model_RFE.summary()
```

    Recursive Feature Elimination removed the following datafields: 





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>log_price</td>    <th>  R-squared:         </th> <td>   0.760</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.760</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2846.</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 17 Aug 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>21:45:52</td>     <th>  Log-Likelihood:    </th> <td> -1360.5</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21559</td>      <th>  AIC:               </th> <td>   2771.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21534</td>      <th>  BIC:               </th> <td>   2971.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    24</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                 <td>    9.3802</td> <td>    0.054</td> <td>  172.320</td> <td> 0.000</td> <td>    9.273</td> <td>    9.487</td>
</tr>
<tr>
  <th>sqft_lot</th>                              <td>    0.0689</td> <td>    0.002</td> <td>   27.912</td> <td> 0.000</td> <td>    0.064</td> <td>    0.074</td>
</tr>
<tr>
  <th>sqft_above</th>                            <td>    0.4142</td> <td>    0.008</td> <td>   49.783</td> <td> 0.000</td> <td>    0.398</td> <td>    0.431</td>
</tr>
<tr>
  <th>has_basement</th>                          <td>    0.1195</td> <td>    0.005</td> <td>   26.457</td> <td> 0.000</td> <td>    0.111</td> <td>    0.128</td>
</tr>
<tr>
  <th>zip_highprice</th>                         <td>    0.3328</td> <td>    0.009</td> <td>   36.140</td> <td> 0.000</td> <td>    0.315</td> <td>    0.351</td>
</tr>
<tr>
  <th>dist_city_center</th>                      <td>   -0.0188</td> <td>    0.000</td> <td>  -90.028</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.018</td>
</tr>
<tr>
  <th>bathroom_bins</th>                         <td>    0.0382</td> <td>    0.003</td> <td>   11.965</td> <td> 0.000</td> <td>    0.032</td> <td>    0.044</td>
</tr>
<tr>
  <th>historic_home</th>                         <td>    0.1507</td> <td>    0.006</td> <td>   25.801</td> <td> 0.000</td> <td>    0.139</td> <td>    0.162</td>
</tr>
<tr>
  <th>waterfront_1.0</th>                        <td>    0.3384</td> <td>    0.026</td> <td>   12.896</td> <td> 0.000</td> <td>    0.287</td> <td>    0.390</td>
</tr>
<tr>
  <th>view_1.0</th>                              <td>    0.1146</td> <td>    0.014</td> <td>    7.929</td> <td> 0.000</td> <td>    0.086</td> <td>    0.143</td>
</tr>
<tr>
  <th>view_2.0</th>                              <td>    0.1074</td> <td>    0.009</td> <td>   12.287</td> <td> 0.000</td> <td>    0.090</td> <td>    0.125</td>
</tr>
<tr>
  <th>view_3.0</th>                              <td>    0.1731</td> <td>    0.012</td> <td>   14.488</td> <td> 0.000</td> <td>    0.150</td> <td>    0.197</td>
</tr>
<tr>
  <th>view_4.0</th>                              <td>    0.3172</td> <td>    0.018</td> <td>   17.525</td> <td> 0.000</td> <td>    0.282</td> <td>    0.353</td>
</tr>
<tr>
  <th>condition_1</th>                           <td>   -0.2570</td> <td>    0.048</td> <td>   -5.311</td> <td> 0.000</td> <td>   -0.352</td> <td>   -0.162</td>
</tr>
<tr>
  <th>condition_2</th>                           <td>   -0.1809</td> <td>    0.020</td> <td>   -8.938</td> <td> 0.000</td> <td>   -0.221</td> <td>   -0.141</td>
</tr>
<tr>
  <th>condition_4</th>                           <td>    0.0449</td> <td>    0.004</td> <td>   10.461</td> <td> 0.000</td> <td>    0.036</td> <td>    0.053</td>
</tr>
<tr>
  <th>condition_5</th>                           <td>    0.1232</td> <td>    0.007</td> <td>   17.927</td> <td> 0.000</td> <td>    0.110</td> <td>    0.137</td>
</tr>
<tr>
  <th>yr_renovated_cat_Since 2000 inclusive</th> <td>    0.1044</td> <td>    0.014</td> <td>    7.680</td> <td> 0.000</td> <td>    0.078</td> <td>    0.131</td>
</tr>
<tr>
  <th>season_Spring</th>                         <td>    0.0444</td> <td>    0.004</td> <td>   11.582</td> <td> 0.000</td> <td>    0.037</td> <td>    0.052</td>
</tr>
<tr>
  <th>grade_bin_5</th>                           <td>   -0.2548</td> <td>    0.017</td> <td>  -15.177</td> <td> 0.000</td> <td>   -0.288</td> <td>   -0.222</td>
</tr>
<tr>
  <th>grade_bin_6</th>                           <td>   -0.1817</td> <td>    0.007</td> <td>  -26.440</td> <td> 0.000</td> <td>   -0.195</td> <td>   -0.168</td>
</tr>
<tr>
  <th>grade_bin_8</th>                           <td>    0.1616</td> <td>    0.005</td> <td>   33.289</td> <td> 0.000</td> <td>    0.152</td> <td>    0.171</td>
</tr>
<tr>
  <th>grade_bin_9</th>                           <td>    0.3419</td> <td>    0.007</td> <td>   47.694</td> <td> 0.000</td> <td>    0.328</td> <td>    0.356</td>
</tr>
<tr>
  <th>grade_bin_10</th>                          <td>    0.4752</td> <td>    0.010</td> <td>   47.177</td> <td> 0.000</td> <td>    0.455</td> <td>    0.495</td>
</tr>
<tr>
  <th>grade_bin_11</th>                          <td>    0.6412</td> <td>    0.015</td> <td>   44.216</td> <td> 0.000</td> <td>    0.613</td> <td>    0.670</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>345.538</td> <th>  Durbin-Watson:     </th> <td>   1.994</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 484.746</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-0.200</td>  <th>  Prob(JB):          </th> <td>5.48e-106</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.616</td>  <th>  Cond. No.          </th> <td>    759.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Once again, we want to check our distribution of errors and the comparison of the actual quantiles versus theoretical in the Q-Q plot


```python
residual_hist_qq(model_RFE)
```


![png](images/output_85_0.png)


Let's take a look at some of the continuous variables and their impacts on the predicted prices.

From the residual plots in the top right corner, we can get a good feel for whether our errors are well distributed around 0.


```python
continuous_var = ['sqft_above', 'sqft_lot', 'dist_city_center']

for col in continuous_var:
    fig = plt.figure(figsize=(15,8))
    fig = sm.graphics.plot_regress_exog(model_RFE, col, fig=fig)
    plt.show()
```


![png](images/output_87_0.png)



![png](images/output_87_1.png)



![png](images/output_87_2.png)


Our model fitted to the target of `'log_price'`.  In order to compare how closely we matched to the actual price, we will need to take the exponential of both the actual `Y['log_price']` and the predicted values of `'log_price'`.

The graphs below shows how the actual prices compare to the predicted prices in our data.  We would hope the reuslts would be relatively close to the line drawn by y=x.

We can also graph our residuals, which is essentially a shift of the first graph so that the line y=x essentially becomes the x-axis.


```python
y_hat_exp = np.exp(model_RFE.predict(df_selected))
y_exp = np.exp(Y.squeeze())

plt.figure(figsize=(10,10))
sns.scatterplot(y_exp, y_hat_exp, alpha=0.05)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
limiter = 2000000
plt.xlim(0, limiter)
plt.ylim(0, limiter)
plt.title('Predicted Price versus Actual')
sns.lineplot(x=[0, limiter], y=[0, limiter]);
plt.show()


plt.figure(figsize=(10,10))
sns.scatterplot(y_exp, y_exp - y_hat_exp, alpha=0.05)
plt.xlabel('Actual Price')
plt.ylabel('Residual')
plt.xlim(0,limiter)
plt.ylim(-500000,500000)
plt.title('Residual Plot versus Actual Price')
sns.lineplot(x=[0,6000000], y=[0, 0]);
```


![png](images/output_90_0.png)



![png](images/output_90_1.png)



```python
y_limit = 10000000
y_exp_filter = y_exp[y_exp < y_limit]
y_hat_exp_filter = y_hat_exp[y_exp < y_limit]


print('**For prices under ${:,}**:'.format(y_limit))
print('Average Absolute Error: ${0:,.0f}'.format(np.mean(abs(y_exp_filter - y_hat_exp_filter))))
print('Average Percent Error: {0:.2f}%'.format(100 * np.mean(abs(y_exp_filter - y_hat_exp_filter)) / y_exp_filter.mean()))
print('Average of Absolute Error for each: {0:.2f}%'.format(100*np.mean(abs(y_exp_filter - y_hat_exp_filter)/y_exp_filter)))
```

    **For prices under $10,000,000**:
    Average Absolute Error: $104,988
    Average Percent Error: 19.44%
    Average of Absolute Error for each: 21.00%


# Interpretation


Because of our variable manipulation, our coefficients need some adjustment in order to glean some useful information.

The adjusted coefficient takes care of the fact that our price has had the natural log applied.  We also adjusted the coefficients for variables that were also logged.


```python
pd.set_option('display.max_colwidth', -1)
df_Interpretation = interpret_coef(model_RFE, df_filter, log_list=log_list, min_max_list=min_max_list, categorical_columns=categorical_columns)
df_Interpretation.iloc[(-(df_Interpretation['adj_coef'].abs())).argsort()]
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>coef</th>
      <th>adj_coef</th>
      <th>interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>grade_bin_11</td>
      <td>0.641155</td>
      <td>89.867283</td>
      <td>Price is expected to increase by 89.8673% for each unit increment in grade_bin_11</td>
    </tr>
    <tr>
      <th>22</th>
      <td>grade_bin_10</td>
      <td>0.475237</td>
      <td>60.839465</td>
      <td>Price is expected to increase by 60.8395% for each unit increment in grade_bin_10</td>
    </tr>
    <tr>
      <th>21</th>
      <td>grade_bin_9</td>
      <td>0.341875</td>
      <td>40.758386</td>
      <td>Price is expected to increase by 40.7584% for each unit increment in grade_bin_9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>waterfront_1.0</td>
      <td>0.338358</td>
      <td>40.264260</td>
      <td>Price is expected to increase by 40.2643% if condition met for waterfront_1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>zip_highprice</td>
      <td>0.332802</td>
      <td>39.487132</td>
      <td>Price is expected to increase by 39.4871% for each unit increment in zip_highprice</td>
    </tr>
    <tr>
      <th>11</th>
      <td>view_4.0</td>
      <td>0.317185</td>
      <td>37.325656</td>
      <td>Price is expected to increase by 37.3257% if condition met for view_4.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>condition_1</td>
      <td>-0.257045</td>
      <td>-22.666652</td>
      <td>Price is expected to increase by -22.6667% if condition met for condition_1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>grade_bin_5</td>
      <td>-0.254802</td>
      <td>-22.493006</td>
      <td>Price is expected to increase by -22.4930% for each unit increment in grade_bin_5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>view_3.0</td>
      <td>0.173125</td>
      <td>18.901416</td>
      <td>Price is expected to increase by 18.9014% if condition met for view_3.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>grade_bin_8</td>
      <td>0.161610</td>
      <td>17.540193</td>
      <td>Price is expected to increase by 17.5402% for each unit increment in grade_bin_8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>grade_bin_6</td>
      <td>-0.181733</td>
      <td>-16.617641</td>
      <td>Price is expected to increase by -16.6176% for each unit increment in grade_bin_6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>condition_2</td>
      <td>-0.180937</td>
      <td>-16.551172</td>
      <td>Price is expected to increase by -16.5512% if condition met for condition_2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>historic_home</td>
      <td>0.150706</td>
      <td>16.265508</td>
      <td>Price is expected to increase by 16.2655% for each unit increment in historic_home</td>
    </tr>
    <tr>
      <th>15</th>
      <td>condition_5</td>
      <td>0.123224</td>
      <td>13.113720</td>
      <td>Price is expected to increase by 13.1137% if condition met for condition_5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>has_basement</td>
      <td>0.119484</td>
      <td>12.691476</td>
      <td>Price is expected to increase by 12.6915% for each unit increment in has_basement</td>
    </tr>
    <tr>
      <th>8</th>
      <td>view_1.0</td>
      <td>0.114634</td>
      <td>12.146295</td>
      <td>Price is expected to increase by 12.1463% if condition met for view_1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>view_2.0</td>
      <td>0.107435</td>
      <td>11.341902</td>
      <td>Price is expected to increase by 11.3419% if condition met for view_2.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>yr_renovated_cat_Since 2000 inclusive</td>
      <td>0.104394</td>
      <td>11.003752</td>
      <td>Price is expected to increase by 11.0038% for each unit increment in yr_renovated_cat_Since 2000 inclusive</td>
    </tr>
    <tr>
      <th>14</th>
      <td>condition_4</td>
      <td>0.044873</td>
      <td>4.589505</td>
      <td>Price is expected to increase by 4.5895% if condition met for condition_4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>season_Spring</td>
      <td>0.044366</td>
      <td>4.536492</td>
      <td>Price is expected to increase by 4.5365% if condition met for season_Spring</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bathroom_bins</td>
      <td>0.038189</td>
      <td>3.892719</td>
      <td>Price is expected to increase by 3.8927% for each unit increment in bathroom_bins</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dist_city_center</td>
      <td>-0.018837</td>
      <td>-1.866106</td>
      <td>Price is expected to increase by -1.8661% for each unit increment in dist_city_center</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sqft_above</td>
      <td>0.414234</td>
      <td>0.414234</td>
      <td>Price is expected to increase by 0.4142% for each 1% increase in sqft_above</td>
    </tr>
    <tr>
      <th>0</th>
      <td>sqft_lot</td>
      <td>0.068950</td>
      <td>0.068950</td>
      <td>Price is expected to increase by 0.0689% for each 1% increase in sqft_lot</td>
    </tr>
  </tbody>
</table>



