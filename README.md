I have been assigned to a project that focuses on the use of influencer marketing. For this task, you will explore the relationship between your radio promotion budget and your sales.

The dataset provided includes information about marketing campaigns across TV, radio, and social media, as well as how much revenue in sales was generated from these campaigns. Based on this information, company leaders will make decisions about where to focus future marketing resources. Therefore, it is critical to provide them with a clear understanding of the relationship between types of marketing campaigns and the revenue generated as a result of this investment.


```python
# imports library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
import seaborn as sns
from  statsmodels.formula.api import ols
import statsmodels.api as sm
```


```python
#load dataset
data = pd.read_csv("marketing_sales_data.csv")
```

The dataset provided is a .csv file (named marketing_sales_data.csv), which contains information about marketing conducted in collaboration with influencers, along with corresponding sales.


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Social Media</th>
      <th>Influencer</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>1.218354</td>
      <td>1.270444</td>
      <td>Micro</td>
      <td>90.054222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Medium</td>
      <td>14.949791</td>
      <td>0.274451</td>
      <td>Macro</td>
      <td>222.741668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Low</td>
      <td>10.377258</td>
      <td>0.061984</td>
      <td>Mega</td>
      <td>102.774790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>High</td>
      <td>26.469274</td>
      <td>7.070945</td>
      <td>Micro</td>
      <td>328.239378</td>
    </tr>
    <tr>
      <th>4</th>
      <td>High</td>
      <td>36.876302</td>
      <td>7.618605</td>
      <td>Mega</td>
      <td>351.807328</td>
    </tr>
    <tr>
      <th>5</th>
      <td>High</td>
      <td>25.561910</td>
      <td>5.459718</td>
      <td>Micro</td>
      <td>261.966812</td>
    </tr>
    <tr>
      <th>6</th>
      <td>High</td>
      <td>37.263819</td>
      <td>6.886535</td>
      <td>Nano</td>
      <td>349.861575</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Low</td>
      <td>13.187256</td>
      <td>2.766352</td>
      <td>Macro</td>
      <td>140.415286</td>
    </tr>
    <tr>
      <th>8</th>
      <td>High</td>
      <td>29.520170</td>
      <td>2.333157</td>
      <td>Nano</td>
      <td>264.592233</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Low</td>
      <td>3.773287</td>
      <td>0.135074</td>
      <td>Nano</td>
      <td>55.674214</td>
    </tr>
  </tbody>
</table>
</div>



i observe about the different variables included in the data.
The data includes the following information:
- TV promotion budget (expressed as "Low", "Medium", or "High")
- Radio promotion budget
- Social media promotion budget
- Type of influencer that the promotion is in collaboration with (expressed as "Mega", "Macro", or "Micro", or "Nano")
- 
**Note**: Mega-influencers have over 1 million followers, macro-influencers have 100,000 to 1 million followers, micro-influencers have 10,000 to 100,000 followers, and nano-influencers have fewer than 10,000 followers.
Sales accrued from the promotion


```python
# dataset shape 
data.shape
```




    (572, 5)




```python
#checking null 
data.isna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Social Media</th>
      <th>Influencer</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>567</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>568</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>569</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>570</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>571</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>572 rows × 5 columns</p>
</div>




```python
data.isna().sum()
```




    TV              1
    Radio           1
    Social Media    0
    Influencer      0
    Sales           1
    dtype: int64




```python
#boolean indication along with columns
data.any(axis = 1)
```




    0      True
    1      True
    2      True
    3      True
    4      True
           ... 
    567    True
    568    True
    569    True
    570    True
    571    True
    Length: 572, dtype: bool




```python
#drop missing values

data.dropna(axis = 0).reset_index(inplace = True, drop = True)
```


```python
# Start with .isna() to get booleans indicating whether each value in the data is missing.

data.isna().any(axis = 1).sum()
```




    np.int64(3)




```python
# drop the three missing value to prepare data for the modeling
data = data.dropna(axis = 0)
```

# Model Assumptions
- The next step for this task is checking model assumptions. To explore the relationship between radio promotion budget and sales, model the relationship using linear regression. Begin by confirming whether the model assumptions for linear regression can be made in this context.

**Note**: Some of the assumptions can be addressed before the model is built. These will be addressed in this section. After the model is built, you will finish checking the assumptions.

- Create a plot of pairwise relationships in the data. This will help you visualize the relationships and check model assumptions.


```python
sns.pairplot(data)
```









    

    


In the scatterplot **Sales** over **Radio**. All the points are clusters around the line which indicates a positive association between two variables.so the first lineaity assumption is met.

# Model building



```python
#next we will separate the import metrics from the dataset.

ols_data = data[['Radio', 'Sales']]
ols_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Radio</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.218354</td>
      <td>90.054222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.949791</td>
      <td>222.741668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.377258</td>
      <td>102.774790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.469274</td>
      <td>328.239378</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.876302</td>
      <td>351.807328</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25.561910</td>
      <td>261.966812</td>
    </tr>
    <tr>
      <th>6</th>
      <td>37.263819</td>
      <td>349.861575</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.187256</td>
      <td>140.415286</td>
    </tr>
    <tr>
      <th>8</th>
      <td>29.520170</td>
      <td>264.592233</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.773287</td>
      <td>55.674214</td>
    </tr>
  </tbody>
</table>
</div>




```python
# write a ols formula 
ols_formula = "Sales ~ Radio"

```

# Implement the Ordinary Least Squares (OLS) approach for linear regression.


```python
OLS = ols(formula = ols_formula, data = ols_data)
```


```python
# fit the model to the dat

model = OLS.fit()
```


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Sales</td>      <th>  R-squared:         </th> <td>   0.757</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.757</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1768.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 02 Oct 2024</td> <th>  Prob (F-statistic):</th> <td>2.07e-176</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:02:04</td>     <th>  Log-Likelihood:    </th> <td> -2966.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   569</td>      <th>  AIC:               </th> <td>   5937.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   567</td>      <th>  BIC:               </th> <td>   5946.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   41.5326</td> <td>    4.067</td> <td>   10.211</td> <td> 0.000</td> <td>   33.544</td> <td>   49.521</td>
</tr>
<tr>
  <th>Radio</th>     <td>    8.1733</td> <td>    0.194</td> <td>   42.048</td> <td> 0.000</td> <td>    7.791</td> <td>    8.555</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.267</td> <th>  Durbin-Watson:     </th> <td>   1.880</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.322</td> <th>  Jarque-Bera (JB):  </th> <td>   2.221</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.102</td> <th>  Prob(JB):          </th> <td>   0.329</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.772</td> <th>  Cond. No.          </th> <td>    45.7</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The coefficient here being the y-intercept and the slope "m" 
- The y-intercept is 41.5326
- The slope is 8.1733

The linear equation to express the relationship between  sales and radio promotion budget in the form of y= slope*x+y-intercept


The slope in this context mean 
- one interpretation: if a company spend a 1 million dollars more for promoting their products/services on the radio, the company's sales would be increase by 8.1733 millions dollars an average



```python
#finish model assumption 

sns.regplot(x = "Radio", y = "Sales", data = ols_data)
```




    <Axes: xlabel='Radio', ylabel='Sales'>




    
![png](output_24_1.png)
    


The linear relationship between two varibles along with the best fit line. this confrim the linearity


```python
# Residual
residuals = model.resid
```


```python
# visualize the distribution of residual
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of residuals")
plt.show()
```


    
![png](output_27_0.png)
    


The distribution of residuals is normal. This indicates that the assumption of normality is likely met.


```python
#Create a qqplot
sm.qqplot(residuals, line = 's')
plt.title("Q-Q plot of residual")
plt.show()
```


    
![png](output_29_0.png)
    


# Check the assumptions of independent observation and homoscedasticity.
getting the fitted value from the model


```python
fitted_values = model.predict(ols_data["Radio"])
```


```python
# Create a scatterplot of the residuals against the fitted values.
fig = sns.scatterplot(x = fitted_values, y = residuals)
fig.axhline(0)
fig.set_xlabel("fitted values")
fig.set_ylabel("Residuals")
plt.show()
```


    
![png](output_32_0.png)
    


In the preceding scatterplot, the data points have a cloud-like resemblance and do not follow an explicit pattern. So it appears that the independent observation assumption has not been violated. Given that the residuals appear to be randomly spaced, the homoscedasticity assumption seems to be met.

# Conclusion
What are the key takeaways?

- Data visualizations and exploratory data analysis can be used to check if linear regression is a well suited approach for modeling the relationship between two variables.
The results of a linear regression model can be used to express the relationship between two variables.
What results can be presented from this.

- In the simple linear regression model, the y-intercept is 41.5326 and the slope is 8.1733. One interpretation: If a company has a budget of 1 million dollars more for promoting their products/services on the radio, the company's sales would increase by 8.1733 million dollars on average. Another interpretation: Companies with 1 million dollars more in their radio promotion budget accrue 8.1733 million dollars more in sales on average.

- The results are statistically significant with a p-value of 0.000, which is a very small value (and smaller than the common significance level of 0.05). This indicates that there is a very low probability of observing data as extreme or more extreme than this dataset when the null hypothesis is true.

- In this context, the null hypothesis is that there is no relationship between radio promotion budget and sales i.e. the slope is zero, and the alternative hypothesis is that there is a relationship between radio promotion budget and sales i.e. the slope is not zero. So, we  reject the null hypothesis and state that there is a relationship between radio promotion budget and sales for companies in this data.

- The slope of the line of best fit that resulted from the regression model is approximate and subject to uncertainty (not the exact value). The 95% confidence interval for the slope is from 7.791 to 8.555. This indicates that there is a 95% probability that the interval [7.791, 8.555] contains the true value for the slope.

**How would we frame  findings to external stakeholders?**

- Based on the dataset at hand and the regression analysis conducted here, there is a notable relationship between radio promotion budget and sales for companies in this data, with a p-value of 0.000 and a standard error of 0.194. For companies represented by this data, a 1 million dollar increase in radio promotion budget could be associated with an 8.1733 million dollar increase in sales. It would be worth continuing to promote products/services on the radio. Also, it is recommended that the relationship between the two variables (radio promotion budget and sales) be further examined in different contexts. For example, it would help to gather more data to understand whether this relationship is different in certain industries or when promoting certain types of products/services.


```python

```
