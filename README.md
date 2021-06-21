# <p align="center">M5-Forecasting</p> 

<p align="center">
  <img src=https://img.golos.io/proxy/http://lk.aldmi.ru/wp-content/uploads/2016/04/Divider_03-1.png width="600" height="80">
</p>

## Problem formulation

<p align="justify"> You will use hierarchical sales data from Walmart (the world’s largest company by revenue) to forecast daily sales. </p>

The original assignment is at [link](https://www.kaggle.com/c/m5-forecasting-accuracy/overview)

## Dataset
<p align="justify"> The data covers stores in three US States (California, Texas, and Wisconsin). It includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events.</p>

## Solution description

As an example of a forecasting problem in this task we will predict future sales for the state of Wisconsin. As seen in Figure 1, sales value is time series data.

<p align="center">
  <img src=pictures/WI_sales.png "Sales value for Wisconsin" width="250" height="370">
</p>

*<p align="center">Fig. 1 Sales value for Wisconsin</p>* 
                
<p align="justify">Also the time series is not stationary and has a pronounced seasonality (Figure 2), that imposes additional conditions to prepare data and tune models.</p>

<p align="center">
  <img src=pictures/stationarity_test.PNG "% Stationarity test">
</p> 

*<p align="center">Fig. 2 Stationarity test</p>*

### Models
<p align="justify">There are two ways to consider the problem: predict time series values or solve the classic regression problem. Models for the first type: ARIMAX, seasonal ARIMAX, Triple HWES and AutoRegressive model. For the second type: CatBoost and StackingRegressor (where base models are LinearSVR and Ridge, meta model is RandomForestRegressor). Data for them are features, which are generated using TSFRESH library.  
</p>

### Metrics
<p align="justify">For the final comparison of models the following metrics are used (Figure 3).</p>

<p align="center">
  <img src=pictures/metrics.PNG "Metrics" width="500" height="200">
</p>

*<p align="center">Fig. 3 Сomparative metrics </p>* 
