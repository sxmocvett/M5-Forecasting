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


To solve this problem a binary classification problem is considered. As seen in Figure 1, the target variable classes are unbalanced.

<p align="center">
  <img src=pictures/target_balance.png?raw=true "Target Class Balance" width="250" height="370">
</p>

*<p align="center">Fig. 1 Target class balance (0 – Not looking for job change, 1 – Looking for a job change)</p>* 
                
<p align="justify">Also some features have from 0% to 40% missing values (Figure 2), that imposes additional conditions to prepare data and tune models.</p>

<p align="center">
  <img src=pictures/missing_values.PNG?raw=true "% Missing values">
</p> 

*<p align="center">Fig. 2 Missing values (%) </p>*

### Models
<p align="justify">Models are used in the project: CatBoostClassifier, RandomForestClassifier and VotingClassifier (based on LGBMClassifier, KNNClassifier and LogisticRegression). Taking into account the above-described issues, in addition to tuning the main hyperparameters of models, various combinations of imputation values (SimpleImputer and KNNImputer) and data recovery (SMOTE) strategies are also considered. Automation of the parameters selection for these operations is achieved by GridSearchCV using ColumnTransformer and Pipeline constructors.</p>

### Metrics
<p align="justify">Base metric for making choice of the best parameters for each of the models in conditions of class imbalance is "balanced accuracy score". The following metrics are also used for the final comparison of models (Figure 3).</p>

<p align="center">
  <img src=pictures/metrics.PNG?raw=true "Metrics" width="500" height="200">
</p>

*<p align="center">Fig. 3 Сomparative metrics </p>* 
