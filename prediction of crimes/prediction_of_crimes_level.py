#!/usr/bin/env python
# coding: utf-8

# # Задача 1: Baseline model

# На сайте https://russia.duck.consulting/ можно найти множество статистических данных по регионам России.
# 
# Все нижеиспользуемые датасеты были взяты оттуда.

# In[1008]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn import metrics

from pandas_profiling import ProfileReport


# In[1009]:


# Импортируем датасет, содержащий информацию об экономических преступлениях
# crimes - количество преступлений на 10.000 жителей

df_ec_crimes = pd.read_csv('ec_crimes.csv')
df_ec_crimes.columns = ['year', 'region', 'crimes']


# In[1010]:


# Импортируем датасет с информацией об уровне безработицы
# unemployments - безработица в процентах

df_unemp = pd.read_csv('unemployments.csv')
df_unemp.columns = ['year', 'region', 'unemployments']


# In[1011]:


# Импортируем датасет с информацией о среднем заработке россиян

df_avg_salary = pd.read_csv('avg_salary.csv')
df_avg_salary.columns = ['year', 'region', 'avg_salary']


# In[1012]:


# Соединяем вышеперечисленные датасеты в один по региону и году

df = pd.merge(df_ec_crimes, df_avg_salary, on=['region', 'year'])


# In[1013]:


df = pd.merge(df, df_unemp, on=['region', 'year'])


# In[1014]:


# Удаляем строки с NaN

df = df.dropna()


# In[1015]:


# Итоговый датасет

df


# In[1016]:


TRAIN_PERCENT = int(df.shape[0]*0.66) #указывает размер обучающей выборки
df_train = df[:TRAIN_PERCENT]
df_test = df[TRAIN_PERCENT:]


# In[1017]:


# Выводим статистическую информацию о данных

profile = ProfileReport(df, title='Pandas Profiling Report')
profile


# На основе информации о корреляциии между стобцами (вкладка 'Correlations' в ячейке выше), можно сделать вывод, что на качество обучения больше всего будет влиять информация о среднем заработке россиян (самая высокая корреляция м/у crimes и avg_salary).

# In[1018]:


features_orig = ['unemployments', 'avg_salary']

features = features_orig


# In[1019]:


# Разделяем датасет на обучающую и тестовую выборку

X_train = df_train[features]
y_train = df_train['crimes']

X_test = df_test[features]
y_test = df_test['crimes']


# In[1020]:


def get_regression_metrics(model, actual, predicted):
    
    """Calculate main regression metrics.
    
    Args:
        model (str): The model name identifier
        actual (series): Contains the test label values
        predicted (series): Contains the predicted values
        
    Returns:
        dataframe: The combined metrics in single dataframe
    
    
    """
    regr_metrics = {
                        'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,
                        'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),
                   }

    #return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
    df_regr_metrics.columns = [model]
    return df_regr_metrics


# In[1021]:


def plot_features_weights(model, weights, feature_names, weights_type='c'):
    
    """Plot regression coefficients weights or feature importance.
    
    Args:
        model (str): The model name identifier
        weights (array): Contains the regression coefficients weights or feature importance
        feature_names (list): Contains the corresponding features names
        weights_type (str): 'c' for 'coefficients weights', otherwise is 'feature importance'
        
    Returns:
        plot of either regression coefficients weights or feature importance
        
    
    """
    (px, py) = (8, 10) if len(weights) > 30 else (8, 5)
    W = pd.DataFrame({'Weights':weights}, feature_names)
    W.sort_values(by='Weights', ascending=True).plot(kind='barh', color='r', figsize=(px,py))
    label = ' Coefficients' if weights_type =='c' else ' Features Importance'
    plt.xlabel(model + label)
    plt.gca().legend_ = None


# In[1022]:


def plot_residual(model, y_train, y_train_pred, y_test, y_test_pred):
    
    """Print the regression residuals.
    
    Args:
        model (str): The model name identifier
        y_train (series): The training labels
        y_train_pred (series): Predictions on training data
        y_test (series): The test labels
        y_test_pred (series): Predictions on test data
        
    Returns:
        Plot of regression residuals
    
    """
    
    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-50, xmax=400, color='red', lw=2)
    plt.title(model + ' Residuals')
    plt.show()


# In[1023]:


# Линейная регрессия

linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

y_test_predict = linreg.predict(X_test)
y_train_predict = linreg.predict(X_train)

print('MAE training: %.3f, MAE test: %.3f' % (
      (metrics.mean_absolute_error(y_train, y_train_predict)), 
      (metrics.mean_absolute_error(y_test, y_test_predict))))

linreg_metrics_orig_features = get_regression_metrics('Linear Regression (Baseline model)', y_test, y_test_predict)
linreg_metrics_orig_features


# In[1024]:


plot_features_weights('Linear Regression', linreg.coef_, X_train.columns, 'c')


# In[1025]:


plot_residual('Linear Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[1026]:


# Полиномиальная регрессия

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


polyreg = linear_model.LinearRegression()
polyreg.fit(X_train_poly, y_train)

y_test_predict = polyreg.predict(X_test_poly)
y_train_predict = polyreg.predict(X_train_poly)

print('MAE training: %.3f, MAE test: %.3f' % (
      (metrics.mean_absolute_error(y_train, y_train_predict)), 
      (metrics.mean_absolute_error(y_test, y_test_predict))))

polyreg_metrics_orig_features = get_regression_metrics('Polynomial Regression (Baseline model)', y_test, y_test_predict)
polyreg_metrics_orig_features


# In[1027]:


plot_residual('Polynomial Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[1028]:


# Сравним реальные и предсказанные значения (Плиномиальная регрессия)

pr_pred_dict = {
                'Actual' : y_test,
                'Prediction' : y_test_predict
            }
    
pr_pred = pd.DataFrame.from_dict(pr_pred_dict).T
pr_pred


# In[1029]:


# Сравним итоговые результаты

reg_metrics_bfe = pd.concat([linreg_metrics_orig_features, polyreg_metrics_orig_features], axis=1)
reg_metrics_bfe


# # Задача 2: User model

# За целевую переменную я так же взял уровень преступности,
# добавил  независимые переменные:
# - интегрированная оценка качества жизни
# - процент бедного населения
# 

# In[1030]:


# Датасет с интегрированной оценкой качества жизни населения

df_q = pd.read_csv('life_quality.csv')
df_q.columns = ['year', 'region', 'life_quality']


# In[1031]:


# Датасет с процентом бедного населения

df_p = pd.read_csv('percentage_of_poor.csv')
df_p.columns = ['year', 'region', 'poor']


# In[1032]:


# Соединяем датасеты

df = pd.merge(df, df_q, on=['region', 'year'])


# In[1033]:


df = pd.merge(df, df_p, on=['region', 'year'])


# In[1034]:


df = df.dropna()


# In[1035]:


df


# In[1036]:


# Выводим статистическую информацию о данных

profile = ProfileReport(df, title='Pandas Profiling Report')
profile


# In[1037]:


TRAIN_PERCENT = int(df.shape[0]*0.66) #указывает размер обучающей выборки
df_train = df[:TRAIN_PERCENT]
df_test = df[TRAIN_PERCENT:]


# In[1038]:


columns_for_analysis = ['life_quality', 'poor', 'crimes']

clmns = columns_for_analysis


# In[1039]:


# remove outliers

df_train = df_train[clmns][df_train[clmns].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
df_test = df_test[clmns][df_test[clmns].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]


# In[1040]:


features_orig = ['unemployments', 'avg_salary']

features_another = ['life_quality', 'poor']

features = features_another


# In[1041]:


# Разделяем датасет на обучающую и тестовую выборку

X_train = df_train[features]
y_train = df_train['crimes']

X_test = df_test[features]
y_test = df_test['crimes']


# In[1042]:


# Линейная регрессия

linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

y_test_predict = linreg.predict(X_test)
y_train_predict = linreg.predict(X_train)

print('MAE training: %.3f, MAE test: %.3f' % (
      (metrics.mean_absolute_error(y_train, y_train_predict)), 
      (metrics.mean_absolute_error(y_test, y_test_predict))))

linreg_metrics_new_features = get_regression_metrics('Linear Regression (User Model)', y_test, y_test_predict)
linreg_metrics_new_features


# In[1043]:


# Полиномиальная регрессия

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


polyreg = linear_model.LinearRegression()
polyreg.fit(X_train_poly, y_train)

y_test_predict = polyreg.predict(X_test_poly)
y_train_predict = polyreg.predict(X_train_poly)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))

polyreg_metrics_new_features = get_regression_metrics('Polynomial Regression (User Model)', y_test, y_test_predict)
polyreg_metrics_new_features


# In[1044]:


# Random Forest

rf = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=1)

rf.fit(X_train, y_train)

y_test_predict = rf.predict(X_test)
y_train_predict = rf.predict(X_train)

print('MAE training: %.3f, MAE test: %.3f' % (
      (metrics.mean_absolute_error(y_train, y_train_predict)), 
      (metrics.mean_absolute_error(y_test, y_test_predict))))

rf_metrics = get_regression_metrics('Random Forest Regression (User Model)', y_test, y_test_predict)
rf_metrics


# In[1045]:


plot_features_weights('Random Forest Regressor', rf.feature_importances_, X_train.columns, 't' )


# In[1046]:


plot_residual('Random Forest Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[1047]:


# Сравним реальные и предсказанные значения

rf_pred_dict = {
                'Actual' : y_test,
                'Prediction' : y_test_predict
            }
    
rf_pred = pd.DataFrame.from_dict(rf_pred_dict).T
rf_pred


# In[1048]:


# Также я поробовал использовать XGBoost (до я с ним не работал), варьируя гиперпараметры, получилось добиться приемлимого результата

xgdmat=xgb.DMatrix(X_train,y_train)
our_params={'learning_rate':0.2, 'subsample':0.8, 'max_depth':6}
final_gb=xgb.train(our_params,xgdmat)
tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb.predict(tesdmat)


# In[1049]:


xg_metrics = get_regression_metrics('XGBoost (User Model)', y_test, y_pred)
xg_metrics


# In[1050]:


xg_pred_dict = {
                'Actual' : y_test,
                'Prediction' : y_pred
            }
    
xg_pred = pd.DataFrame.from_dict(xg_pred_dict).T
xg_pred


# In[1051]:


# Сравним итоговые результаты

reg_metrics_bfe = pd.concat([linreg_metrics_orig_features, polyreg_metrics_orig_features, linreg_metrics_new_features, polyreg_metrics_new_features, rf_metrics, xg_metrics], axis=1)
reg_metrics_bfe


# Выводы: Лучше всего предсказывает уровень преступности модель XGBoost, используя данные об интгрированной оценке уровня жизни и информацию об уровни бедности. Сравнимые результаты показала модель Random Forest.

# In[ ]:




