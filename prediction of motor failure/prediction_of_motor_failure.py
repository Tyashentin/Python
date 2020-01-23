#!/usr/bin/env python
# coding: utf-8

# Отслеживание и предсказание отказа оборудования помогает избежать внезапных ситуаций, спрогнозировать их заранее. При помощи прогноза можно осуществить плановый ремонт оборудования заранее (до поломки) и сэкономить ресурсы.

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn import metrics

from pandas_profiling import ProfileReport


# In[4]:





# In[3]:


df = pd.read_csv('data.csv')
df


# Исходя из описания задания:
# - id - идентификатор мотора,
# - cycle - цикл работы,
# - p00-p20 - показания датчиков считываемые в процессе работы,
# - s0, s1 - настройки изменяемые в конце каждого цикла.
#     
# Решить задачу прогнозирования отказа моторов можно при помощи различных методов: 
# - регрессии,
# - бинарной классификации,
# - мультиклассовой классификации.
# 
# Я решил сделать это при помощи регрессии.
# Нам необходимо предсказывать число циклов работы мотора, которые могут успешно пройти до появления критических неисправностей.
# 
# Имея данные, которые у нас есть можно вычислить для каждого мотора в отдельный момент времени (номер цикла) количество циклов, которое ещё мотор проработает.

# In[270]:


# Добавление столбца, которой показывает количество циклов до поломки

df['cycles_until_failure'] = df.groupby(['id'])['cycle'].transform(max)-df['cycle']


# In[271]:


# Разделяем обучающую выборку и тестовую выборку в соотношении (70/30)
df_train = df[:14500]
df_test = df[14500:]


# In[272]:


pd.DataFrame.describe(df)


# In[247]:


# Выводим статистическую информацию о данных
profile = ProfileReport(df, title='Pandas Profiling Report')


# In[248]:


profile


# In[249]:


features_orig = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 's1', 's2']

# Берём признаки, у которых высокая корелляция с номером цикла работы мотора

features_for_using = ['p02', 'p03', 'p04', 'p05', 'p06', 'p08', 'p11', 'p12', 'p13', 'p14', 'p15', 'p18', 'p19', 'p20']

features = features_for_using


# In[250]:


X_train = df_train[features]
y_train = df_train['cycles_until_failure']

X_test = df_test[features]
y_test = df_test['cycles_until_failure']


# In[251]:


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
                        'R^2' : metrics.r2_score(actual, predicted),
                        'Explained Variance' : metrics.explained_variance_score(actual, predicted)
                   }

    #return reg_metrics
    df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
    df_regr_metrics.columns = [model]
    return df_regr_metrics


# In[252]:


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


# In[253]:


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


# In[254]:


# Линейная регрессия

linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

y_test_predict = linreg.predict(X_test)
y_train_predict = linreg.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))

linreg_metrics = get_regression_metrics('Linear Regression', y_test, y_test_predict)
linreg_metrics


# In[255]:


plot_features_weights('Linear Regression', linreg.coef_, X_train.columns, 'c')


# In[256]:


plot_residual('Linear Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[257]:


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

polyreg_metrics = get_regression_metrics('Polynomial Regression', y_test, y_test_predict)
polyreg_metrics


# In[258]:


plot_residual('Polynomial Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[259]:


# Decision Tree regressor

dtrg = DecisionTreeRegressor(max_depth=7, max_features=12, random_state=123) # selected features
#dtrg = DecisionTreeRegressor(max_depth=7, random_state=123)
dtrg.fit(X_train, y_train)

y_test_predict = dtrg.predict(X_test)
y_train_predict = dtrg.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))

dtrg_metrics = get_regression_metrics('Decision Tree Regression', y_test, y_test_predict)
dtrg_metrics


# In[260]:


plot_features_weights('Decision Tree Regressor', dtrg.feature_importances_, X_train.columns, 't' )


# In[261]:


plot_residual('Decision Tree Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[262]:


# Random Forest

rf = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=7, n_jobs=-1, random_state=1)

rf.fit(X_train, y_train)

y_test_predict = rf.predict(X_test)
y_train_predict = rf.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))

rf_metrics = get_regression_metrics('Random Forest Regression', y_test, y_test_predict)
rf_metrics


# In[263]:


plot_features_weights('Random Forest Regressor', rf.feature_importances_, X_train.columns, 't' )


# In[264]:


plot_residual('Random Forest Regression', y_train_predict, y_train, y_test_predict, y_test)


# In[265]:


# Сравним реальные и предсказанные значения

rf_pred_dict = {
                'Actual' : y_test,
                'Prediction' : y_test_predict
            }
    
rf_pred = pd.DataFrame.from_dict(rf_pred_dict).T
rf_pred


# In[266]:


# Также я поробовал использовать XGBoost (до я с ним не работал), варьируя гиперпараметры, получилось добиться приемлимого результата

xgdmat=xgb.DMatrix(X_train,y_train)
our_params={'learning_rate':0.2, 'subsample':0.8, 'max_depth':8}
final_gb=xgb.train(our_params,xgdmat)
tesdmat=xgb.DMatrix(X_test)
y_pred=final_gb.predict(tesdmat)


# In[267]:


xg_metrics = get_regression_metrics('XGBoost', y_test, y_pred)
xg_metrics


# In[268]:


xg_pred_dict = {
                'Actual' : y_test,
                'Prediction' : y_pred
            }
    
xg_pred = pd.DataFrame.from_dict(xg_pred_dict).T
xg_pred


# In[269]:


# Сравним итоговые результаты

reg_metrics_bfe = pd.concat([linreg_metrics, dtrg_metrics, polyreg_metrics, rf_metrics, xg_metrics], axis=1)
reg_metrics_bfe


# Наилучшие результаты показали модели Random Forest и XGBoost.

# Доп. задание: Задачу можно было усложнить, предложив реализовать мультиклассовую классификацию отказа моторов.
#     На каждой итерации работы мотора его состояние необходимо соотнести с одним из трёх (опционально) классов:
# 0. исправно работающий мотор
# 1. мотор, которому в скором времени понадобится ремонт
# 2. мотор, который скоро выйдет из строя.

# In[ ]:




