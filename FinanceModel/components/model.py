"""_summary_
Module to develop the Fraud Model 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,  accuracy_score, precision_score, recall_score, f1_score 
from sklearn import preprocessing
from collections import Counter
from verstack import LGBMTuner
import pickle
import os
import mlflow
from mlflow.tracking import MlflowClient
from components import plot_img as pltimg

# Function to extract only the columns that will be used in the model
def get_variables(df):
    columns = ['calculation','count_zip','count_trans_person','is_fraud'] 
    df = df[columns]
    # Create the independent and dependent variables
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1] 
    return X, y

# Function to calculate the scale_pos_weight (class_weight parameter in algorithm)
def calc_pos_weight(y):
    counter=Counter(y)
    print(counter)
    scale_pos_weight = counter[0] / counter[1]
    print('scale_pos_weight: %.3f' % scale_pos_weight)
    return scale_pos_weight

# Function to create the training and test sets
def make_train_test_split(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify=y, random_state=101)
    print(f'X train set size: {X_train.shape};'f'\nX test set size: {X_test.shape};', f'\ny train set size: {y_train.shape};' f'\ny test set size: {y_test.shape};')
    return X_train, X_test, y_train, y_test

# Function to calcule the metrics to be used
def eval_metrics(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score_f1 = f1_score(y_test, y_pred)
    return acc, precision, recall, score_f1, mse, mae

# Function to report metrics
def report(params,acc,precision,recall,score_f1,mse,mae):
    print(f'Best parameters: {params}')
    print(f'Accuracy of LightGBM classifier is: {acc}')
    print(f'Precision Score of LightGBM classifier is: {precision}')
    print(f'Recall Score of LightGBM classifier is: {recall}')
    print(f'F1 Score of LightGBM classifier is: {score_f1}')
    print(f'mse: {mse}')
    print(f'mae: {mae}')
    
# Function to save the model
def save_best_estimator(estimator, path= 'models',pFile = 'fraud_model.pickle'):
    if not os.path.exists('models'):
        os.makedirs('models')
    with open(os.path.join(path, pFile), 'wb') as file:
        pickle.dump(estimator, file)
        
# Function to load the model      
def load_best_estimator(path='models', pFile='estimator.pickle'):
    if not os.path.exists(path):
        return None
    with open(os.path.join(path, pFile), "rb") as file:
        estimator = pickle.load(file)
    return estimator

# Function to do the manual training of the data, to extract some parameters to be compared
def manual_train_estimator(df):
    mlflow.lightgbm.autolog(disable=True)
    n_leaves = np.linspace(20, 300, 20)
    r_alpha = np.linspace(0.0001, 0.5, 20)
    r_lambda = np.linspace(0.0001, 0.5, 20)
    X, y = get_variables(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X,y)
    scale_pos_weight = calc_pos_weight(y)
    # Create the model
    lst = []
    for i,k,j in zip(n_leaves, r_alpha, r_lambda):
        model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, num_leaves=int(i),reg_alpha= k,reg_lambda = j,boosting_type='gbdt',n_jobs=-1,random_state=101)
        # Fitting the model to the training data
        model.fit(X_train, y_train)
        # Making predictions
        y_pred = model.predict(X_test) 
        params = model.get_params()
        # Model evaluation
        acc, precision, recall, score_f1, mse, mae = eval_metrics(y_test,y_pred)
        lst.append([acc,precision,recall,score_f1,mse,mae])
    return lst

# Fuction to tune the hyperparameters of the LightGBM algorithm
def get_tuned_params(df):
    mlflow.lightgbm.autolog(disable=True)
    X, y = get_variables(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X,y)
    scale_pos_weight = calc_pos_weight(y)
    
    # Create the model
    tuner=LGBMTuner(scale_pos_weight=scale_pos_weight,metric='f1', trials=1000)
    
    # Fitting the model to the training data
    tuner.fit(X_train, y_train)
    
    # Making predictions
    y_pred = tuner.predict(X_test) 
    
    # Model evaluation
    acc, precision, recall, score_f1, mse, mae = eval_metrics(y_test,y_pred)
    report(tuner.best_params,acc,precision,recall,score_f1,mse,mae)
    
    return tuner.best_params

# Fuction to train the LightGBM algorithm with the best hyperparameters
def best_train_estimator(df,tuner_model):
    X, y = get_variables(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X,y)
    scale_pos_weight = calc_pos_weight(y)
    
    # Create the model
    params = tuner_model
    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight,boosting_type='gbdt',n_jobs=-1, **params)
    
    # Fitting the model to the training data
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model.predict(X_test) 
    
    # Model evaluation
    acc, precision, recall, score_f1, mse, mae = eval_metrics(y_test,y_pred)
    report(model.get_params(),acc,precision,recall,score_f1,mse,mae)
    
    # Plot graphs
    pltimg.precision_recall_curve_plot(model,X_test,y_test)
    pltimg.roc_curve_plot(model,X_test,y_test)
    
    # Save model results
    save_best_estimator(model)
    model.booster_.save_model('models/fraud_model.txt')
    model.booster_.save_model('models/fraud_model.json')
    
    return y_test,y_pred,model, acc, precision, recall, score_f1, mse, mae

def mlflow_model(exp_name,df,tuned_params,main_path):
    client = MlflowClient()
    mlflow.set_experiment(exp_name)
    exp = client.get_experiment_by_name(exp_name)
    # Absolute 
    img_path = os.path.join(main_path, 'images')
    # Start a new run in the experiment and track 
    with mlflow.start_run(experiment_id=exp.experiment_id):
        #Set model description
        mlflow.set_tag('mlflow.note.content',"Model to predict fraud transactions")   
        mlflow.lightgbm.autolog(registered_model_name=exp_name) 
        y_test,y_pred,model,acc,precision,recall,score_f1,mse,mae = best_train_estimator(df,tuned_params)
        mlflow.log_metric("Accuracy",acc)
        mlflow.log_metric("Precision",precision)
        mlflow.log_metric("Recall",recall)
        mlflow.log_metric("F1 Score",score_f1)
        mlflow.log_metric("MSE",mse)
        mlflow.log_metric("MAE",mae)
        pltimg.confusion_matrix_plot(y_test,y_pred)
        mlflow.log_artifact(img_path)
        autolog_run= mlflow.active_run()
        autolog_run_id = mlflow.last_active_run().info.run_id
        # mlflow.register_model(f"runs:/{autolog_run_id}",exp_name)
    mlflow.end_run()
    return autolog_run_id

# Function to change the stage of the model
def change_stage(exp_name,version,stage):
    client = MlflowClient()
    client.update_model_version(name=exp_name,version=version  ,description="Model to predict fraud transactions")
    client.transition_model_version_stage(
            name=exp_name,
            version=version,
            #  Staging|Archived|Production|None
            stage=stage,     
        )
    
# Fuction to make predictions using the model
def predict(df,exp_name,version):
    X, y = get_variables(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X,y)

    model_name = exp_name 
    model_version = version

    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    return model.predict(X_test[0:1000])

       