import os
import datetime
import pandas as pd
import requests
import logging 
import sys
from components import extract_send
from components import process
from components import assignment
from components import model
from components import plot_img as pltimg
from components import add_data
from tee import Tee



#Create paths to find the data and the results
MAIN_PATH = os.path.dirname(__file__)
FILE_PATH = os.path.join(MAIN_PATH,'data','FraudTest.csv')
exc_date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
RESULTS_PATH = os.path.join(MAIN_PATH, 'data', 'ResultFraudTest_{}.xlsx'.format(exc_date))
PREDICTIONS_PATH = os.path.join(MAIN_PATH, 'data', 'PredictionsFraudTest_{}.csv'.format(exc_date))

if __name__ == '__main__':
    # Read the data from the csv file and create a dataframe
    dfFraudTest = extract_send.read_data(FILE_PATH,FILE_PATH.split('.')[-1])
    # Merchant transactions of the categories (column 'category'), that have less than 15000 records.
    dfFraudMod = process.less_records(dfFraudTest)
    # Extract exchange rates from the API and create a new column with the exchange rate all/COP.
    request = requests.get('https://api.exchangerate.host/latest')
    dfFraudMod = process.amt_cop(dfFraudMod,request)
    # Calculate a new column with the first 6 numbers from the 'cc_num' column and name it 'bin'.
    dfFraudMod = process.bin(dfFraudMod)
    # Calculate in a new column in the same table the count of unique card numbers per 'bin' and per 'merchant' in the entire database.
    dfFraudMod = process.calculation(dfFraudMod)
    # Calculate how many ZIP codes were registered for each city.
    dfFraudMod = process.zip(dfFraudMod)
    # Calculate the number of transactions that each payer (first and last name) made at the same time and on the same day as the date of each transaction.
    dfFraudMod = process.full_name(dfFraudMod)
    dfFraudMod = process.trans_person(dfFraudMod)
    # Assign the value 1 in the 'is_fraud' column when the values calculated in the three previous points are above the average. Otherwise, leave the values that are already in that variable.
    dfFraudMod = assignment.is_fraud(dfFraudMod)
    #Send the data to the results file
    # extract_send.send_data(dfFraudMod,RESULTS_PATH)
    # Show distribution of the target variable
    pltimg.distribution_classes(dfFraudMod)
    # Calculate metrics for the manual traning of the model
    metric = model.manual_train_estimator(dfFraudMod)
    pltimg.plot_metrics(metric)
    # Get the tuned parameters for the model
    tuned_params = model.get_tuned_params(dfFraudMod)
    # Get the best model, metrics, save and register it in MLflow
    exp_name = "LightGBM_autolog_Fraud_Model"
    model.mlflow_model(exp_name,dfFraudMod,tuned_params,MAIN_PATH)
    # Change the stage of the model
    model.change_stage(exp_name,1,'Production')
    # Make predictions of 1000 data with the model
    predictions = model.predict(dfFraudMod,exp_name,1)
    df_predictions = pd.DataFrame(predictions,columns=['is_fraud'])
    extract_send.save_data(df_predictions,PREDICTIONS_PATH)



        
    

    


        


                    




            
    




    
    
    
    
    