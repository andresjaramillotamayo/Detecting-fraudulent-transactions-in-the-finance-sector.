"""_summary_
Module to extract and send the data to be executed
"""

import pandas as pd
import xlsxwriter

# Read the data from the file_path
def read_data(file_path,file_type):
    try:
        if file_type == 'csv':
            print('Reading the csv file')
            df = pd.read_csv(file_path,sep=';')
        else:
            print('Reading the xlsx file')
            df = pd.read_excel(file_path)
    except Exception as e:
        print("Error",e)
        print('Error reading the file')
    return df

# Send the result data
def send_data(df,file_path):
    columns = ['cc_num','trans_num','merchant','category','first','last','full_name','gender','city','state','zip','job','currency','amt','amt_cop','bin','calculation','count_zip','count_trans_person','is_fraud'] 
    try:
        writer = pd.ExcelWriter(file_path,engine='xlsxwriter')
        df_result = df[columns].to_excel(writer,index=False)
        writer.close()
        print('Result file created')
    except Exception as e:
        print("Error",e)
        print('Result file not created')
    return df_result

# Save the data in a csv file
def save_data(df,file_path):
    try:
        df.to_csv(file_path,index=False,sep=';')
        print('Prediction saved')
    except Exception as e:
        print("Error",e)
        print('Prediction not saved')
    return df