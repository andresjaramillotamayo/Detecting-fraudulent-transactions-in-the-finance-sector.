"""_summary_
Module to define the function to assign the value of 1 if the transaction is fraudulent
"""

import pandas as pd

def is_fraud(df):
    try:
        for index,row in df.iterrows():
            if row['calculation'] > df['calculation'].mean() and row['count_zip'] > df['count_zip'].mean() and row['count_trans_person'] > df['count_trans_person'].mean():
                row['is_fraud'] = 1
            else:
                row['is_fraud']
    except Exception as e:
        print("Error",e)
        print('Not possible to assign the value')
    return df