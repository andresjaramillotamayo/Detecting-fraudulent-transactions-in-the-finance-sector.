"""_summary_
Module to define functions to process the data
"""

import pandas as pd

def less_records(df):
    df_15k = df.groupby('category').filter(lambda x: len(x) < 15000)
    return df_15k

def bin(df):
    df['bin']= df['cc_num'].astype(str).str[:6]
    return df

def calculation(df):
    df['calculation'] = df.groupby(['bin','merchant'])['cc_num'].transform('nunique')
    return df 

def zip(df):
    df['count_zip'] = df.groupby(['city'])['zip'].transform('count')
    return df

def full_name(df):
    df['full_name'] = df['first'] + ' ' + df['last']
    return df

def trans_person(df):
    df['count_trans_person']= df.groupby(['trans_date_trans_time','full_name'])['cc_num'].transform('count')
    return df

def amt_cop(df,request):
    try:
        amt_cop=[]
        for index,row in df.iterrows():
            if row['currency'] == 'COP': 
                amt_cop.append(round(row['amt'],2))
            else:
                amt_cop.append(round(row['amt']*(1/request.json()['rates'][row['currency']])*request.json()['rates']['COP'],2))
        df['amt_cop'] = amt_cop
    except Exception as e:
        print("Error",e)
        print('Not possible to calculate the value in COP')
    return df
