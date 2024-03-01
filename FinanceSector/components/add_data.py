import pandas as pd
"""_summary_
Module to add some data to be presented
"""

def group_merchant_data(df):
    df = df.groupby('merchant').agg({'trans_num':'count','amt_cop':'sum','is_fraud':'sum','cc_num':'nunique'})
    return df

def calc_total_fraud_cop(df):
    df['total_fraud_amt_cop']=df.groupby('merchant').apply(lambda x: x[x['is_fraud'] == 1]['amt_cop'].sum())
    return df

def perc_fraud_trans(df):
    df['perc_fraud_trans']=round((df['is_fraud']/df['trans_num'])*100,2)
    return df

def perc_fraud_total_amt(df):
    df['perc_fraud_total']=round((df['total_fraud_amt_cop']/df['amt_cop'])*100,2)
    return df
def rename_df_col(df):
    df.rename(columns={'amt_cop':'total_amt_cop','is_fraud':'count_is_fraud','cc_num':'countDis_cc_num'},inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df