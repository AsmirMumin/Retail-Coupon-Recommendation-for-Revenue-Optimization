import pandas as pd
import numpy as np
import pathlib

def merge_dataset(path, name_left_df, name_right_df, on, how):
    
    'Load Data Sets'
    
    format_left_df = pathlib.Path(name_left_df).suffix
    
    if format_left_df == '.parquet':
        left_df = pd.read_parquet(path + '/' + name_left_df)
    elif format_left_df == '.csv':
        left_df = pd.read_csv(path + '/' + name_left_df)
    else:
        print('Function only reads parquet and csv file. Please adapt format')
    
    format_right_df = pathlib.Path(name_right_df).suffix
    
    if format_right_df == '.parquet':
        right_df = pd.read_parquet(path + '/' + name_right_df)
    elif format_right_df == '.csv':
        right_df = pd.read_csv(path + '/' + name_right_df)
    else:
        print('Function only reads parquet and csv file. Please adapt format')
        
    'Merge Data Sets'
    
    data = pd.merge(left_df, right_df, on = ['week', 'shopper', 'product'], how = 'outer')
    
    return (data)