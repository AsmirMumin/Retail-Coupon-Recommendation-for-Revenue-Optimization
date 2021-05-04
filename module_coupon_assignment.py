"""
Created on Wed Mar 10 00:54:21 2021

The purpose of this module is to:
* load a trained model
* make predictions under various discounts d e {15,20,25,30}
* get top 5 coupons for every shopper based on expected revenue
"""


# Libraries
import pandas as pd
import numpy as np
import time
import lightgbm as lgbm 
import pickle

import module_week90_generate_dataset
from module_week90_generate_dataset import week90_generate_dataset 


# load the trained LightGBM model 
    # example of a filename = 'lightgbm_model.pkl'

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model 

# get week 90 testing data 
# X_test_90 = week90_generate_dataset(path_datasets)


# get week 90 testing data to predict on  
#def get_test_data():
 #   X_test_90 = week90_generate_dataset(path_datasets) 
 #   X_test_90 = X_test_90.drop(['product_bought', 'shopper', 'product'], axis = 1)
    
 #   return X_test_90


# create prediction datasets 

def get_test_discounts(X_test_90):
    """
    generate test data sets for various discounts d e {15,20,25,30}
    """
    # test data for discount = 15
    X_test_d15 = X_test_90.copy()
    X_test_d15['discount'] = 15 
    X_test_d15['price'] = X_test_d15['max_price'] * 0.85
    
    # test data for discount = 20 
    X_test_d20 = X_test_90.copy()
    X_test_d20['discount'] = 20
    X_test_d20['price'] = X_test_d20['max_price'] * 0.80
    
    # test data for discount = 25
    X_test_d25 = X_test_90.copy()
    X_test_d25['discount'] = 25
    X_test_d25['price'] = X_test_d25['max_price'] * 0.75
    
    # test data for discount = 30 
    X_test_d30 = X_test_90.copy()
    X_test_d30['discount'] = 30
    X_test_d30['price'] = X_test_d30['max_price'] * 0.70
    
    # returning discount test data sets 
    return X_test_d15, X_test_d20, X_test_d25, X_test_d30
    

# create predictions using the trained LightGBM model 
def predictions(model, X_test_d15, X_test_d20, X_test_d25, X_test_d30):
    pred_d15 = model.predict_proba(X_test_d15.values)
    pred_d20 = model.predict_proba(X_test_d20.values)
    pred_d25 = model.predict_proba(X_test_d25.values)
    pred_d30 = model.predict_proba(X_test_d30.values)
    return pred_d15, pred_d20, pred_d25, pred_d30

def merge_predictions(X_test_90, X_test_d15, X_test_d20, X_test_d25, X_test_d30, pred_d15, pred_d20, pred_d25, pred_d30):
    """
    data structure: 
        week [optional]
        shopper
        product
        price
        discount
        proba (y_hat)
    """
    # add product and shopper
    X_test_d15['product'] = X_test_90['product']
    X_test_d15['shopper'] = X_test_90['shopper']
    # add product and shopper
    X_test_d20['product'] = X_test_90['product']
    X_test_d20['shopper'] = X_test_90['shopper']
    # add product and shopper
    X_test_d25['product'] = X_test_90['product']
    X_test_d25['shopper'] = X_test_90['shopper']
    # add product and shopper
    X_test_d30['product'] = X_test_90['product']
    X_test_d30['shopper'] = X_test_90['shopper']
    
    # add predicitons to respective test data sets 
    X_test_d15['proba'] = pred_d15[::,1]
    X_test_d20['proba'] = pred_d20[::,1]
    X_test_d25['proba'] = pred_d25[::,1]
    X_test_d30['proba'] = pred_d30[::,1]
    
    return X_test_d15, X_test_d20, X_test_d25, X_test_d30

def coupon_assignment(X_test_d15, X_test_d20, X_test_d25, X_test_d30):
    """
    input: 
        X_test_d15: prediction dataset for d = 15
        X_test_d20: prediction dataset for d = 20
        X_test_d25: prediction dataset for d = 25
        X_test_d30: prediction dataset for d = 30
    output: 
        top5coupons: coupon recommendation dataset
    """
    # concat the various prediction files 
    preds = pd.concat([X_test_d15,X_test_d20,X_test_d25,X_test_d30], axis=0)
    
    # subset the data to only required columns
    preds = preds[['week', 'shopper', 'product', 'price', 'discount', 'proba']]
    
    # calculate expected revenue
    # adjusted price * purchase probability|coupon d e {15,20,25,30}
    preds['e_revenue'] = preds['price'] * preds['proba']
    
    # finding top-k (here, 1) discount on a product-shopper pair 
    topdiscount = 1
    top1discount = preds.groupby(['shopper', 'product']).apply(lambda x: x.nlargest(topdiscount, ['e_revenue'])).reset_index(drop=True)
    
    # finding top-k (here, 5) best products of a shopper by expected revenue
    topcoupons = 5
    top5coupons = top1discount.groupby(['shopper']).apply(lambda x: x.nlargest(topcoupons, ['e_revenue'])).reset_index(drop=True)
    
    # assinging coupons 
    top5coupons['coupon'] = top5coupons.groupby("shopper").cumcount()
    
    #resorting the final dataframe
    top5coupons_final = top5coupons[['shopper', 'week', 'coupon', 'product', 'discount']]
    
    # unit test 
    #assert top5coupons_final.shape[0] == 10000 #10.000 = 2000*5
    assert top5coupons_final.shape[1] == 5 #shopper, week, coupon, product, discount
    assert top5coupons_final.isna().sum().sum() == 0
    assert list(top5coupons_final.columns) == ['shopper', 'week', 'coupon', 'product', 'discount']
    assert max(top5coupons_final['coupon']) == 4 #or 5?????
    assert min(top5coupons_final['coupon']) == 0 #or 1?????
    #assert top5coupons_final['shopper'].nunique() == 2000

    # returning output dataset
    return top5coupons_final

