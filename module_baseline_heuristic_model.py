"""
Created on Wed Mar 10 00:54:21 2021

The purpose of this module is to:
* apply a heuristic/rule-based logic for coupon assignment 
* get top 5 coupons for every shopper 


The Heuristic "Model"
Assumptions:
(1) A customer will continue buying products with a coupon, if she has done so previously. 
(2) Products which were bought only once (with a coupon) will not be bought again. 

Logic/Algorithm:
1. Reserve week 89 data for the test set
2. Calculate the rate an item is bought with a coupon for each customer
[dropped] 3. Filter out items which are bought less often than twice with a coupon
[dropped] 4. Filter out items which are bought less often than the average item for that user
5. Order the items descending by coupon redemption rate
6. Return the top 5 products
7. Assign a coupon to these products for the week t+1, here week 90
"""


# Libraries
import pandas as pd
import numpy as np

# generate dataset for heuristic model   
def generate_heuristic_data(path_datasets):
    baskets = pd.read_parquet(path_datasets + "/baskets.parquet")
    baskets = baskets[baskets.shopper <2000]
    coupons = pd.read_parquet(path_datasets + "/coupons.parquet")
    coupons = coupons[coupons.shopper <2000]
    #Merging coupon data to basket data
    bc = pd.merge(baskets, coupons, on=['week','shopper','product'], how='left')
    # set missing discounts to no discount (0)
    bc['discount'] = bc['discount'].replace(np.nan, 0)
    # create needed features for logic 
    bc['discounted'] = np.where(bc['discount']>0,1,0)
    bc['dis_prod_share'] = bc.groupby(['shopper','product'])['discounted'].transform('mean')
    bc['product_discount_buys'] = bc.groupby(['shopper', 'product'])['discounted'].transform('sum')
    bc['product_buys'] = bc.groupby(['shopper', 'product'])['price'].transform('count')
    bc['buys'] = bc.groupby(['shopper'])['price'].transform('count')
    bc['unique_products'] = bc.groupby(['shopper'])['product'].transform('nunique')
    bc['discount_buys'] = bc.groupby(['shopper'])['discounted'].transform('mean')
    # filter out products which were bought less than twice and less frequently than average 
    #bc_filtered = bc[(bc['product_discount_buys']>1)] #& (bc['product_buys']>(bc['buys']/bc['unique_products']))]
    
    # group on shopper and product 
    bc_filtered_grouped = bc.groupby(['shopper', 'product']).mean().reset_index(drop=False)
    # return dataset 
    return bc_filtered_grouped


def heuristic_model(path_datasets, bc_filtered_grouped):
    """
    input: 
        bc_filtered_grouped: dataset for heuristic model 
    output: 
        top5coupons: coupon recommendation dataset
    """    
    # finding top-k (here, 5) best products of a shopper by product coupon redemption rate
    topk = 5
    heuristic_df = bc_filtered_grouped.groupby(['shopper']).apply(lambda x: x.nlargest(topk,['dis_prod_share'])).reset_index(drop=True)
    
    # assinging coupons 
    heuristic_df['coupon'] = heuristic_df.groupby("shopper").cumcount()
    # read idx file 
    heur_idx = pd.read_parquet(path_datasets + "/coupon_index.parquet")
    # merge files
    heur_idx = pd.merge(heur_idx, heuristic_df[['shopper','coupon','product']], on=['shopper', 'coupon', 'product'], how='left')
    # mapping discounts
    discount_dict = {
         0: 30,
         1: 25,
         2: 20,
         3: 15,
         4: 15
    }
    heur_idx['discount'] = heur_idx['coupon'].map(discount_dict)
    
    heur_idx = heur_idx[['shopper', 'week', 'coupon', 'product','discount']]
    # unit test 
    assert heur_idx.shape[0] == 10000 #10.000 = 2000*5
    assert heur_idx.shape[1] == 5 #shopper, week, coupon, product, discount
    assert heur_idx.isna().sum().sum() == 0
    assert list(heur_idx.columns) == ['shopper', 'week', 'coupon', 'product', 'discount']
    assert max(heur_idx['coupon']) == 4 #did this change to 4
    assert min(heur_idx['coupon']) == 0 #did this change to 0
    assert heur_idx['shopper'].nunique() == 2000

    # returning output dataset
    return heur_idx

