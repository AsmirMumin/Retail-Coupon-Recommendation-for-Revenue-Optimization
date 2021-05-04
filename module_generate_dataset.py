"""
The purpose of this module is to:
* generate datasets that can be used for the training and testing the model

Note: The split in train and test set before feature engineering is essential since bias in the features are reduced. For instance, the later training set only contains characteristics of the data set that was subsetted for training purposes. We aim to make the model more generizable with this.

Prerequisite: 
* The following datasets should be downloaded/generated and named: 
    * download: baskets.parquet, coupons.parquet
    * create: df_negative_samples.parquet, product_categories.csv, avg_no_weeks_between_two_purchases.parquet, lags.parquet, purchase_temporal_distribution.parquet

* Table of Content:
    Train+Test
    * Load Data Sets
    * Merge Data Sets
    * Feature Engineering Part I
    * Unit Test Block I
    * Train-Test-Split
    * Clear Memory
    Train
    * Feature Engineering Part II.a
    * Clear Memory
    * Feature Engineering Part III.a
    * Imputing/Fixing Missing Values
    * Unit Test Block II.a
    * Store data_train
    Test
    * Feature Engineering Part II.b
    * Clear Memory
    * Feature Engineering Part III.b
    * Imputing/Fixing Missing Values
    * Unit Test Block II.b
    * Store data_test
    Train+Test
    * Unit Test Block III
    
"""

import pandas as pd
import numpy as np

def generate_dataset(path, train_start, train_end, test_start, test_end):
    
    """
    input: 
        path: path where data sets are stored -> outputted data sets are saved as train_s2000_final.parquet and test_s2000_final.parquet in the pwd
        train_start: first week that the training set should start with
        train_end: the last week the training set should end with
        test_start: analogue to train
        test_end: analogue to train
    
    output: 
        data_train: engineered training set
        data_test: engineered testing set
     
    """
    
    print('The dataframes should be named: \nbaskets.parquet, \ncoupons.parquet, \ndf_negative_samples.parquet, \nproduct_categories.csv, \navg_no_weeks_between_two_purchases.parquet, \nlags.parquet and \npurchase_temporal_distribution.parquet')
    
    'Load Data Sets'
    #customers past purchase (week 0-89, shopper, product, price in € cents)
    basket_df = pd.read_parquet(path + '/baskets.parquet')
    #coupons customers received in the past (week, shopper, product, discount in %)
    coupon_df = pd.read_parquet(path + '/coupons.parquet')
    #negative samples generate randomly from the customer preference with the length i of the corresponding week j
    negative_sample_df = pd.read_parquet(path + '/df_negative_samples.parquet')
    #categories among products that were created with TSNE
    categories = pd.read_csv(path + '/product_categories.csv', sep = ',')
    #average number of week that passed between two purchases
    avg_no_weeks_between_two_purchases = pd.read_parquet(path + '/avg_no_weeks_between_two_purchases.parquet')
    #time that has passed since the shopper i bought product j the last time
    lags = pd.read_parquet(path + '/lags.parquet')
    lags = lags.drop('product_bought', axis=1)
    lags = lags[lags['week'] <= 89]
    #distribution of purchases, e.g whether they occurs frequently or rather in the first/last half of the timeseries
    purchase_temporal_distribution = pd.read_parquet(path + '/purchase_temporal_distribution.parquet')
    
    #shrink dataframe to only the 2000 shopper that we need to make predictions for
    basket_df = basket_df[(basket_df['shopper'] < 2000)]
    coupon_df = coupon_df[(coupon_df['shopper'] < 2000)]
    
    'Merge Data Sets'
    data = pd.merge(basket_df, coupon_df, on=['week', 'shopper', 'product'], how='outer')
    data = pd.merge(data, negative_sample_df, on=['week', 'shopper', 'product'], how='outer')
    data = pd.merge(data, categories, on=['product'], how='left')
    data = pd.merge(data, avg_no_weeks_between_two_purchases, on=['shopper', 'product'], how='left')
    data = pd.merge(data, lags, on=['shopper', 'product', 'week'], how='left')
    data = pd.merge(data, purchase_temporal_distribution, on=['shopper', 'product'], how='left')
    
    'Feature Engineering Part I'
    #missing values are imputed with -1 (=never bought) since there missing not differentiates from the pure absense (0)
    data['avg_no_weeks_between_two_purchases'] = data['avg_no_weeks_between_two_purchases'].replace(np.nan, -1)
    data['purchase_temporal_distribution'] = data['purchase_temporal_distribution'].replace(np.nan, -1)
    data['lag_weeks_of_product_per_customer'] = data['lag_weeks_of_product_per_customer'].replace(np.nan, -1)
    #replace NaN value of the col discount with 0 aka no discount
    data['discount'] = data['discount'].replace(np.nan, 0)
    #for now replace missing prices with 0, later we will adjust them by the price with the corresponding discount
    data['price'] = data['price'].replace(np.nan, 0)
    #discount offered to the shopper
    data['discount_offered'] = np.where(data['discount'] != 0, 1, 0)
    #product purchased
    data['product_bought'] = np.where(data['price'] != 0, 1, 0)
    #purchase without having a discount
    data['purchase_w/o_dis'] = np.where(((data['product_bought'] == 1) & (data['discount_offered'] == 0)), 1, 0)
    #no purchase even though a discount was offered
    data['no_purchase_w_dis'] = np.where(((data['product_bought'] == 0) & (data['discount_offered'] == 1)), 1, 0)
    #discount effect --> either neutral/negative (if shopper would have bought the item anyways, eventually market lost revenue) or positive 
    data['discount_effect'] = np.where(((data.discount_offered == 1) & (data.product_bought == 1)), 1, 0)
       
    'Unit Test Block I'    
    assert data['shopper'].nunique() == 2000
    assert data['discount_offered'].nunique() == 2
    assert data['product_bought'].nunique() == 2
    assert data['purchase_w/o_dis'].nunique() == 2
    assert data['no_purchase_w_dis'].nunique() == 2
    assert data['discount_effect'].nunique() == 2
    assert data.isna().sum().sum() == 0
    
    'Train-Test-Split'
    data_train = data[((data['week'] >= train_start) & (data['week'] <= train_end))]
    data_test = data[((data['week'] >= test_start) & (data['week'] <= test_end))]
    
    'Clear Memory'
    del coupon_df
    del negative_sample_df
    del categories
    del data
    
    'Feature Engineering Part II.a'
    #maximal price of product
    max_price = data_train.groupby('product')['price'].agg(max).reset_index()
    max_price = max_price.rename(columns = {'price': 'max_price'})
    #merge max price to the df
    data_train = pd.merge(data_train, max_price, on='product', how='left')
    #impute missing prices by the max price minues the offered discount (because this was the price the shoppers was offered);
    #for the missing prices of the negative sample df it will automatically insert the max_price since discount is 0
    data_train['price'] = np.where(data_train['price'] == 0, data_train['max_price'] * (1 - data_train['discount'] / 100), data_train['price'])
    #minimal price of product; we need to take the minimal price of the bought products; thus, from the basket_df; otherwise, the min_price will also be 0 since we imputed the NaNs with 0 before
    min_price = basket_df.groupby('product')['price'].agg(min).reset_index()
    min_price = min_price.rename(columns={'price': 'min_price'})
    #merge min price to the df
    data_train = pd.merge(data_train, min_price, on='product', how='left')
    
    'Clear Memory'
    del max_price
    del min_price
    
    'Feature Engineering Part III.a'
            
    #CUSTOMER DIMENSION 
    #no_products_bought: number products bought by a customer i      
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper'])['product'].agg('count').reset_index().rename(columns={'product': 'no_products_bought'})
    data_train = pd.merge(data_train, tmp, on='shopper', how='left')
    #spend_per_customer: Customer Lifetime Value (sum € spend by a customer i
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper'])['price'].agg('sum').reset_index().rename(columns={'price': 'spend_per_customer'})
    data_train = pd.merge(data_train, tmp, on='shopper', how='left')
    #no_unique_products: number unique products bought by customer i 
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper'])['product'].agg('nunique').reset_index().rename(columns={'product': 'no_unique_products'})
    data_train = pd.merge(data_train, tmp, on='shopper', how='left')
    #discount_purchase: number products bought at discount by a customer i 
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper'])['discount_offered'].agg('sum').reset_index().rename(columns={'discount_offered': 'discount_purchase'})
    data_train = pd.merge(data_train, tmp, on='shopper', how='left')
    
    #--------------------------
            
    #PRODUCT DIMENSION
    #product_sells: number of times the product was sold        
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['product'])['price'].agg('count').reset_index().rename(columns={'price': 'product_sells'})
    data_train = pd.merge(data_train, tmp, on='product', how='left')
    #product_dis_sells: number of times a product was bought with a discount
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['product'])['discount_offered'].agg('sum').reset_index().rename(columns={'discount_offered': 'product_dis_sells'})
    data_train = pd.merge(data_train, tmp, on='product', how='left')
    #product_dis_sells_share
    data_train['product_dis_sells_share'] = data_train['product_dis_sells'] / data_train['product_sells']
    
     #--------------------------
            
    #CUSTOMER X PRODUCT DIMENSION
    #no_products_bought_per_product: no product j purchases for customer i         
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper', 'product'])['price'].agg('count').reset_index().rename(columns={'price': 'no_products_bought_per_product'})
    data_train = pd.merge(data_train, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_purchases: number purchases of a product j at discount by customer i 
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper', 'product'])['discount_effect'].agg('sum').reset_index().rename(columns={'discount_effect': 'customer_prod_dis_purchases'})
    data_train = pd.merge(data_train, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_bought_dis_share: share a product j is bought at a discount by customer i 
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper', 'product'])['discount_effect'].agg('mean').reset_index().rename(columns={'discount_effect': 'customer_prod_bought_dis_share'})
    data_train = pd.merge(data_train, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_offers: number discount offers of a product j for customer i
    tmp = data_train.groupby(['shopper', 'product'])['discount_offered'].agg('count').reset_index().rename(columns={'discount_offered': 'customer_prod_dis_offers'})
    data_train = pd.merge(data_train, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_offered_share: share of deemed coupons per customer i
    tmp = data_train[(data_train['discount_offered'] == 1)].groupby(['shopper', 'product'])['product_bought'].agg('mean').reset_index().rename(columns={'product_bought': 'customer_prod_dis_offered_share'})
    data_train = pd.merge(data_train, tmp, on=['shopper', 'product'], how='left')
    #customer_product_share: share of product j was bought by customer i in comparison to all other products that were bought by customer i
    data_train['customer_product_share'] = data_train['no_products_bought_per_product'] / data_train['no_products_bought']
    #customer_mean_product_price: average price of an item bought by a customer i 
    data_train['customer_mean_product_price'] = data_train['spend_per_customer'] / data_train['no_products_bought']
    #customer_discount_buy_share: the percentage of products bought at discount by customer i
    data_train['customer_discount_buy_share'] = data_train['discount_purchase'] / data_train['no_products_bought']
    
    #--------------------------
    
    #WEEK X CUSTOMER DIMENSION
    #week_basket_size: number products bought by a customer i in week t 
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['week', 'shopper'])['product'].agg('count').reset_index().rename(columns={'product': 'week_basket_size'})
    data_train = pd.merge(data_train, tmp, on=['week', 'shopper'], how='left')  
    #week_basket_value: sum products in € by a customer i in week t 
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['week', 'shopper'])['price'].agg('sum').reset_index().rename(columns={'price': 'week_basket_value'})
    data_train = pd.merge(data_train, tmp, on=['week', 'shopper'], how='left')
    
    #--------------------------
            
    #CUSTOMER DIMENSION
    #mean_basket_size: the average basket size of customer i      
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper'])['week_basket_size'].agg('mean').reset_index().rename(columns={'week_basket_size': 'mean_basket_size'})
    data_train = pd.merge(data_train, tmp, on='shopper', how='left')
    #mean_basket_value: the average basket value in € of customer i
    tmp = data_train[(data_train['product_bought'] == 1)].groupby(['shopper'])['week_basket_value'].agg('mean').reset_index().rename(columns={'week_basket_value': 'mean_basket_value'})
    data_train = pd.merge(data_train, tmp, on='shopper', how='left')
                     
    'Imputing/Fixing Missing Values'
    #30 shoppers data the first time in week 1; therefore there are no data for week 0
    data_train = data_train[data_train['week_basket_size'].notna()]
    #some shopper never bought a product, even not when it was e.g. discounted; this is valuabe information, therefore, NaNs are set to -1
    data_train['customer_product_share'] = data_train['customer_product_share'].replace(np.nan, -1)
    data_train['no_products_bought_per_product'] = data_train['no_products_bought_per_product'].replace(np.nan, -1)
    data_train['customer_prod_dis_purchases'] = data_train['customer_prod_dis_purchases'].replace(np.nan, -1)
    data_train['customer_prod_bought_dis_share'] = data_train['customer_prod_bought_dis_share'].replace(np.nan, -1)
    data_train['customer_prod_dis_offered_share'] = data_train['customer_prod_dis_offered_share'].replace(np.nan, -1)
    
    'Unit Test Block II.a'
    assert len(list(data_train.columns)) == 35
    assert data_train.isna().sum().sum() == 0
    assert data_train['product_bought'].nunique() == 2
    
    'Store data_train'
    data_train.sort_values(by=['week', 'shopper', 'product'], inplace=True)
    data_train.to_parquet(path + '/train_s2000_final.parquet')
 
    'Feature Engineering Part II.b'
    #maximal price of product
    max_price = data_test.groupby('product')['price'].agg(max).reset_index()
    max_price = max_price.rename(columns = {'price': 'max_price'})
    #merge max price to the df
    data_test = pd.merge(data_test, max_price, on='product', how='left')
    #impute missing prices by the max price minues the offered discount (because this was the price the shoppers was offered);
    #for the missing prices of the negative sample df it will automatically insert the max_price since discount is 0
    data_test['price'] = np.where(data_test['price'] == 0, data_test['max_price'] * (1 - data_test['discount'] / 100), data_test['price'])
    #minimal price of product; we need to take the minimal price of the bought products; thus, from the basket_df; otherwise, the min_price will also be 0 since we imputed the NaNs with 0 before
    min_price = basket_df.groupby('product')['price'].agg(min).reset_index()
    min_price = min_price.rename(columns={'price': 'min_price'})
    #merge min price to the df
    data_test = pd.merge(data_test, min_price, on='product', how='left')
    
    'Clear Memory'
    del basket_df
    del max_price
    del min_price
    
    'Feature Engineering Part III.b'
            
    #CUSTOMER DIMENSION 
    #no_products_bought: number products bought by a customer i      
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper'])['product'].agg('count').reset_index().rename(columns={'product': 'no_products_bought'})
    data_test = pd.merge(data_test, tmp, on='shopper', how='left')
    #spend_per_customer: Customer Lifetime Value (sum € spend by a customer i
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper'])['price'].agg('sum').reset_index().rename(columns={'price': 'spend_per_customer'})
    data_test = pd.merge(data_test, tmp, on='shopper', how='left')
    #no_unique_products: number unique products bought by customer i 
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper'])['product'].agg('nunique').reset_index().rename(columns={'product': 'no_unique_products'})
    data_test = pd.merge(data_test, tmp, on='shopper', how='left')
    #discount_purchase: number products bought at discount by a customer i 
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper'])['discount_offered'].agg('sum').reset_index().rename(columns={'discount_offered': 'discount_purchase'})
    data_test = pd.merge(data_test, tmp, on='shopper', how='left')
    
    #--------------------------
            
    #PRODUCT DIMENSION
    #product_sells: number of times the product was sold        
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['product'])['price'].agg('count').reset_index().rename(columns={'price': 'product_sells'})
    data_test = pd.merge(data_test, tmp, on='product', how='left')
    #product_dis_sells: number of times a product was bought with a discount
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['product'])['discount_offered'].agg('sum').reset_index().rename(columns={'discount_offered': 'product_dis_sells'})
    data_test = pd.merge(data_test, tmp, on='product', how='left')
    #product_dis_sells_share
    data_test['product_dis_sells_share'] = data_test['product_dis_sells'] / data_test['product_sells']
    
     #--------------------------
            
    #CUSTOMER X PRODUCT DIMENSION
    #no_products_bought_per_product: no product j purchases for customer i         
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper', 'product'])['price'].agg('count').reset_index().rename(columns={'price': 'no_products_bought_per_product'})
    data_test = pd.merge(data_test, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_purchases: number purchases of a product j at discount by customer i 
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper', 'product'])['discount_effect'].agg('sum').reset_index().rename(columns={'discount_effect': 'customer_prod_dis_purchases'})
    data_test = pd.merge(data_test, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_bought_dis_share: share a product j is bought at a discount by customer i 
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper', 'product'])['discount_effect'].agg('mean').reset_index().rename(columns={'discount_effect': 'customer_prod_bought_dis_share'})
    data_test = pd.merge(data_test, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_offers: number discount offers of a product j for customer i
    tmp = data_test.groupby(['shopper', 'product'])['discount_offered'].agg('count').reset_index().rename(columns={'discount_offered': 'customer_prod_dis_offers'})
    data_test = pd.merge(data_test, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_offered_share: share of deemed coupons per customer i
    tmp = data_test[(data_test['discount_offered'] == 1)].groupby(['shopper', 'product'])['product_bought'].agg('mean').reset_index().rename(columns={'product_bought': 'customer_prod_dis_offered_share'})
    data_test = pd.merge(data_test, tmp, on=['shopper', 'product'], how='left')
    #customer_product_share: share of product j was bought by customer i in comparison to all other products that were bought by customer i
    data_test['customer_product_share'] = data_test['no_products_bought_per_product'] / data_test['no_products_bought']
    #customer_mean_product_price: average price of an item bought by a customer i 
    data_test['customer_mean_product_price'] = data_test['spend_per_customer'] / data_test['no_products_bought']
    #customer_discount_buy_share: the percentage of products bought at discount by customer i
    data_test['customer_discount_buy_share'] = data_test['discount_purchase'] / data_test['no_products_bought']
    
    #--------------------------
    
    #WEEK X CUSTOMER DIMENSION
    #week_basket_size: number products bought by a customer i in week t 
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['week', 'shopper'])['product'].agg('count').reset_index().rename(columns={'product': 'week_basket_size'})
    data_test = pd.merge(data_test, tmp, on=['week', 'shopper'], how='left')  
    #week_basket_value: sum products in € by a customer i in week t 
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['week', 'shopper'])['price'].agg('sum').reset_index().rename(columns={'price': 'week_basket_value'})
    data_test = pd.merge(data_test, tmp, on=['week', 'shopper'], how='left')
    
    #--------------------------
            
    #CUSTOMER DIMENSION
    #mean_basket_size: the average basket size of customer i      
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper'])['week_basket_size'].agg('mean').reset_index().rename(columns={'week_basket_size': 'mean_basket_size'})
    data_test = pd.merge(data_test, tmp, on='shopper', how='left')
    #mean_basket_value: the average basket value in € of customer i
    tmp = data_test[(data_test['product_bought'] == 1)].groupby(['shopper'])['week_basket_value'].agg('mean').reset_index().rename(columns={'week_basket_value': 'mean_basket_value'})
    data_test = pd.merge(data_test, tmp, on='shopper', how='left')
                     
    'Imputing/Fixing Missing Values'
    #30 shoppers data the first time in week 1; therefore there are no data for week 0
    data_test = data_test[data_test['week_basket_size'].notna()]
    #some shopper never bought a product, even not when it was e.g. discounted; this is valuabe information, therefore, NaNs are set to -1
    data_test['customer_product_share'] = data_test['customer_product_share'].replace(np.nan, -1)
    data_test['no_products_bought_per_product'] = data_test['no_products_bought_per_product'].replace(np.nan, -1)
    data_test['customer_prod_dis_purchases'] = data_test['customer_prod_dis_purchases'].replace(np.nan, -1)
    data_test['customer_prod_bought_dis_share'] = data_test['customer_prod_bought_dis_share'].replace(np.nan, -1)
    data_test['customer_prod_dis_offered_share'] = data_test['customer_prod_dis_offered_share'].replace(np.nan, -1)
    
    'Unit Test Block II.b'
    assert len(list(data_test.columns)) == 35
    assert data_test.isna().sum().sum() == 0
    assert data_test['product_bought'].nunique() == 2
    
    'Store data_test'
    data_test.sort_values(by=['week', 'shopper', 'product'], inplace = True)
    data_test.to_parquet(path + '/test_s2000_final.parquet')
    
    'Unit Test Block III'
    assert min(data_train['week']) == train_start
    assert max(data_train['week']) == train_end
    assert min(data_test['week']) == test_start
    assert max(data_test['week']) == test_end
    
    print('\nData sets (train and test) are generated and saved as a parquet file to: ' + path)
    
    data_train = data_train.reset_index(drop = True)
    data_test = data_test.reset_index(drop = True)
    
    return (data_train, data_test)