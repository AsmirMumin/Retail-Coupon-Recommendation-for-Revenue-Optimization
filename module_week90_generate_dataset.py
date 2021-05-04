"""
The purpose of this module is to:
* generate a dataset that can be used for the prediction for week90

In order to achieve this, the same feature engineering is applied as for the construction of the train and test set

Prerequisite: 
* The following datasets should be downloaded/generated and named: 
    * download: baskets.parquet, coupons.parquet
    * create: df_negative_samples.parquet, product_categories.csv, avg_no_weeks_between_two_purchases.parquet, lags.parquet, purchase_temporal_distribution.parquet

"""

#load libraries
import pandas as pd
import numpy as np

def week90_generate_dataset(path):
    
    """
    input: 
        path: path to datasets -> outputted data set is also saved as /week90_s2000_final.parquet to this path
    output: 
        week90: dataset for week90 
    
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
    lags90 = lags[lags['week'] == 90]
    #distribution of purchases, e.g whether they occurs frequently or rather in the first/last half of the timeseries
    purchase_temporal_distribution = pd.read_parquet(path + '/purchase_temporal_distribution.parquet')
    
    #shrink dataframe to only the 2000 shopper that we need to make predictions for
    basket_df = basket_df[(basket_df['shopper'] < 2000)]
    coupon_df = coupon_df[(coupon_df['shopper'] < 2000)]
    
    'Merge Data Sets'
    data = pd.merge(basket_df, coupon_df, on=['week', 'shopper', 'product'], how='outer')
    data = pd.merge(data, negative_sample_df, on=['week', 'shopper', 'product'], how='outer')
    data = pd.merge(data, categories, on=['product'], how='left')
    
    week90 = pd.merge(lags90, categories, on=['product'], how='left')
    week90 = pd.merge(week90, avg_no_weeks_between_two_purchases, on=['shopper', 'product'], how='left')
    week90 = pd.merge(week90, purchase_temporal_distribution, on=['shopper', 'product'], how='left')
    
    'Feature Engineering Part I'
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

    #missing values are imputed with -1 (=never bought) since there missing not differentiates from the pure absense (0)
    week90['avg_no_weeks_between_two_purchases'] = week90['avg_no_weeks_between_two_purchases'].replace(np.nan, -1)
    week90['purchase_temporal_distribution'] = week90['purchase_temporal_distribution'].replace(np.nan, -1)
    
    'Unit Test Block I'    
    assert data['shopper'].nunique() == 2000 
    assert data['discount_offered'].nunique() == 2
    assert data['product_bought'].nunique() == 2
    assert data['purchase_w/o_dis'].nunique() == 2
    assert data['no_purchase_w_dis'].nunique() == 2
    assert data['discount_effect'].nunique() == 2
    assert max(data['week']) == 89
    assert min(data['week']) == 0
    assert data.isna().sum().sum() == 0
    
    assert week90['shopper'].nunique() == 2000
    assert max(week90['week']) == 90
    assert min(week90['week']) == 90
    assert week90.isna().sum().sum() == week90.shape[0]
    
    'Clear Memory'
    del coupon_df
    del negative_sample_df
    del categories
    
    'Feature Engineering Part II'
    #maximal price of product
    max_price = data.groupby('product')['price'].agg(max).reset_index()
    max_price = max_price.rename(columns = {'price': 'max_price'})
    #merge max price to week90
    week90 = pd.merge(week90, max_price, on = 'product', how = 'left')
    #minimal price of product; we need to take the minimal price of the bought products; thus, from the basket_df; otherwise, the min_price will also be 0 since we imputed the NaNs with 0 before
    min_price = basket_df.groupby('product')['price'].agg(min).reset_index()
    min_price = min_price.rename(columns = {'price': 'min_price'})
    #merge max price to week90
    week90 = pd.merge(week90, min_price, on = 'product', how = 'left')
    
    'Clear Memory'
    del max_price
    del min_price
    del basket_df
    
    'Feature Engineering Part III'
            
    #CUSTOMER DIMENSION 
    #no_products_bought: number products bought by a customer i      
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper'])['product'].agg('count').reset_index().rename(columns={'product': 'no_products_bought'})
    week90 = pd.merge(week90, tmp, on='shopper', how='left')
    #spend_per_customer: Customer Lifetime Value (sum € spend by a customer i
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper'])['price'].agg('sum').reset_index().rename(columns={'price': 'spend_per_customer'})
    week90 = pd.merge(week90, tmp, on='shopper', how='left')
    #no_unique_products: number unique products bought by customer i 
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper'])['product'].agg('nunique').reset_index().rename(columns={'product': 'no_unique_products'})
    week90 = pd.merge(week90, tmp, on='shopper', how='left')
    #discount_purchase: number products bought at discount by a customer i 
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper'])['discount_offered'].agg('sum').reset_index().rename(columns={'discount_offered': 'discount_purchase'})
    week90 = pd.merge(week90, tmp, on='shopper', how='left')
    
    #--------------------------
            
    #PRODUCT DIMENSION
    #product_sells: number of times the product was sold        
    tmp = data[(data['product_bought'] == 1)].groupby(['product'])['price'].agg('count').reset_index().rename(columns={'price': 'product_sells'})
    week90 = pd.merge(week90, tmp, on='product', how='left')
    #product_dis_sells: number of times a product was bought with a discount
    tmp = data[(data['product_bought'] == 1)].groupby(['product'])['discount_offered'].agg('sum').reset_index().rename(columns={'discount_offered': 'product_dis_sells'})
    week90 = pd.merge(week90, tmp, on='product', how='left')
    #product_dis_sells_share
    week90['product_dis_sells_share'] = week90['product_dis_sells'] / week90['product_sells']
    
     #--------------------------
            
    #CUSTOMER X PRODUCT DIMENSION
    #no_products_bought_per_product: no product j purchases for customer i         
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper', 'product'])['price'].agg('count').reset_index().rename(columns={'price': 'no_products_bought_per_product'})
    week90 = pd.merge(week90, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_purchases: number purchases of a product j at discount by customer i 
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper', 'product'])['discount_effect'].agg('sum').reset_index().rename(columns={'discount_effect': 'customer_prod_dis_purchases'})
    week90 = pd.merge(week90, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_bought_dis_share: share a product j is bought at a discount by customer i 
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper', 'product'])['discount_effect'].agg('mean').reset_index().rename(columns={'discount_effect': 'customer_prod_bought_dis_share'})
    week90 = pd.merge(week90, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_offers: number discount offers of a product j for customer i
    tmp = data.groupby(['shopper', 'product'])['discount_offered'].agg('count').reset_index().rename(columns={'discount_offered': 'customer_prod_dis_offers'})
    week90 = pd.merge(week90, tmp, on=['shopper', 'product'], how='left')
    #customer_prod_dis_offered_share: share of deemed coupons per customer i
    tmp = data[(data['discount_offered'] == 1)].groupby(['shopper', 'product'])['product_bought'].agg('mean').reset_index().rename(columns={'product_bought': 'customer_prod_dis_offered_share'})
    week90 = pd.merge(week90, tmp, on=['shopper', 'product'], how='left')
    #customer_product_share: share of product j was bought by customer i in comparison to all other products that were bought by customer i
    week90['customer_product_share'] = week90['no_products_bought_per_product'] / week90['no_products_bought']
    #customer_mean_product_price: average price of an item bought by a customer i 
    week90['customer_mean_product_price'] = week90['spend_per_customer'] / week90['no_products_bought']
    #customer_discount_buy_share: the percentage of products bought at discount by customer i
    week90['customer_discount_buy_share'] = week90['discount_purchase'] / week90['no_products_bought']
    
    #--------------------------
    
    #WEEK X CUSTOMER DIMENSION
    #weekly feature on can be adapted to the week90 data set because we don't know anything about the exact purchase behaviour in week 90 yet; however,
    #they are needed to compute the mean_basket_size and mean_basket_value which will be merge to the week90 data set
    
    #week_basket_size: number products bought by a customer i in week t 
    tmp = data[(data['product_bought'] == 1)].groupby(['week', 'shopper'])['product'].agg('count').reset_index().rename(columns={'product': 'week_basket_size'})
    data = pd.merge(data, tmp, on=['week', 'shopper'], how='left')  
    #week_basket_value: sum products in € by a customer i in week t 
    tmp = data[(data['product_bought'] == 1)].groupby(['week', 'shopper'])['price'].agg('sum').reset_index().rename(columns={'price': 'week_basket_value'})
    data = pd.merge(data, tmp, on=['week', 'shopper'], how='left')
    
    #--------------------------
            
    #CUSTOMER DIMENSION
    #mean_basket_size: the average basket size of customer i      
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper'])['week_basket_size'].agg('mean').reset_index().rename(columns={'week_basket_size': 'mean_basket_size'})
    week90 = pd.merge(week90, tmp, on='shopper', how='left')
    #mean_basket_value: the average basket value in € of customer i
    tmp = data[(data['product_bought'] == 1)].groupby(['shopper'])['week_basket_value'].agg('mean').reset_index().rename(columns={'week_basket_value': 'mean_basket_value'})
    week90 = pd.merge(week90, tmp, on='shopper', how='left')
    #--------------------------
                     
    'Imputing/Fixing Missing Values'
    #some shopper never bought a product, even not when it was e.g. discounted; this is valuabe information, therefore, NaNs are set to -1
    week90['customer_product_share'] = week90['customer_product_share'].replace(np.nan, -1)
    week90['no_products_bought_per_product'] = week90['no_products_bought_per_product'].replace(np.nan, -1)
    week90['customer_prod_dis_purchases'] = week90['customer_prod_dis_purchases'].replace(np.nan, -1)
    week90['customer_prod_bought_dis_share'] = week90['customer_prod_bought_dis_share'].replace(np.nan, -1)
    week90['customer_prod_dis_offered_share'] = week90['customer_prod_dis_offered_share'].replace(np.nan, -1)
    
    'Unit Test Block II.a'
    assert len(list(week90.columns)) == 27
    #all product_bought rows should be NaN but nothing else 
    assert week90.isna().sum().sum() == week90.shape[0]
    
    'Store data'
    week90.sort_values(by = ['week', 'shopper', 'product'], inplace = True)
    week90.to_parquet(path + '/week90_s2000_final.parquet')
    
    print('\nThe data set for week 90 is generated and saved as a parquet file to: ' + path)
    
    week90 = week90.reset_index(drop = True)
    
    return week90