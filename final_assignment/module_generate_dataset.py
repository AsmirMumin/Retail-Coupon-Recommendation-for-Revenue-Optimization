#load packages
import pandas as pd
import numpy as np


def generate_dataset(path):
    
            'Load Data Sets'
            #customers past purchase (week 0-89, shopper, product, price in € cents)
            basket_df = pd.read_parquet(path + '/baskets.parquet')
            #coupons customers received in the past (week, shopper, product, discount in %)
            coupon_df = pd.read_parquet(path + '/coupons.parquet')
            #negative samples generate randomly from the customer preference with the length i of the corresponding week j
            negative_sample_df = pd.read_parquet(path + '/df_negative_samples.parquet')
            #categories among products that were created with TSNE
            categories = pd.read_csv(path + '/product_categories.csv')
            
            #shrink dataframe to only the 2000 shopper that we need to make predictions for
            basket_df = basket_df[basket_df['shopper'] < 2000]
            coupon_df = coupon_df[coupon_df['shopper'] < 2000]
    
            'Merge Data Sets'
            data = pd.merge(basket_df, coupon_df, on = ['week', 'shopper', 'product'], how = 'outer')
            data = pd.merge(data, negative_sample_df, on = ['week', 'shopper', 'product'], how = 'outer')
            data = pd.merge(data, categories, on = ['product'], how = 'left')
    
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
            
            'Unit Test Block I'
            assert data['shopper'].nunique() == 2000
            assert data['discount_offered'].nunique() == 2
            assert data['product_bought'].nunique() == 2
            assert data['purchase_w/o_dis'].nunique() == 2
            assert data['no_purchase_w_dis'].nunique() == 2
            assert data['discount_effect'].nunique() == 2
            assert data.isna().sum().sum() == 0
            
            #clear memory 
            del basket_df
            del coupon_df
            del negative_sample_df
            
            'Feature Engineering Part II'
            #maximal price of product
            max_price = data.groupby('product')['price'].agg(max).reset_index()
            max_price = max_price.rename(columns = {'price': 'max_price'})
            #merge max price to the df
            data = pd.merge(data, max_price, on = 'product', how = 'left')
            #impute missing prices by the max price minues the offered discount (because this was the price the shoppers was offered);
            #for the missing prices of the negative sample df it will automatically insert the max_price since discount is 0
            data['price'] = np.where(data['price'] == 0, data['max_price'] * (1-(data['discount']/100)), data['price'])
            
            #minimal price of product
            min_price = data.groupby('product')['price'].agg(min).reset_index()
            min_price = min_price.rename(columns = {'price': 'min_price'})
            #merge min price to the df
            data = pd.merge(data, min_price, on = 'product', how = 'left')
            
            #clear memory
            del max_price
            del min_price
            
            'Feature Engineering Part III'
            #CUSTOMER DIMENSION 
            #no_products_bought: number products bought by a customer i
            tmp = data[data.product_bought == 1].groupby(['shopper'])['product'].agg('count').reset_index().rename(columns={'product':'no_products_bought'})
            data = pd.merge(data, tmp, on='shopper', how='left')
        
            #spend_per_customer: Customer Lifetime Value (sum € spend by a customer i
            tmp = data[data.product_bought == 1].groupby(['shopper'])['price'].agg('sum').reset_index().rename(columns={'price':'spend_per_customer'})
            data = pd.merge(data, tmp, on='shopper', how='left')
        
            #no_unique_products: number unique products bought by customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper'])['product'].agg('nunique').reset_index().rename(columns={'product':'no_unique_products'})
            data = pd.merge(data, tmp, on='shopper', how='left')
        
            #discount_purchase: number products bought at discount by a customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper'])['discount'].agg('count').reset_index().rename(columns={'discount':'discount_purchase'})
            data = pd.merge(data, tmp, on='shopper', how='left')
        
             #--------------------------
        
            #PRODUCT DIMENSION
            #product_sells: number of times the product was sold 
            tmp = data[data.product_bought == 1].groupby(['product'])['price'].agg('count').reset_index().rename(columns={'price':'product_sells'})
            data = pd.merge(data, tmp, on='product', how='left')
        
            #product_dis_sells
            tmp = data[data.product_bought == 1].groupby(['product'])['discount_offered'].agg('count').reset_index().rename(columns={'discount_offered':'product_dis_sells'})
            data = pd.merge(data, tmp, on='product', how='left')
        
            #product_dis_sells_share
            data['product_dis_sells_share'] = data['product_dis_sells']/data['product_sells']
        
             #--------------------------
        
            #CUSTOMER X PRODUCT DIMENSION
            #no_products_bought_per_product: no product j purchases for customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper','product'])['price'].agg('count').reset_index().rename(columns={'price':'no_products_bought_per_product'})
            data = pd.merge(data, tmp, on=['shopper','product'], how='left')
        
            #customer_prod_dis_purchases: number purchases of a product j at discount by customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper','product'])['discount_effect'].agg('count').reset_index().rename(columns={'discount_effect':'customer_prod_dis_purchases'})
            data = pd.merge(data, tmp, on=['shopper','product'], how='left')
        
            #customer_prod_bought_dis_share: share a product j is bought at a discount by customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper','product'])['discount_effect'].agg('mean').reset_index().rename(columns={'discount_effect':'customer_prod_bought_dis_share'})
            data = pd.merge(data, tmp, on=['shopper','product'], how='left')
        
            #customer_prod_dis_offers: number discount offers of a product j for customer i 
            tmp = data.groupby(['shopper','product'])['discount_offered'].agg('count').reset_index().rename(columns={'discount_offered':'customer_prod_dis_offers'})
            data = pd.merge(data, tmp, on=['shopper','product'], how='left')
        
            #customer_product_dis_offered_share: share product j was offered at discount and bought by customer i
            tmp = data[data.product_bought == 1].groupby(['shopper','product'])['discount_offered'].agg('mean').reset_index().rename(columns={'discount_offered':'customer_prod_dis_offer_share'})
            data = pd.merge(data, tmp, on=['shopper','product'], how='left')
        
            #customer_product_share: share of product j was bought by customer i in comparison to all other products that were bought by customer i
            data['customer_product_share'] = data['no_products_bought_per_product']/data['no_products_bought']
        
            #customer_mean_product_price: average price of an item bought by a customer i  
            data['customer_mean_product_price'] = data['spend_per_customer']/data['no_products_bought']
        
            #customer_discount_buy_share: the percentage of products bought at discount by customer i 
            data['customer_discount_buy_share'] = data['discount_purchase']/(data['no_products_bought'])
        
            #--------------------------
        
            #WEEK X CUSTOMER DIMENSION
            #week_basket_size: number products bought by a customer i in week t 
            tmp = data[data.product_bought == 1].groupby(['week','shopper'])['product'].agg('count').reset_index().rename(columns={'product':'week_basket_size'})
            data = pd.merge(data, tmp, on=['week','shopper'], how='left')
        
            #week_basket_value: sum products in € by a customer i in week t 
            tmp = data[data.product_bought == 1].groupby(['week','shopper'])['price'].agg('sum').reset_index().rename(columns={'price':'week_basket_value'})
            data = pd.merge(data, tmp, on=['week','shopper'], how='left')
        
             #--------------------------
        
            #CUSTOMER DIMENSION
            #mean_basket_size: the average basket size of customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper'])['week_basket_size'].agg('mean').reset_index().rename(columns={'week_basket_size':'mean_basket_size'})
            data = pd.merge(data, tmp, on='shopper', how='left')
        
            #mean_basket_value: the average basket value in € of customer i 
            tmp = data[data.product_bought == 1].groupby(['shopper'])['week_basket_value'].agg('mean').reset_index().rename(columns={'week_basket_value':'mean_basket_value'})
            data = pd.merge(data, tmp, on='shopper', how='left')
            
            'Imputing/Fixing Missing Values'
            #30 shoppers data the first time in week 1; therefore there are no data for week 0
            data = data[data['week_basket_size'].notna()]
            #some shopper never bought a product, even not when it was e.g. discounted; this is valuabe information, therefore, NaNs are set to -1
            data['customer_product_share'] = data['customer_product_share'].replace(np.nan, -1)
            data['no_products_bought_per_product'] = data['no_products_bought_per_product'].replace(np.nan, -1)
            data['customer_prod_dis_purchases'] = data['customer_prod_dis_purchases'].replace(np.nan, -1)
            data['customer_prod_bought_dis_share'] = data['customer_prod_bought_dis_share'].replace(np.nan, -1)
            data['customer_prod_dis_offer_share'] = data['customer_prod_dis_offer_share'].replace(np.nan, -1)
        
            'Unit Test Block II'
            assert len(list(data.columns)) == 32
            assert data.isna().sum().sum() == 0
            assert data['product_bought'].nunique() == 2
            
            data.sort_values(by=['week','shopper','product'], inplace=True)
            data.to_parquet('data_s2000_final.parquet')
            
            print("Data set is generated and saved as a parquet file to: " + path)
            return data
            
            
