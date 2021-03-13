import pandas as pd

def train_test_splitting(path, train_start, train_end, test_start, test_end, eval_set = True):
    
    #takes data sets that are created by the module 'module_generate_dataset.py'
    train = pd.read_parquet(path + '/train_s2000_final.parquet')
    test = pd.read_parquet(path + '/test_s2000_final.parquet')
    
    print('The following features will be removed from the training data sets (besides the target variable product_bought): \nshopper, \nproduct, \npurchase_w/o_dis, \nno_purchase_w_dis, \ndiscount_effect, \nweek_basket_size and \nweek_basket_value. \nAmong others, reasons are target leakage and non-reproducibility for week 90.')

    if eval_set:
        train = train[(train['week'] >= train_start) & (train['week'] <= train_end)]
        evaluation = test[(test['week'] > train_end) & (test['week'] < test_start)]
        test = test[(test['week'] >= test_start) & (test['week'] <= test_end)]
    
        print('Training Observations: %d' % (len(train)))
        print('Evaluation Observations: %d' % (len(evaluation)))
        print('Testing Observations: %d' % (len(test)))
        print('Observations: %d' % (len(train) + len(test) + len(evaluation)))
        
        X_train = train.drop(['product_bought', 'shopper', 'product', 'purchase_w/o_dis', 'no_purchase_w_dis', 'discount_effect', 'week_basket_size', 'week_basket_value'], axis = 1).values
        X_eval = evaluation.drop(['product_bought', 'shopper', 'product', 'purchase_w/o_dis', 'no_purchase_w_dis', 'discount_effect', 'week_basket_size', 'week_basket_value'], axis = 1).values
        X_test = test.drop(['product_bought', 'shopper', 'product', 'purchase_w/o_dis', 'no_purchase_w_dis', 'discount_effect', 'week_basket_size', 'week_basket_value'], axis = 1).values 

        y_train, y_eval, y_test = train[['product_bought']].values.reshape(-1), evaluation[['product_bought']].values.reshape(-1), test[['product_bought']].values.reshape(-1)
        
        #should results in the number of X or y data set. Here: 3 - train, eval, test
        assert (len(X_train)/len(y_train)) + (len(X_eval)/len(y_eval)) + (len(X_test)/len(y_test)) == 3
        
        return X_train, X_test, X_eval, y_train, y_eval, y_test
    
    else:
        train = train[(train['week'] >= train_start) & (train['week'] <= train_end)]
        test = test[(test['week'] >= test_start) & (test['week'] <= test_end)]
    
        print('Training Observations: %d' % (len(train)))
        print('Testing Observations: %d' % (len(test)))
        print('Observations: %d' % (len(train) + len(test)))
        
        X_train = train.drop(['product_bought', 'shopper', 'product', 'purchase_w/o_dis', 'no_purchase_w_dis', 'discount_effect', 'week_basket_size', 'week_basket_value'], axis = 1).values
        X_test = test.drop(['product_bought', 'shopper', 'product', 'purchase_w/o_dis', 'no_purchase_w_dis', 'discount_effect', 'week_basket_size', 'week_basket_value'], axis = 1).values 

        y_train, y_test = train[['product_bought']].values.reshape(-1), test[['product_bought']].values.reshape(-1)
        
        #should results in the number of X or y data set. Here: 2 - train, eval, test
        assert (len(X_train)/len(y_train)) + (len(X_test)/len(y_test)) == 2
        
        return X_train, X_test, y_train, y_test