import time
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

def lightgbm_model(X_train, X_test, y_train, y_test, X_eval = None, y_eval = None, eval_set = True, output_probabilities = True, n_estimators = 300, early_stopping_rounds = 50, reg_alpha = 0.3, subsample = 0.5, learning_rate = 0.01, max_depth = 8, verbose = 200):
    
    if eval_set:
        
        start = time.time()
    
        model = lgbm.LGBMClassifier(objective = 'binary', reg_alpha = reg_alpha, subsample = subsample, learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators, metric = ['auc', 'logloss'], random_state = 42) 
        model.fit(X_train, y_train, verbose = verbose, eval_set = [(X_train, y_train), (X_eval, y_eval), (X_test, y_test)], early_stopping_rounds = early_stopping_rounds, eval_metric = ['auc', 'logloss'])

        print(model.best_score_)
    
        end = time.time()
        print('\nThe computation took %.2f minutes.'%((end - start)/60))
    
        #show train, eval and test set over time
        model.evals_result_['evaluation'] = model.evals_result_.pop('valid_1')
        model.evals_result_['testing'] = model.evals_result_.pop('valid_2')
        lgbm.plot_metric(model.evals_result_, metric = 'auc')
        lgbm.plot_metric(model.evals_result_, metric = 'binary_logloss')
        plt.show()
    
        #save the model
        filename = 'lightgbm_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    
    else:
        
        start = time.time()
    
        model = lgbm.LGBMClassifier(objective = 'binary', reg_alpha = reg_alpha, subsample = subsample, learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators, metric = ['auc', 'logloss'], random_state = 42) 
        model.fit(X_train, y_train, verbose = verbose, eval_set = [(X_train, y_train), (X_test, y_test)], early_stopping_rounds = early_stopping_rounds, eval_metric = ['auc', 'logloss'])

        print(model.best_score_)
    
        end = time.time()
        print('\nThe computation took %.2f minutes.'%((end - start)/60))
    
        #show train and test set over time
        model.evals_result_['testing'] = model.evals_result_.pop('valid_1')
        lgbm.plot_metric(model.evals_result_, metric = 'auc')
        lgbm.plot_metric(model.evals_result_, metric = 'binary_logloss')
        plt.show()
        
        #save the model
        filename = 'lightgbm_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
    if output_probabilities:
        
        y_pred = model.predict_proba(X_test)
        
        #confusion matrix
        y_pred_binary = np.where(np.array([col[1] for col in y_pred]) >= 0.5, 1, 0)
        confusion_mat = confusion_matrix(y_test, y_pred_binary)
        
        y_pred = y_pred[:,1]
        
        #auc score
        auc = roc_auc_score(y_test, y_pred)
        #binary logloss
        binary_log_loss = log_loss(y_test, y_pred)
        
        print('Confusion Matrix: \n', confusion_mat)
        print('The AUC score is: ', auc)
        print('The Binary_Log_Loss is: ', binary_log_loss)
        
    else:
        
        y_pred = model.predict(X_test) 
        
        #confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        print('Confusion Matrix: ', confusion_mat)
     
    #save predictions
    savetxt('y_pred.csv', y_pred, delimiter = ',')
    
    return y_pred