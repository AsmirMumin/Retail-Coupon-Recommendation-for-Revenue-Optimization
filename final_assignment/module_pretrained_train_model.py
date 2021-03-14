import pickle
import module_lightgbm
from module_lightgbm import predict_lightgbm

def module_pretrained_train_model(pretrained_model = True, X_train = None, X_test = None, y_train = None, y_test = None):

    if pretrained_model:
        print('Please input the name of the saved model:')

        filename = input()
        with open(filename, 'rb') as file:
            pickle_model = pickle.load(file)

        print('Model is loaded.')
        return pickle_model

    else:
        print('Available parameters are: X_train, X_test, y_train, y_test, X_eval = None, y_eval = None, eval_set = True, output_probabilities = True, estimators = 300, early_stopping_rounds = 50, \nreg_alpha = 0.3, subsample = 0.5, learning_rate = 0.01, max_depth = 8, verbose = 200.')
        print('Would you like to use the pre-configurable (all except train and test) parameters? Please answer with yes or no: ')
        answer = input()

        if answer == 'Yes' or answer == 'yes':
            print('The training und testing sets should be named as followed: X_train, X_test, y_train, y_test')
            y_pred = predict_lightgbm(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
            print('Prediction is computed.')
              
            return y_pred
        
        elif answer == 'No' or answer == 'no':
            print('Go to hell and make the prediction yourself with the module. Too many parameters to input.')
        else:
            print('Please type in yes or no. Those are the only valid answers.')

