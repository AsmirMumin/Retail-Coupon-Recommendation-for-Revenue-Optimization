# Maximizing Revenue with Individualized Coupon Optimization - Final Assignment for Course "Machine Learning in Marketing"

In 2020, U.S. retailers distributed over 470 billion coupons for packaged consumer goods. At the same time, less than 1 percent of the coupons issued were redeemed (Statista, 2021). Yet, 90 percent of all consumers use coupons, 43 percent of them regularly (very often). Nevertheless, half of consumers would redeem more coupons if it was easier to find coupons for the items they really need. (Valassis, 2019)

Our project tackles this challenge and aims to predict individualised coupons by extracting information from a large set of historical transaction data from a grocery retailer. Our "real-life" data set comprises weekly purchases by 2000 customers of 250 products in 90 weeks and randomly distributed coupons. Further objectives for our project are: 

  (1) Create a high dimensional feature set and come up with sophisticated train-test splitting methods to account for the time-series aspects in the data<br>
  (2) Design and optimize state-of-the-art machine learning models in order to make purchase predictions and compare their average performances<br>
  (3) Assign and choose 5 top ranked coupons to all customers for the following week that yield the highest revenue uplift for the retailer<br>
  
We chose two recent and most widely used tree-based models, XGBoost and LightGBM, and one heuristic model. Among these, our final model for purchase predictions is LightGBM because it outperforms the other models. 

Note: If you are interested in the data sets, please send a request to sebastian.gabel@hu-berlin.de

## Organization

__Institute:__ Humboldt University Berlin, Institute of Marketing <br>
__Course:__ Machine Learning in Marketing <br>
__Semester:__ WS 2020/21 <br>
__Authors:__ 
 - Sophie Bonczyk
 - Anna Franziska Bothe 
 - Christopher Gerling 
 - Asmir Muminovic 
 - Arash Moussavi Tasouj
 
## Content

```
.
├── MLiM___Coupon_Optimization.pdf       # PDF of final assignment
├── final_notebook.jpynb                 # jupyter notebook with coding pipeline
├── coupon_index.parquet                 # final predictions for coupon assignments
├── README.md                            # this readme file
├── requirements.txt                     # configuration file with package versions
├── module_baseline_heuristic_model      # module for calculating heuristic model
├── module_coupon_assignment.py          # module for final coupon assignment
├── module_clustering.py                 # module for clustering, TSNE and category generation
├── module_generate_dataset.py           # module for generating datasets that can be used for the model
├── module_lags.py                       # module for calculating lagged features
├── module_lightgbm.py                   # module for training the LightBGM model
├── module_negatives.py                  # module for calculating negative samples
├── module_p2v.py                        # module for training a gensim P2V model
├── module_train_test_splitting.py       # module for creating a train-test-split
└── module_week90_generate_dataset.py    # module for simulating products of week 90
```

## Requirements

1. This project is implemented with Python 3.8.
2. The final assignment is implemented in Jupyter Lab 2.2.
3. The required packages with their versions are listed in requirements.txt

## Setup
```
1. Download the Jupyter Notebook, the data folder and all .py files
2. Run Jupyter Lab
3. Open "final_notebook.jpynb"
4. Set "path_datasets" to the path with the parquet files and .py files
5. Run all code cells
```

