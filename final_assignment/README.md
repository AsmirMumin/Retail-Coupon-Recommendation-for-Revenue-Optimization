# Final Assignment for course "Machine Learning in Marketing"


## Organization

__Institute:__ Humboldt University Berlin, Institute of Marketing <br>
__Semester:__ WS 2020/21 <br>
__Students:__ 
 - Sophie Bonczyk (561846)
 - Anna Franziska Bothe (576309)
 - Christopher Gerling (598370)
 - Asmir Muminovic (600582)
 - Arash Moussavi Tasouj (600651)
 

## Content

```
.
├── final_assignment.pdf                 # PDF of final assignment
├── final_notebook.jpynb                 # jupyter notebook with coding pipeline
├── coupons_predictions.parquet          # final predictions for coupon assignments
├── README.md                            # this readme file
├── requirements.txt                     # configuration file with package versions
├── module_baseline_heuristic_model      # module for calculating heuristic model
├── module_coupon_assignment.py          # module for final coupon assignment
├── module_clustering.py                 # module for clustering, TSNE and category generation
├── module_generate_dataset.py           # module for generating datasets that can be used for the model
├── module_lags.py                       # module for calculating lagged features
├── module_lightgbm.py                   # module for training the LightBGM model
├── module_merge_datasets.py             # module for merging all dataframes to one
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
1. Run Jupyter Lab
2. Open "final_notebook.jpynb"
3. Set "path_datasets" to the path with the parquet files
4. Run all code cells
```