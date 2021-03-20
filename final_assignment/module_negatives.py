"""
The purpose of this module is to:
* create negative samples in order to tackle class imblanace
"""

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import random

class NegativeSampleGenerator():
    
    def __init__(self, baskets, no_customers=100000, no_products=250):
        self.no_customers = no_customers
        self.no_products = no_products
        self.total_frequency = csr_matrix((self.no_customers, self.no_products), dtype=np.int8).toarray()
        self.baskets = baskets
        
    def calculate_frequencies(self):
        """
        calculates the total frequencies of product purchases per shopper
        """
        # get product frequency per customer
        for index, row in self.baskets.iterrows():
            for p in row["products"]:
                self.total_frequency[row["shopper"]][p] += 1
        
        # remove impact of last week (89)
        self.total_frequency_without89 = csr_matrix((self.no_customers, self.no_products), dtype=np.int8).toarray()
        self.total_frequency_only89 = csr_matrix((self.no_customers, self.no_products), dtype=np.int8).toarray()

        self.total_frequency_without89 = self.total_frequency.copy()

        for index, row in self.baskets[self.baskets["week"] == 89].iterrows():
            for p in row["products"]:
                self.total_frequency_without89[row["shopper"]][p] -= 1
                self.total_frequency_only89[row["shopper"]][p] += 1
                
    def get_total_frequency_without89(self):
        return self.total_frequency_without89
    
    def get_total_frequency_only89(self):
        return self.total_frequency_only89
    
    def calculate_customer_preferences(self, min_frequency=3):
        """
        calculate the customer preferences bases on the total frequencies with a minimum support of min_frequency
        """
        self.cons_preferences = dict()
        for consumer in range(0, self.no_customers):
            self.cons_preferences[consumer] = list()
            for prod in range(0, self.no_products):
                if self.total_frequency_without89[consumer][prod] >= min_frequency:
                    self.cons_preferences[consumer].append(prod)
        return self.cons_preferences
    
    def show_customer_preferences(self, n=5):
        """
        print method
        """
        for i in range(n):
            print(f"{i}:\n{self.cons_preferences[i]}")
            
    def generate(self):
        """
        generate neagtive samples and
        return: dataframe of negative samples per week per shopper
        """
        
        random.seed(42)

        self.df_negative_samples = pd.DataFrame(columns=["week", "shopper", "product"])

        for week_id in range(90):
            if round(week_id%9==0):
                print(f"{int(100*week_id/90)}% done.")
            for shopper_id in range(self.no_customers):
                try:
                    this_weeks_basket = list(
                        self.baskets[(self.baskets["shopper"] == shopper_id) & (self.baskets["week"] == week_id)][
                            "products"
                        ]
                    )[0]
                except:
                  continue

                not_bought = set(self.cons_preferences[shopper_id]) - set(this_weeks_basket)
                number_of_samples = min(len(not_bought), len(this_weeks_basket))
                negative_samples = random.sample(not_bought, k=number_of_samples)
                for neg in negative_samples:
                    self.df_negative_samples = self.df_negative_samples.append(
                        {"week": week_id,"shopper": shopper_id, "product": neg},
                        ignore_index=True,
                    )
        self.df_negative_samples["product_bought"]=0
        print("100% done.")
        return self.df_negative_samples