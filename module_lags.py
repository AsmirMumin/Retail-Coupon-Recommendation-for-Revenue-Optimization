"""
The purpose of this module is to:
* calculate laged features such as
** lags (= number of weeks since last purchase)
** temporal distribution of purchases
** average number of weeks between two purchases
"""

import numpy as np
import pandas as pd

class LagCalculator:
    """
    This class provides the functionalities for creating lagged features
    """
    def __init__(self, input_dataframe):
        self.tmp_df_for_lags = input_dataframe

    def calculate_lags(self):
        """
        returns: lags between two purchases (= number of weeks since last purchase)
        """
        self.tmp_df_for_lags["lag_weeks_of_product_per_customer"] = -1

        current_product = -1
        current_shopper = -1
        ten_percent = round(len(self.tmp_df_for_lags)/10)

        for row_idx in range(len(self.tmp_df_for_lags)):
            # print progress
            if row_idx % ten_percent == 0:
                print(f"{round(100*row_idx / len(self.tmp_df_for_lags))}% done.")
            # read in row
            row = self.tmp_df_for_lags.iloc[row_idx]

            # do we look at a new product or are we still at the same?
            if row["product"] != current_product:
                new_product = True
            else:
                new_product = False
            # same for shopper
            if row["shopper"] != current_shopper:
                new_shopper = True
            else:
                new_shopper = False
            if new_product or new_shopper:
                current_product = row["product"]
                current_shopper = row["shopper"]
                has_been_bought_already = False
                # jump to the point where the new product has been bought for the first time
                # untill then everything stays at -1
                if row["product_bought"] == 0:
                    continue
            if row["product_bought"] == 1:
                if has_been_bought_already:
                    self.tmp_df_for_lags["lag_weeks_of_product_per_customer"].iloc[
                        row_idx
                    ] = (row["week"] - last_purchase_week)
                    last_purchase_week = row["week"]
                else:
                    has_been_bought_already = True
                    last_purchase_week = row["week"]
            else:
                if has_been_bought_already:
                    self.tmp_df_for_lags["lag_weeks_of_product_per_customer"].iloc[
                        row_idx
                    ] = (row["week"] - last_purchase_week)
        return self.tmp_df_for_lags

    def calculate_purchase_temporal_distribution(self, lags):
        """
        returns: temporal distribution of purchases
        """
        self.purchase_temporal_distribution = (
            lags[(lags["week"] < 89) & (lags["product_bought"] == 1)][
                ["shopper", "product", "week"]
            ]
            .groupby(by=["shopper", "product"])
            .agg("mean")
        )
        self.purchase_temporal_distribution = (
            self.purchase_temporal_distribution.reset_index()
        )
        self.purchase_temporal_distribution = (
            self.purchase_temporal_distribution.rename(
                columns={"week": "purchase_temporal_distribution"}
            )
        )
        return self.purchase_temporal_distribution

    def calculate_avg_no_weeks_between_two_purchases(self, lags):
        """
        returns: average number of weeks between two purchases
        """
        self.avg_no_weeks_between_two_purchases = (
            lags[(lags["week"] < 89) & (lags["product_bought"] == 1)][
                ["shopper", "product", "lag_weeks_of_product_per_customer"]
            ]
            .groupby(by=["shopper", "product"])
            .agg("mean")
        )  
        self.avg_no_weeks_between_two_purchases = (
            self.avg_no_weeks_between_two_purchases.reset_index()
        )
        self.avg_no_weeks_between_two_purchases = self.avg_no_weeks_between_two_purchases.rename(
            columns={
                "lag_weeks_of_product_per_customer": "avg_no_weeks_between_two_purchases"
            }
        )
        return self.avg_no_weeks_between_two_purchases
