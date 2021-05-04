"""
The purpose of this module is to:
* create product clusters
* provide TSNE plot
* provide final category labels
"""

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class p2cluster():
    """
    This class creates product clusters
    """
    
    def __init__(self, w2v_model):
        """
        Class constructor
        """
        self.w2v_model = w2v_model
        self.labels = []
        self.x = []
        self.y = []
    
    def tsne_train(self, perplexity=2, no_iterations=5000):
        """
        Creates TSNE model
        """
        
        tokens = []

        for word in self.w2v_model.wv.vocab:
            tokens.append(self.w2v_model.wv[word])
            self.labels.append(word)

        tsne_model = TSNE(
            perplexity=perplexity,
            n_components=2,
            init="random",
            n_iter=no_iterations,
            random_state=23,
        )
        new_values = tsne_model.fit_transform(tokens)
  
        for value in new_values:
            self.x.append(value[0])
            self.y.append(value[1])


    def tsne_plot(self):
        """
        Create TSNE plot
        """
        plt.figure(figsize=(8, 8))
        for i in range(len(self.x)):
            plt.scatter(self.x[i], self.y[i])
            plt.annotate(
                self.labels[i],
                xy=(self.x[i], self.y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )

        plt.show()
        
    def elbow_plot(self, min_val=2, max_val=40):
        """
        Create elbow plot
        """
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(min_val, max_val))
        visualizer.fit(np.column_stack((self.x, self.y)))  # Fit the data to the visualizer
        visualizer.show()
        
    def train_cluster(self, nclut=25):
        """
        Train kmeans clustering
        """
        self.kmeans = KMeans(n_clusters=nclut, random_state=0).fit(np.column_stack((self.x, self.y)))
        
    def clust_plot(self):
        """
        Plot the clustering results
        """
        p = sns.scatterplot(x=self.x, y=self.y, hue=self.kmeans.labels_, palette="deep") # other palette
        p.legend_.remove()
        plt.show()
        
    def get_categories(self):
        """
        Return DF with product categories
        """
        product_categories = {
            "tsne_x": self.x,
            "tsne_y": self.y,
            "product": self.labels,
            "category_label": self.kmeans.predict(np.column_stack((self.x, self.y))),
            "tmp_sort": self.labels,
        }
        product_categories = pd.DataFrame(data=product_categories)
        product_categories["tmp_sort"] = product_categories["tmp_sort"].astype(float)
        product_categories = product_categories.sort_values(by="tmp_sort")
        del product_categories["tmp_sort"]
        return product_categories