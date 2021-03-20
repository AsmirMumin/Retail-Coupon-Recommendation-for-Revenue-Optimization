"""
The purpose of this module is to:
* train a product2vec model based on gensim:Word2Vec
"""

import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class p2v:
    """This class trains a P2V model based on the gensim Word2Vec approach."""

    def __init__(self, input_baskets):
        self.product_list = input_baskets
        
    def create_product_list(self):
        """
        generate list of all purchased products
        """
        self.product_list = list(self.product_list)
        self.product_list = [[str(i) for i in line] for line in self.product_list]

    def head(self, n=10):
        """
        print method
        """
        print(self.product_list[0:n])

    def train_p2v(self, vec_dim=30, epochs=100):
        """
        train gensim model
        """
        epoch_logger = EpochLogger()
        self.vec_dim = vec_dim
        self.p2v_model = Word2Vec(
            self.product_list,
            min_count=30,
            window=15,
            iter = epochs,
            size=self.vec_dim,
            workers=4,
            callbacks=[epoch_logger],
        )

    def get_insights(self, product_id):
        """
        print method of insights
        """
        print(f"{self.vec_dim}-dimensional vector for {product_id}:")
        print(self.p2v_model.wv[str(product_id)])
        print(f"\nMost similar products for {product_id}:")
        print(self.p2v_model.wv.most_similar(str(product_id)))


class EpochLogger(CallbackAny2Vec):
    """
    Print progress of P2V training to console
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        if self.epoch % 5 == 0:
            print("Epoch #{}".format(self.epoch))

    def on_epoch_end(self, model):
        # print("Epoch #{} end".format(self.epoch))
        self.epoch += 1