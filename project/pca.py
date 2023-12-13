import numpy as np
import pandas as pd
from utils import *
from eig_utils import *

class PCA():
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.__covariance_matrix__ = None
        self.__eigenvalues__ = None
        self.__eigenvectors__ = None
        self.explained_variance = None
        self.__standardized_data__ = None
        self.components_ = None
        self.mean_ = None

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def fit(self, data):
        if self.n_components is None:
            self.n_components = min(*data.shape)
        
        self.mean_ = data.mean(axis=0)
        standardized_data = (data - self.mean_)
        self.__covariance_matrix__ = np.cov(standardized_data, rowvar = False)
        
        # TODO: replace with own function finding eig values and vectors
        self.__eigenvalues__, self.__eigenvectors__ = np.linalg.eig(self.__covariance_matrix__)
        # eigv, eigvec = qr_eigenvalues(self.__covariance_matrix__)
        
        # print(self.__eigenvalues__)
        # print(eigv)
        print(self.__eigenvectors__) 
        # print(eigvec)


        order_of_importance = np.argsort(self.__eigenvalues__)[::-1]

        sorted_eigenvalues = self.__eigenvalues__[order_of_importance]
        self.__sorted_eigenvectors__ = -self.__eigenvectors__[:,order_of_importance]
        self.components_ = (self.__sorted_eigenvectors__.T)[:self.n_components,:]
        self.explained_variance_ratio_ = (sorted_eigenvalues / np.sum(sorted_eigenvalues))[:self.n_components]
        self.explained_variance = sum(self.explained_variance_ratio_[:self.n_components])

    def transform(self, data):
        if self.__eigenvectors__ is None:
            raise "Model doesn't fitted. Use method fit"
        standardized_data = (data - data.mean(axis=0))
        
        try:
            return np.dot(standardized_data, self.__sorted_eigenvectors__[:,:self.n_components])
        except:
            raise  "Something went wrong. Hint: dimensions can be wrong size of the fitted"

    def get_covariance(self):
        if self.__covariance_matrix__ is None:
            raise "Cov matrix doesn't exists. First, fit the model. Use method fit"
        
        return self.__covariance_matrix__


if __name__ == "__main__":
    data = np.array([
        [   1,   2,  -1,   4,  10],
        [   3,  -3,  -3,  12, -15],
        [   2,   1,  -2,   4,   5],
        [   5,   1,  -5,  10,   5],
        [   2,   3,  -3,   5,  12],
        [   4,   0,  -3,  16,   2],
    ])

    img_path = "pig.jpeg"
    out_path = f"{''.join((lambda x: x[:-1] + ['_compressed.'] + [x[-1]])(img_path.split('.')))}"
    comp_im = save_image_from_data(pca_transform(pca_compose(img_path=img_path), n_components=50), filename=out_path)
    print("Original image params:\n", image_info(img_path=img_path), sep="")
    print("Compressed image params:\n",image_info(img_path=out_path), sep="")

    # pca = PCA(n_components=2)
    # pca.fit(data)

    # print(pca.explained_variance_ratio_)
    # print()
    # print(pca.components_)
    # print()
    # print(pca.transform(data))