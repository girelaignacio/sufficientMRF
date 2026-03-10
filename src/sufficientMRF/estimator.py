# src/markov_networks/estimator.py
from .parametric import IsingModel
# from .nonparametric import NonParametricModel (Lo importaremos cuando lo crees)

class MarkovNetwork:
    def __init__(self, method= 'parametric', **kwargs):
        self.method = method.lower()
        
        if self.method == 'parametric':
            self.model = IsingModel(**kwargs)
        elif self.method == 'nonparametric':
            pass # self.model = NonParametricModel(**kwargs)
        else:
            raise ValueError("'method' argument should be either 'parametric' or 'nonparametric'.")

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self