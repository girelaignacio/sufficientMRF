# src/markov_networks/base.py

class BaseMRF:
    """
    Base class with utilities shared by all the Markov Network models
    """
    def __init__(self):
        self._coefs = None
        self._is_fitted = False
        self._is_parametric = True # By default, we assume the model is parametric. It estimates the Ising model.

    def _check_if_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"The instance {self.__class__.__name__} has not been fitted. "
                "Call '.fit()' before using this method."
            )