# src/sufficientMRF/nonparametric.py

import numpy as np
import pandas as pd
from itertools import combinations
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base import BaseMRF, logger

class NonParametricModel(BaseMRF):
    """
    Non-parametric Markov Random Field estimation using 
    Exhaustive Search and BIC criterion. 
    Use it when positivity condition does not hold. 
    Estimation procedure is based on Leonardi, F., Carvalho, R., & Frondana, I. (2024). 
    Structure recovery for partially observed discrete Markov random fields on graphs under not necessarily positive distributions.
    Scand J Statist, 51(1), 64–88. https://doi.org/10.1111/sjos.12674
    """
    
    def __init__(self, c: np.ndarray = None, max_neighbors: int = None, n_jobs: int = 1):
        super().__init__()
        if c is None:
            c = np.geomspace(1e-9, 1e1, 10)
        self.c = np.sort(c.reshape(-1))[::-1].reshape(-1, 1)
        self.n_jobs = n_jobs
        self.max_neighbors = max_neighbors # Limit search space if provided
        self._is_parametric = False
        
    def _lpl_bic(self, X, idx_v, idx_W) -> np.ndarray:
        """
        Calculate the BIC-penalized Log-Pseudo-Likelihood

        Args:
            X (np.ndarray): is the design matrix (inherited). It may include the exogenous variable or not.
            idx_v (_type_): index of the target node
            idx_W (_type_): index of the nodes in the neighborhood candidate (W)
        
        Returns:
            np.ndarray: BIC values a penalization hyperparameter
        """
        
        n = X.shape[0]
        # Select only the target variable and its current neighborhood candidates
        active_cols = idx_v + idx_W
        # Xy layout: [X_0, ..., X_p-1, y_0, ..., y_q-1]
        sub_data = X[:, active_cols]
        
        # Identify unique configurations of (v, neighbors)
        # Use return_inverse to map each row to its unique configuration index
        unique_configs, inverse, counts_vW = np.unique(
            sub_data, axis=0, return_inverse=True, return_counts=True
        )
        
        # If there are neighbors, we need the counts of neighbor configurations only (aw)
        if len(idx_W) > 0:
            # The neighborhood is all columns except the first one (idx_v)
            neighbor_data = sub_data[:, 1:]
            _, inv_W, counts_w = np.unique(
                neighbor_data, axis=0, return_inverse=True, return_counts=True
            )
            # Map counts_w back to the same shape as counts_vw using inverse mapping
            # This allows direct element-wise division: P(v|W) = Count(v,W) / Count(W)
            prob_v_given_W = counts_vW[inverse] / counts_w[inv_W]
        else:
            prob_v_given_W = counts_vW[inverse] / n


        lpl = np.sum(np.log(np.clip(prob_v_given_W, 1e-12, 1.0))) # Avoid log(0) by clipping probabilities to a small positive value
        
        # BIC: LPL - Penalty. 
        # complexity is 2^|neighborhood|
        bic = lpl - self.c * (2**len(idx_W)) * np.log(n)
        return bic
    
    def _create_design_matrix(self, X, y = None) -> np.ndarray:
        """
        Create design matrix for the active nodes: [X, y]
        """
        if y is None:
            return X # Only active neighbors, no covariates
        return np.hstack((X, y))
        
    
    def _compute_ne_i(self, i, X_vals, y_vals) -> np.ndarray:
        """
        Estimate neighborhood for the i-th variable using exhaustive search.
        """
        Xy = self._create_design_matrix(X_vals, y_vals)

        q = y_vals.shape[1] if y_vals is not None else 0
        p = X_vals.shape[1]
        
        idx_v = [i]
        idx_others_x = [j for j in range(p) if j != i]
        idx_y = list(range(p, p + q))          # Covariates start after X
        
        # Limit search depth if max_neighbors is set
        limit = self.max_neighbors + 1 if self.max_neighbors else p
        
        best_bic = np.full((len(self.c), 1), -np.inf)
        best_ne = [[] for _ in range(len(self.c))]

        for ne_size in range(limit):
            for indx_W_x in combinations(idx_others_x, ne_size):
                # We always condition on the candidate neighbors in X +  Y (covariates)
                current_w = list(indx_W_x) + idx_y 
                bic = self._lpl_bic(Xy, idx_v, current_w)
                
                # Update best neighborhood for each 'c' value
                for ic in range(len(self.c)):
                    if bic[ic] > best_bic[ic]:
                        best_bic[ic] = bic[ic]
                        best_ne[ic] = list(indx_W_x)
        
        # Convert list of best neighbors to boolean adjacency row
        ne_v_optim = np.zeros((len(self.c), p), dtype=bool)
        for ic, neighbors in enumerate(best_ne):
            ne_v_optim[ic, neighbors] = True
        return ne_v_optim
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, criterion: str = 'min'):
        """
        Fit non-parametric MRF using Pseudo-likelihood.
        
        Args:
            X (pd.DataFrame): Binary features.
            y (pd.DataFrame, optional): Covariates.
            criterion (str): 'min' for AND rule (conservative), 'max' for OR rule.
            
        Returns:
            NonParametricModel: fitted model with symmetric coefficients
        """
        logger.info(f"Fitting Non-Parametric MRF with {self.n_jobs} cores...")
        X_vals = X.values
        Y_vals = y.values if y is not None else None
        n_vars = X_vals.shape[1]
        
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as exec:
                f_partial = partial(self._compute_ne_i, X=X_vals, y=Y_vals)
                results = list(tqdm(exec.map(f_partial, range(n_vars)), 
                                    total=n_vars, desc="Training Nodes"))
        else:
            results = [self._compute_ne_i(i, X_vals, Y_vals) for i in tqdm(range(n_vars), desc="Training Nodes")]
            
        # Stack results into |c| x p x p matrix
        NE = np.stack(results, axis=2)
        
        # Symmetrization
        NE_T = np.moveaxis(NE, -1, -2)
        if criterion == 'min':
            # min criterion y the conservative choice: an edge is included only if both nodes agree
            self._coefs = NE & NE_T
        else:
            # max criterion is the more inclusive choice: an edge is included if at least one node includes it
            self._coefs = NE | NE_T
            
        self._is_fitted = True
        return self