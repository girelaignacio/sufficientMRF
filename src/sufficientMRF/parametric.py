# src/sufficientMRF/parametric.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV as LogRegcv
from multiprocessing import cpu_count
from tqdm import tqdm
from .base import BaseMRF

import logging
# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class IsingModel(BaseMRF):
    
    def __init__(self, n_lambda: int = 70, cv: int = 10, max_iter: int = 1000, n_jobs = None):
        """
        Initialize the Sparse Ising model with covariates

        Args:
            n_lambda (int, optional): number of lambda values. Defaults to 70.
            cv (int, optional): number of cross-validation folds. Defaults to 10.
            max_iter (int, optional): maximum number of iterations for the solver. Defaults to 1000.
            n_jobs (optional): number of parallel jobs. Defaults to None.
        """
        # Model estimation attributes
        super().__init__()
        self._n_lambda = n_lambda
        self._cv = cv
        self._penalty = 'l1'
        self._solver = 'saga'
        self._max_iter = max_iter
        self._n_jobs = n_jobs or cpu_count() - 1
        
        self._coefs = None

    def _create_design_matrix(self, features, y = None) -> np.ndarray:
        """Create design matrix with interactions between features and covariates if any"""
        if y is None:
            return features # Only features, no covariates
        
        # If there are covariates, create interaction terms
        interactions = features * y
        return np.hstack((features, y, interactions))
        
    def _make_symmetric(self, matrix: np.ndarray, criterion: str = 'min') -> np.ndarray:
        """
        Converts a matrix into a symmetric one according to a criterion

        Args:
            matrix (np.ndarray): square matrix to be symmetrized
            criterion (str, optional): symmetric according to a criterion. Defaults to 'min'.

        Returns:
            np.ndarray: symmetric matrix
        """
        m_t = matrix.T
        if criterion == 'min':
            return np.where(np.abs(matrix) <= np.abs(m_t), matrix, m_t)
        else:
            return np.where(np.abs(matrix) >= np.abs(m_t), matrix, m_t)
    
    @property
    def coefs(self):
        """Get coefficients as attributes: model.coefs"""
        if self._coefs is None:
            logger.warning("The model has not been fitted yet. Call .fit() first.")
        return self._coefs
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Fit Ising model with covariates using logistic regression with L1 penalty
        
        Args:
            X (pd.DataFrame): DataFrame with binary variables (0/1)
            y (pd.DataFrame): DataFrame with covariates (preferably binary)
            
        Returns:
            Ising: Fitted Ising model with covariates
        """
        logger.info("Fitting Ising model...")
        # Load logistic regression model
        model = LogRegcv(cv=self._cv, 
                        max_iter=self._max_iter, 
                        penalty=self._penalty,
                        solver=self._solver, 
                        n_jobs=self._n_jobs, 
                        Cs=self._n_lambda
        )
        
        # Create empty lists to store coefficients
        J_0, h_0 = [], []
        J_y, h_y = ([], []) if y is not None else (None, None)
        
        # Fit a logistic regression for each variable
        X = X.values
        y_vals = y.values.reshape(-1, 1) if y is not None else None

        nvars = X.shape[1]
        
        for idx in tqdm(range(nvars), desc="Training variables"):
            # Prepare data for the model            
            target = X[:, idx]
            features = np.delete(X, idx, axis=1)
            n_features = features.shape[1]
            
            X_design = self._create_design_matrix(features, y_vals)
            
            # Fit model
            fit_ = model.fit(X_design, target)
            
            # Store coefficients
            h_0.append(fit_.intercept_[0])
            j0 = fit_.coef_[0, :n_features]
            j0 = np.insert(j0, idx, 0)
            J_0.append(j0)

            if y is not None:
                h_y.append(fit_.coef_[0, n_features])
                jy = fit_.coef_[0, n_features + 1:]
                jy = np.insert(jy, idx, 0)
                J_y.append(jy)
                
        logger.info("Training completed successfully.")

        # Store coefficients and make J_0 and J_y symmetric according to criterion
        self._coefs_conserv = {'J_0': self._make_symmetric(np.vstack(J_0), criterion = 'min'),
                        'h_0': np.vstack(h_0)}
        self._graph_conserv  = self._coefs_conserv['J_0']  != 0
        
        self._coefs_nconserv = {'J_0': self._make_symmetric(np.vstack(J_0), criterion = 'max'),
                        'h_0': np.vstack(h_0)}
        self._graph_nconserv = self._coefs_nconserv['J_0'] != 0

        if y is not None:
            self._coefs_conserv['J_y'] = self._make_symmetric(np.vstack(J_y), criterion = 'min')
            self._coefs_conserv['h_y'] = np.vstack(h_y)
            self._graph_conserv  = (self._coefs_conserv['J_0']  + self._coefs_conserv['J_y'])  != 0


            self._coefs_nconserv['J_y'] = self._make_symmetric(np.vstack(J_y), criterion = 'max')
            self._coefs_nconserv['h_y'] = np.vstack(h_y)
            self._graph_nconserv = (self._coefs_nconserv['J_0'] + self._coefs_nconserv['J_y']) != 0            
        
        self._is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame, level: str = 'global',
            criterion: str = 'conserv') -> dict:
        self._check_if_fitted()
        
        if criterion == 'conserv':
            coefs = self._coefs_conserv
            graph = self._graph_conserv
            
        elif criterion == 'nconserv':
            coefs = self._coefs_nconserv
            graph = self._coefs_nconserv
        else:
            raise ValueError("criterion must be either 'conserv' or 'nconserv'")

        # Check if covariate effects parameters are not None
        if 'J_y' not in coefs or 'h_y' not in coefs:
            raise ValueError("SDR computation is only valid for models with covariates.")
        
        X_vals = X.values if hasattr(X, 'values') else np.asarray(X)
        X_centered = X_vals - X_vals.mean(axis=0)
    
        h_y = coefs['h_y'].flatten()   # (p,)
        J_y = coefs['J_y']             # (p, p)
        
        return self._predict_from_params(X_centered, graph, h_y, J_y, level)
        
    def _predict_from_params(self, X: np.ndarray, graph: np.ndarray,
                        h_y: np.ndarray, J_y: np.ndarray,
                        level: str) -> dict:
        n, p = X.shape
        
        # Ri[s, j] = θj* · xj  +  0.5 · Σ_{k ∈ N(j)} θjk · xj · xk
        Ri = np.zeros((n, p))
        for j in range(p):
            neighbors = np.where(graph[j])[0]
            Ri[:, j] = h_y[j] * X[:, j]
            for k in neighbors:
                Ri[:, j] += 0.5 * J_y[j, k] * X[:, j] * X[:, k]
        
        R_global = Ri.sum(axis=1)  # (n,)
        
        if level == 'global':
            return {'R': R_global, 'Ri': Ri}
        
        elif level == 'node':
            return {'Ri': Ri, 'R': R_global}
        
        elif level == 'component':
            components = self._get_connected_components(graph)
            R_comp = np.stack(
                [Ri[:, idx].sum(axis=1) for idx in components],
                axis=1
            )  # (n, K)
            return {'R_components': R_comp, 'R': R_global,
                    'Ri': Ri, 'components': components}
        
        else:
            raise ValueError("level must be 'node', 'component' or 'global'")
        
    def _get_connected_components(self, graph: np.ndarray) -> list:
        
        p = graph.shape[0]
        visited = np.zeros(p, dtype=bool)
        components = []
        
        for i in range(p):
            if not visited[i]:
                stack, comp = [i], []
                visited[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in np.flatnonzero(graph[u]):
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
                components.append(np.array(comp, dtype=int))
        
        return components   
    
    def _linear_component(self, X: pd.DataFrame, y: pd.DataFrame = None) -> np.ndarray:
        """Compute linear components of the model."""
        
        # Check if model is fitted
        self._check_if_fitted()
        
        # Fit a logistic regression for each variable
        X = X.values
        y_vals = y.values.reshape(-1, 1) if y is not None else None
        
        if y is not None:
            return np.einsum('ij,jk-> i', X, self._coefs['h_0']) + np.einsum('ij,jk-> i', X * y_vals, self._coefs['h_y']) + \
                   np.einsum('ij,jk,ik-> i', X, self._coefs['J_0'], X) * 0.5 + \
                   np.einsum('ij,jk,ik-> i', X * y_vals, self._coefs['J_y'], X * y_vals) * 0.5

        else:
            return np.einsum('ij,jk-> i', X, self._coefs['h_0']) + np.einsum('ij,jk,ik-> i', X, self._coefs['J_0'], X) * 0.5

    def probabilities(self, X: pd.DataFrame, y: str = None):
        pass