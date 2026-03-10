# src/sufficientMRF/parametric.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV as LogRegcv
from multiprocessing import cpu_count
from tqdm import tqdm
from .base import BaseMRF, logger

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
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, criterion: str = 'min'):
        """
        Fit Ising model with covariates using logistic regression with L1 penalty
        
        Args:
            X (pd.DataFrame): DataFrame with binary variables (0/1)
            y (pd.DataFrame): DataFrame with covariates (preferably binary)
            criterion (str, optional): Criterion to make J symmetric ('min' or 'max'). Defaults to 'min'.
            
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

        # Store coefficients and make J_0 and J_y symmetric according to criterion
        self._coefs = {'J_0': self._make_symmetric(np.vstack(J_0), criterion),
                        'h_0': np.vstack(h_0)}
        if y is not None:
            self._coefs['J_y'] = self._make_symmetric(np.vstack(J_y), criterion)
            self._coefs['h_y'] = np.vstack(h_y)
        
        logger.info("Training completed successfully.")
        return self

    def sdr(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the sufficient dimension reduction (SDR) of the model.
        
        Args:
            X (pd.DataFrame): DataFrame with binary variables (0/1)
        
        Returns:
            np.ndarray: SDR values
        """
        
        # Check if model is fitted
        self._check_if_fitted()
        
        # Check if covariate effects parameters are not None
        if 'J_y' not in self._coefs or 'h_y' not in self._coefs:
            raise ValueError("SDR computation is only valid for models with covariates.")
        
        # Compute SDR
        # First center the data
        X_centered = X.values - X.values.mean(axis=0)
        sdr_values = np.einsum('ij,jk-> i', X_centered, self._coefs['h_y']) + \
                        0.5 * np.einsum('ij,jk,ik-> i', X_centered, self._coefs['J_y'], X_centered)
        return sdr_values
    
    def linear_component(self, X: pd.DataFrame, y: pd.DataFrame = None) -> np.ndarray:
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