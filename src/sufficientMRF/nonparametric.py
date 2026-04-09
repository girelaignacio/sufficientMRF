# src/sufficientMRF/nonparametric.py

import numpy as np
import pandas as pd
from itertools import combinations
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from tqdm import tqdm
from .base import BaseMRF
from sklearn.model_selection import RepeatedKFold

import logging
# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NonParametricModel(BaseMRF):
    """
    Non-parametric Markov Random Field estimation using 
    Exhaustive Search and BIC criterion. 
    Use it when positivity condition does not hold. 
    Estimation procedure is based on Leonardi, F., Carvalho, R., & Frondana, I. (2024). 
    Structure recovery for partially observed discrete Markov random fields on graphs under not necessarily positive distributions.
    Scand J Statist, 51(1), 64–88. https://doi.org/10.1111/sjos.12674
    """
    
    def __init__(self, number_c: int = None, max_neighbors: int = None, n_jobs: int = 1):
        super().__init__()
        if number_c is None:
            number_c = 100
        c = np.geomspace(1e-9, 1e1, number_c)
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
        unique_configs, unique_configs_idxs, counts_vW = np.unique(
            sub_data, axis=0, return_inverse=True, return_counts=True
        )
        
        # If there are neighbors, we need the counts of neighbor configurations only (aw)
        if len(idx_W) > 0:
            # The neighborhood is all columns except the first one (idx_v)
            neighbor_data = sub_data[:, 1:]
            _, sub_unique_configs_idxs, counts_w = np.unique(
                neighbor_data, axis=0, return_inverse=True, return_counts=True
            )
            # Map counts_w back to the same shape as counts_vw using inverse mapping
            # This allows direct element-wise division: P(v|W) = Count(v,W) / Count(W)
            prob_v_given_W = counts_vW[unique_configs_idxs] / counts_w[sub_unique_configs_idxs]
        else:
            prob_v_given_W = counts_vW[unique_configs_idxs] / n


        lpl = np.sum(np.log(np.clip(prob_v_given_W, 1e-12, 1.0))) # Avoid log(0) by clipping probabilities to a small positive value
        
        # BIC: LPL - Penalty. 
        # complexity is 2^|neighborhood| becase the covariate is binary
        bic = lpl - self.c * (2**len(idx_W)) * np.log(n)
        return bic
    
    def _create_design_matrix(self, X, y = None) -> np.ndarray:
        """
        Create design matrix for the active nodes: [X, y]
        """
        if y is None:
            return X # Only active neighbors, no covariates
        return np.column_stack([X, y])
        
    def _compute_ne_i(self, i, X_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        """
        Estimate neighborhood for the i-th variable using exhaustive search.
        """
        Xy = self._create_design_matrix(X_vals, y_vals)
        
        p = X_vals.shape[1]
        q = y_vals.shape[1] if y_vals is not None else 0
        
        
        idx_v = [i]
        idx_others_x = [j for j in range(p) if j != i]
        idx_y = list(range(p, p + q))          # Covariates start after X
        
        # Limit search depth if max_neighbors is set
        limit = self.max_neighbors + 1 if self.max_neighbors else p
        
        best_bic = np.full((len(self.c), 1), -np.inf)
        best_ne = [[] for _ in range(len(self.c))]

        for ne_size in range(limit):
            for idx_W_x in combinations(idx_others_x, ne_size):
                # We always condition on the candidate neighbors in X +  Y (covariates)
                current_w = list(idx_W_x) + idx_y 
                bic = self._lpl_bic(Xy, idx_v, current_w)
                
                # Update best neighborhood for each 'c' value
                for ic in range(len(self.c)):
                    if bic[ic] > best_bic[ic]:
                        best_bic[ic] = bic[ic]
                        best_ne[ic] = list(idx_W_x)
        
        # Convert list of best neighbors to boolean adjacency row
        ne_v_optim = np.zeros((len(self.c), p), dtype=bool)
        
        for ic, neighbors_idx in enumerate(best_ne): # on best_ne because best_ne is a list of lists of neighbors for each 'c' value
            # Save neighbors the the i-th for the best 'c' value
            ne_v_optim[ic, neighbors_idx] = True
            
        return ne_v_optim, best_bic
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, model_selection: str = 'bic', **kwargs):
        """
        Fit non-parametric MRF using Pseudo-likelihood.
        
        Args:
            X (pd.DataFrame): Binary features.
            y (pd.DataFrame, optional): Covariates. Must be binary and 1-D.     
            criterion (str): 'min' for AND rule (conservative), 'max' for OR rule.
            
        Returns:
            NonParametricModel: fitted model with symmetric coefficients
        """
        logger.info(f"Fitting Non-Parametric MRF with {self.n_jobs} cores...")
        X_vals = X.values
        y_vals = y.values.reshape(-1, 1) if y is not None else None
        
        if model_selection == "bic":
            results = self._fit_bic(X_vals, y_vals)
        elif model_selection == "stable":
            results = self._fit_stable(X_vals,y_vals)
        elif model_selection == "kfold":
            pass
        else: 
            raise ValueError(f"model_selection should be 'bic', 'stable' or 'kfold'")
        
        graph_conserv, graph_nconserv = results
        
        self._coefs_conserv   = graph_conserv    # (p, p) bool
        self._coefs_nconserv  = graph_nconserv   # (p, p) bool
        
        self._node_logratios_conserv  = self._build_all_prob_tables(X_vals, y_vals, graph_conserv)
        self._node_logratios_nconserv = self._build_all_prob_tables(X_vals, y_vals, graph_nconserv)

        self._is_fitted = True
        
        return self
    
    def _build_node_tables(self, X_y: np.ndarray, graph: np.ndarray) -> list[dict]:
        p = X_y.shape[1]
        tables = []
        n_y = len(X_y)

        for j in range(p):
            neighbors = np.where(graph[j])[0]
            node_col = X_y[:, j].astype(int)
            neighbor_cols = X_y[:, neighbors].astype(int) if len(neighbors) > 0 else np.empty((n_y, 0), dtype=int)

            # Conteos de x_N(j) — denominador
            if len(neighbors) > 0:
                unique_W, counts_W_arr = np.unique(neighbor_cols, axis=0, return_counts=True)
                counts_W = {tuple(int(x) for x in row): cnt
                            for row, cnt in zip(unique_W, counts_W_arr)}
            
            # Conteos de xj=1 por config de vecinos — numerador
            neighbor_cols_xi1 = neighbor_cols[node_col == 1]
            if len(neighbors) > 0 and len(neighbor_cols_xi1) > 0:
                unique_1W, counts_1W_arr = np.unique(neighbor_cols_xi1, axis=0, return_counts=True)
                counts_1W = {tuple(int(x) for x in row): cnt
                            for row, cnt in zip(unique_1W, counts_1W_arr)}
            else:
                counts_1W = {}

            prob_table = {}

            if len(neighbors) > 0:
                for w_config, n_w in counts_W.items():
                    n_xi1 = counts_1W.get(w_config, 0)
                    # Laplace smoothing
                    p1 = (n_xi1 + 1) / (n_w + 2)
                    p0 = 1 - p1
                    prob_table[(1,) + w_config] = p1
                    prob_table[(0,) + w_config] = p0
            else:
                # Laplace smoothing for isolated nodes
                n_xi1 = int(node_col.sum())
                p1 = (n_xi1 + 1) / (n_y + 2)
                p0 = 1 - p1
                prob_table[(1,)] = p1
                prob_table[(0,)] = p0

            tables.append(prob_table)

        return tables
    
    def _build_all_prob_tables(self, X_vals: np.ndarray, Y_vals: np.ndarray, graph: np.ndarray) -> list:
        mask_1 = Y_vals[:, 0] == 1
        mask_0 = Y_vals[:, 0] == 0

        tables_y1 = self._build_node_tables(X_vals[mask_1], graph)
        tables_y0 = self._build_node_tables(X_vals[mask_0], graph)

        node_logratios = []
        for j in range(X_vals.shape[1]):
            all_configs = set(tables_y1[j].keys()) | set(tables_y0[j].keys())
            
            logratio_table = {}
            for config in all_configs:
                # Con Laplace smoothing ambas tablas tienen valor para toda config
                # .get con fallback a 0.5 para configs absolutamente no vistas
                p1 = tables_y1[j].get(config, 0.5)
                p0 = tables_y0[j].get(config, 0.5)
                logratio_table[config] = np.log(p1) - np.log(p0)

            # defaultdict(float) → configs nuevas en test retornan 0.0
            node_logratios.append(defaultdict(float, logratio_table))

        return node_logratios
        
    def _fit_bic(self, X_vals, y_vals):
        
        results_bic = self._run_exhaustive_search(X_vals, y_vals)    
        
        optim_NEs, BIC_values = zip(*results_bic)
        
        NE = np.stack(optim_NEs, axis=2) # Shape: (len(c), p, p)
        NE_T = np.moveaxis(NE, -1, -2)
        
        # simetrization
        NE_conserv  = NE & np.moveaxis(NE,-1,-2)
        NE_nconserv = NE | np.moveaxis(NE,-1,-2)
        
        # BIC total por λ: suma sobre todos los nodos
        total_bic = np.sum(np.stack(BIC_values, axis=1), axis=1)  # (len(c),)
        best_c_idx = int(np.argmax(total_bic))
        
        return (NE_conserv[best_c_idx],
                NE_nconserv[best_c_idx])
        
    def _fit_stable(self, X_vals: np.ndarray, Y_vals: np.ndarray,
                PFER: float = 0.1, n_partitions: int = 100,
                pi_min: float = 0.5, pi_max: float = 0.9,
                seed: int = None):
        # number of variables
        p = X_vals.shape[1]
        
        # --- 1. Generate n/2-sized subsamples ---
        rkf = RepeatedKFold(n_splits=2, n_repeats=int(n_partitions / 2), random_state=seed)
        index_list = [train_idx for train_idx, _ in rkf.split(X_vals)]

        # --- 2. Estimate parallely the graph for each k subsample ---
        fun_k = partial(self._run_subsample, X_vals=X_vals, Y_vals=Y_vals,
                        index_list=index_list)
        
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results_subsample = list(tqdm(executor.map(fun_k, range(len(index_list))),
                                total=len(index_list), desc="Stable Selection"))
        else:
            results_subsample = [fun_k(idx) for idx in tqdm(range(len(index_list)), desc="Stable Selection")]

        # NE shape: (n_partitions, len(c), 2, p, p)
        # dim 2: [conserv=0, nconserv=1]
        NE = np.stack(results_subsample, axis=0)

        # --- 3. Expected number of selected edges by the model ---
        # cumsum across subsamples → how many subsamples selected each edge
        # mean over axis=0 → expected frequency of each selected edge → q in the formal equation
        # (last 2 dims) → expected value for each c value for the conserv and nconserv criteria
        q_k = np.sum(
            np.sum(np.cumsum(NE, axis=0) > 0, axis=-1), axis=-1
        ) / 2  # shape: (n_partitions, len(c), 2)

        qhat = np.mean(q_k, axis=0)  # shape: (len(c), 2)

        # --- 4. Bounds PFER ---
        n_edges = p * (p - 1) / 2 # n_edges is p in the formal equation
        q_max_val = np.sqrt(n_edges * PFER * (2 * pi_max - 1))
        q_min_val = np.sqrt(n_edges * PFER * (2 * pi_min - 1))

        assert q_min_val >= 0, \
            f"q_min={q_min_val:.3f} < 0. Invalid range. Increase either PFER or pi_min."
        assert q_max_val < n_edges, \
            f"q_max={q_max_val:.3f} > {n_edges}. Invalid range, Reduce either PFER or pi_max."
        
        # We now keep all c values that do not generate too dense or too sparse graphs, i.e., they are within the range
        accepted_q = (qhat > q_min_val) & (qhat < q_max_val)  # (len(c), 2)
        assert np.any(accepted_q), \
                f"No c value in self.c has qhat between [{q_min_val:.3f}, {q_max_val:.3f}] " \
                f"for any criterion. qhat=\n{qhat}\nUse a larger c values grid."
                
        if not np.any(accepted_q[:, 0]):
            logger.warning("Stable selection: no c value accepted for the conservative criterion. "
                "Conservative graph set None.")
        if not np.any(accepted_q[:, 1]):
            logger.warning("Stable selection: no c value accepted for the non-conservative criterion."
                "Non-conservative graph set to None.")
        # --- 5. For each optimal c, count the edges that are greater than the stability threshold
        freq_selected = np.mean(np.cumsum(NE, axis=0) > 0, axis=0)
        # shape: (len(c), 2, p, p)

        best_c_idx_conserv, best_c_idx_nconserv = self._evaluate_stable_c(
            qhat, freq_selected, accepted_q, n_edges, PFER
        )
        
        coefs_conserv, coefs_nconserv = None, None

        if best_c_idx_conserv is not None:
            q_c = qhat[best_c_idx_conserv, 0]
            threshold_c = (1 + q_c**2 / n_edges / PFER) / 2
            coefs_conserv = freq_selected[best_c_idx_conserv, 0] > threshold_c

        if best_c_idx_nconserv is not None:
            q_nc = qhat[best_c_idx_nconserv, 1]
            threshold_nc = (1 + q_nc**2 / n_edges / PFER) / 2
            coefs_nconserv = freq_selected[best_c_idx_nconserv, 1] > threshold_nc

        return coefs_conserv, coefs_nconserv
    
    def _run_subsample(self, i: int, X_vals: np.ndarray, Y_vals: np.ndarray,
                        index_list: list) -> np.ndarray:
        """
        Estima el grafo en el i-ésimo subsample.
        Retorna shape (len(c), 2, p, p) — [conserv, nconserv].
        """
        idx = index_list[i]
        X_sub = X_vals[idx]
        Y_sub = Y_vals[idx] if Y_vals is not None else None
        p = X_vals.shape[1]

        # Búsqueda exhaustiva en el subsample — siempre single core
        # (el paralelismo ya está en el nivel de subsamples)
        results = [self._compute_ne_i(j, X_sub, Y_sub) for j in range(p)]
        
        adj_rows = [r[0] for r in results]   # best_bic no se usa en stable
        NE = np.stack(adj_rows, axis=2)      # (len(c), p, p)
        NE_T = np.moveaxis(NE, -1, -2)

        NE_conserv  = NE & NE_T   # (len(c), p, p)
        NE_nconserv = NE | NE_T

        return np.stack([NE_conserv, NE_nconserv], axis=1)  # (len(c), 2, p, p)
    
    def _run_exhaustive_search(self, X_vals, y_vals):
        
        p = X_vals.shape[1]
        
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as exec:
                f_partial = partial(self._compute_ne_i, X = X_vals, y = y_vals)
                neighbors_and_bic = list(tqdm(exec.map(f_partial, range(p)), 
                                    total=p, desc="Training Nodes"))
        else:
            neighbors_and_bic = [self._compute_ne_i(i, X_vals, y_vals) for i in tqdm(range(p), desc="Training Nodes")]
            
        return neighbors_and_bic
        
    def _evaluate_stable_c(self, Eqhat, freq_selected, accepted_q,
                            n_edges, PFER) -> tuple:
        """
        Elige c óptimo para conserv y nconserv por separado.
        Retorna (best_c_idx_conserv, best_c_idx_nconserv) — puede ser None si 
        ningún c es aceptado para ese criterio.
        """
        results = []
        
        for criterion_idx in range(accepted_q.shape[1]):  # 0=conserv, 1=nconserv
            accepted_indices = np.where(accepted_q[:, criterion_idx])[0]
            
            if len(accepted_indices) == 0:
                results.append(None)
                continue
            
            discoveries = []
            for i in accepted_indices:
                q_i = Eqhat[i, criterion_idx]
                threshold = (1 + q_i**2 / n_edges / PFER) / 2
                n_stable = np.sum(freq_selected[i, criterion_idx] > threshold)
                discoveries.append((i, n_stable))
            
            discoveries = np.array(discoveries)  # (n_accepted, 2)
            best_idx = int(np.median(
                discoveries[discoveries[:, 1] == np.nanmax(discoveries[:, 1]), 0]
            ))
            results.append(best_idx)
        
        return results[0], results[1]  # (best_c_idx_conserv, best_c_idx_nconserv)
    
    def predict(self, Xtest: pd.DataFrame, level: str = 'global',
            criterion: str = 'conserv',
            Xtrain: pd.DataFrame = None, Ytrain: pd.DataFrame = None) -> dict:
        self._check_if_fitted()

        if criterion == 'conserv':
            graph = self._coefs_conserv
        elif criterion == 'nconserv':
            graph = self._coefs_nconserv
        else:
            raise ValueError("criterion must be either 'conserv' or 'nconserv'")

        if graph is None:
            raise ValueError(f"El grafo para criterion='{criterion}' es None. "
                            "Revisá los warnings de fit().")

        Xtest_vals = Xtest.values if hasattr(Xtest, 'values') else np.asarray(Xtest)

        if Xtrain is not None and Ytrain is not None:
            Xtrain_vals = Xtrain.values if hasattr(Xtrain, 'values') else np.asarray(Xtrain)
            Ytrain_vals = (Ytrain.values if hasattr(Ytrain, 'values') else np.asarray(Ytrain)).reshape(-1, 1)
            tables = self._build_all_prob_tables(Xtrain_vals, Ytrain_vals, graph)
        else:
            tables = (self._node_logratios_conserv if criterion == 'conserv'
                    else self._node_logratios_nconserv)

        return self._predict_from_tables(Xtest_vals, tables, graph, level)
    
    def _predict_from_tables(self, X: pd.DataFrame, tables: list, 
                        graph: np.ndarray, level: str) -> dict:
        """
        Args:
            X_test:  (n, p)
            tables:  list of p defaultdicts — output de build_all_prob_tables
            graph:   (p, p) bool — para extraer componentes conexas
            level:   'node' | 'component' | 'global'
        
        Returns:
            dict con las reducciones pedidas
        """
        n, p = X.shape
        
        # --- First step: Ri (n, p) — estimate the SDR for each node given its neighbors ---
        Ri = np.zeros((n, p))
        for j in range(p):
            neighbors = np.where(graph[j])[0]
            for s in range(n):
                xj  = int(X[s, j])
                x_W = tuple(int(X[s, k]) for k in neighbors)
                config = (xj,) + x_W
                Ri[s, j] = tables[j][config]  # defaultdict → 0.0 if not seen
        
        # --- Second step: sum up the Ri's ---
        R_global = Ri.sum(axis=1)  # (n,)
        
        if level == 'global':
            return {'R': R_global}
        
        elif level == 'node':
            return {'R': R_global, 'Ri': Ri}
        
        elif level == 'component':
            components = self._get_connected_components(graph)  # list of arrays
            R_comp = np.stack(
                [Ri[:, idx].sum(axis=1) for idx in components], 
                axis=1
            )  # (n, K)
            return {'R': R_global, 
                    'Ri': Ri, 'R_components': R_comp, 'components': components}
        
        else:
            raise ValueError("level must be either 'node', 'component' or 'global'")
        
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