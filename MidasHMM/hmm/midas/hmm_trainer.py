# midas/hmm_trainer.py
import numpy as np
from hmmlearn import hmm
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score
import logging

class HMMTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.best_model = None
        
    def _calculate_bic(self, model, X):
        """Bayesian Information Criterion"""
        log_likelihood = model.score(X)
        n_params = model.n_components * (model.n_components - 1) + \
                  model.n_components * X.shape[1] * 2  # Means and variances
        return -2 * log_likelihood + n_params * np.log(X.shape[0])
        
    def train(self, features: pd.DataFrame):
        best_score = -np.inf
        best_model = None
        
        # Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for n_components in range(*self.config['n_states_range']):
            try:
                fold_scores = []
                for train_idx, test_idx in tscv.split(features):
                    X_train = features.iloc[train_idx]
                    X_test = features.iloc[test_idx]
                    
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type="diag",
                        n_iter=1000,
                        random_state=42
                    )
                    model.fit(X_train)
                    
                    # Score using both likelihood and regime persistence
                    log_likelihood = model.score(X_test)
                    states = model.predict(X_test)
                    persistence = np.mean([len(list(g)) for _, g in groupby(states)])
                    score = log_likelihood + persistence
                    
                    fold_scores.append(score)
                
                avg_score = np.mean(fold_scores)
                bic = self._calculate_bic(model, features)
                self.logger.info(f"States {n_components}: BIC={bic:.2f}, Score={avg_score:.2f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    self.best_model = model
                    
            except Exception as e:
                self.logger.error(f"Failed training {n_components} states: {str(e)}")
                
        if not self.best_model:
            raise RuntimeError("No valid models trained")
            
        self.best_model.fit(features)  # Final training on full dataset
        return self.best_model

    def save_model(self, path: str):
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'config': self.config
        }, path)
