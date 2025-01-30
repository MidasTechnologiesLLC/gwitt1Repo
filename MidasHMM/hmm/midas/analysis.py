# midas/analysis.py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np

class MarketRegimeAnalysis:
    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.states = model.predict(features)
        
    def plot_regimes(self, prices: pd.Series):
        plt.figure(figsize=(15, 8))
        palette = sns.color_palette("husl", n_colors=self.model.n_components)
        
        for state in range(self.model.n_components):
            mask = self.states == state
            plt.scatter(prices.index[mask], prices[mask], 
                       color=palette[state], s=10, label=f'Regime {state}')
            
        plt.title("Market Regime Visualization")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        return plt
    
    def plot_transition_matrix(self):
        transmat = self.model.transmat_
        plt.figure(figsize=(10, 8))
        sns.heatmap(transmat, annot=True, fmt=".2f", cmap="Blues",
                   xticklabels=range(transmat.shape[0]),
                   yticklabels=range(transmat.shape[1]))
        plt.title("State Transition Probabilities")
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        return plt
    
    def plot_state_durations(self):
        state_changes = np.diff(self.states, prepend=self.states[0])
        change_points = np.where(state_changes != 0)[0]
        durations = np.diff(np.append(change_points, len(self.states)))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(durations, bins=30, kde=True)
        plt.title("Regime Duration Distribution")
        plt.xlabel("Duration (Bars)")
        plt.ylabel("Frequency")
        return plt
