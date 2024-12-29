from scipy import stats, optimize
import numpy as np
from typing import Tuple, Optional, List

class ModificationRateEstimator:
    def __init__(self, bandwidth_method='scott', overlap_tolerance=0.05):
        self.bandwidth_method = bandwidth_method
        self.overlap_tolerance = overlap_tolerance
        
    def fit_kde(self, data: np.ndarray) -> stats.gaussian_kde:
        return stats.gaussian_kde(data, bw_method=self.bandwidth_method)
        
    def plot_fit(self, control_kde: stats.gaussian_kde, modified_kde: stats.gaussian_kde, 
             x_grid: np.ndarray, weight: float, title: str = "") -> None:
        import matplotlib.pyplot as plt
        
        control_pdf = control_kde.evaluate(x_grid)
        modified_pdf = modified_kde.evaluate(x_grid)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, modified_pdf, 'b-', label='Modified Sample')
        plt.plot(x_grid, weight * control_pdf, 'r-', label=f'Control (weight={weight:.3f})')
        plt.fill_between(x_grid, modified_pdf, weight * control_pdf, 
                        where=(weight * control_pdf > modified_pdf), 
                        color='red', alpha=0.3, label='Overlap')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def estimate_modification_rate(self, 
                                 control_data: np.ndarray, 
                                 modified_data: np.ndarray,
                                 control_kdes: Optional[List[stats.gaussian_kde]] = None) -> Tuple[float, float]:
        x_grid = np.linspace(min(modified_data), max(modified_data), 1000)
        modified_kde = self.fit_kde(modified_data)
        modified_pdf = modified_kde.evaluate(x_grid)
        
        if control_kdes is None:
            control_kde = self.fit_kde(control_data)
            control_kdes = [control_kde]
        
        def objective(weight: float, control_kde: stats.gaussian_kde) -> float:
            control_pdf = control_kde.evaluate(x_grid)
            residuals = np.maximum(0, weight * control_pdf - modified_pdf - self.overlap_tolerance)
            return np.sum(residuals**2)
            
        weights = []
        for control_kde in control_kdes:
            res = optimize.minimize_scalar(
                lambda w: objective(w, control_kde),
                bounds=(0, 1),
                method='bounded'
            )
            weights.append(res.x)
            self.plot_fit(control_kde, modified_kde, x_grid, res.x, f"Modification Rate: {1-res.x:.2%}")
            
        return 1 - np.mean(weights), np.std(weights) if len(weights) > 1 else 0.0

def estimate_rates_for_position(position_data: dict) -> dict:
    estimator = ModificationRateEstimator()
    control_sample = position_data["primary_control"][0][1]
    control_signal = control_sample["signal"]
    
    control_kdes = None
    if position_data["secondary_control"]:
        secondary_control = position_data["secondary_control"][0][1]
        control_kdes = [
            estimator.fit_kde(control_signal),
            estimator.fit_kde(secondary_control["signal"])
        ]
    
    results = {}
    for sample, metrics in position_data["modified_samples"]:
        rate, uncertainty = estimator.estimate_modification_rate(
            control_signal,
            metrics["signal"],
            control_kdes
        )
        results[sample.name] = {"rate": rate, "uncertainty": uncertainty}
    
    return results