import logging
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


class BenchmarkProfile(Enum):
    """Available benchmark profiles for different measurement focuses"""
    FHE_COMPUTATION = auto()  # encryption/decryption/homomorphic times
    FHE_BANDWIDTH = auto()    # ciphertext and key sizes, network transfer
    MODEL_PERFORMANCE = auto() # accuracy, loss, convergence
    MEMORY_USAGE = auto()     # memory consumption
    ALL = auto()              # enable all metrics


class BenchmarkManager:
    """Collects benchmark events and provides export + plotting helpers.

    The manager stores timestamped metric events and supports filtering
    by profile. Seaborn is used when available, otherwise matplotlib
    fallbacks are used for plotting.
    
    Custom metrics can be registered using register_metric() and will be
    automatically evaluated during benchmark collection if their triggers match.
    """

    def __init__(self, profiles: Optional[List[BenchmarkProfile]] = None):
        self.logs: List[Dict] = []
        self.current_round: int = 0
        self.custom_metrics: Dict[str, Dict] = {}
        self.active_profiles: Set[BenchmarkProfile] = (
            {BenchmarkProfile.ALL} if profiles is None else set(profiles)
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize profile metrics
        self.profile_metrics: Dict[BenchmarkProfile, Set[str]] = {
            BenchmarkProfile.FHE_COMPUTATION: {
                'Encryption Time', 'Decryption Time', 'Homomorphic Operation Time',
                'Key Generation Time', 'Context Generation Time'
            },
            BenchmarkProfile.FHE_BANDWIDTH: {
                'Ciphertext Size', 'Public Key Size', 'Private Key Size',
                'Evaluation Key Size', 'Network Transfer Size'
            },
            BenchmarkProfile.MODEL_PERFORMANCE: {
                'Model Accuracy', 'Model Loss', 'Training Time', 'Local Train Time',
                'Convergence Rate', 'Client Update Size'
            },
            BenchmarkProfile.MEMORY_USAGE: {
                'Peak Memory Usage', 'Average Memory Usage',
                'Memory Growth Rate', 'Garbage Collection Time'
            }
        }
        
    def register_metric(self, name: str, fn: Callable, profile: BenchmarkProfile,
                       trigger: str = 'round_end', unit: str = '', 
                       requires: List[str] = None, **kwargs):
        """Register a custom metric function to be evaluated automatically.
        
        Args:
            name: The name of the metric (used in plots and exports)
            fn: The metric function that computes the value
            profile: Which benchmark profile this metric belongs to
            trigger: When to evaluate ('round_end', 'experiment_end', etc.)
            unit: Optional unit for the metric value
            requires: List of required arguments for the metric function
            **kwargs: Additional static arguments to pass to the metric function
        """
        if name in self.custom_metrics:
            self.logger.warning(f"Overwriting existing metric: {name}")
            
        self.custom_metrics[name] = {
            'function': fn,
            'profile': profile,
            'trigger': trigger,
            'unit': unit,
            'requires': requires or [],
            'static_args': kwargs
        }
        # Add to profile metrics mapping
        if profile not in self.profile_metrics:
            self.profile_metrics[profile] = set()
        self.profile_metrics[profile].add(name)
        
        self.logger.info(f"Registered custom metric: {name} for profile {profile.name}")
        
    def evaluate_custom_metrics(self, trigger: str, **kwargs):
        """Evaluate all custom metrics that match the given trigger.
        
        Args:
            trigger: The trigger event ('round_end', 'experiment_end', etc.)
            **kwargs: Arguments available to the metric functions
        """
        for name, metric in self.custom_metrics.items():
            if metric['trigger'] != trigger:
                continue
                
            # Check if we have all required arguments
            if not all(req in kwargs for req in metric['requires']):
                missing = [req for req in metric['requires'] if req not in kwargs]
                self.logger.warning(f"Skipping metric {name}, missing args: {missing}")
                continue
                
            try:
                # Prepare arguments for the metric function
                fn_kwargs = {k: kwargs[k] for k in metric['requires']}
                fn_kwargs.update(metric['static_args'])
                
                # Evaluate the metric
                value = metric['function'](**fn_kwargs)
                self.log_event('custom', name, value, metric['unit'])
            except Exception as e:
                self.logger.warning(f"Failed to evaluate metric {name}: {str(e)}")

        # metrics grouped by profile
        self.profile_metrics: Dict[BenchmarkProfile, Set[str]] = {
            BenchmarkProfile.FHE_COMPUTATION: {
                'Encryption Time', 'Decryption Time', 'Homomorphic Operation Time',
                'Key Generation Time', 'Context Generation Time'
            },
            BenchmarkProfile.FHE_BANDWIDTH: {
                'Ciphertext Size', 'Public Key Size', 'Private Key Size',
                'Evaluation Key Size', 'Network Transfer Size'
            },
            BenchmarkProfile.MODEL_PERFORMANCE: {
                'Model Accuracy', 'Model Loss', 'Training Time',
                'Convergence Rate', 'Client Update Size'
            },
            BenchmarkProfile.MEMORY_USAGE: {
                'Peak Memory Usage', 'Average Memory Usage',
                'Memory Growth Rate', 'Garbage Collection Time'
            }
        }

        self.logger.info(
            f"BenchmarkManager initialized with profiles: {[p.name for p in self.active_profiles]}"
        )

    # ---- basic operations -------------------------------------------------
    def set_round(self, round_num: int):
        self.current_round = int(round_num)

    def should_log_metric(self, metric_name: str) -> bool:
        """Decide whether to log a metric based on active profiles.

        Matching is flexible: exact match, substring match, or token overlap
        will allow common names like 'Final Accuracy' to match 'Model Accuracy'.
        """
        if BenchmarkProfile.ALL in self.active_profiles:
            return True
        m_low = metric_name.lower()
        # simple tokenization helper
        def tokens(s: str):
            return set([t for t in ''.join(c if c.isalnum() else ' ' for c in s.lower()).split() if t])

        m_tokens = tokens(metric_name)
        for profile in self.active_profiles:
            if profile == BenchmarkProfile.ALL:
                continue
            for candidate in self.profile_metrics.get(profile, set()):
                c_low = candidate.lower()
                if m_low == c_low:
                    return True
                if c_low in m_low or m_low in c_low:
                    return True
                # token overlap
                if m_tokens & tokens(candidate):
                    return True
        return False

    def log_event(self, component_id: str, metric_name: str, value: float, unit: str = '', tags: Dict[str, str] = None):
        """Record a timestamped metric event (if enabled by profile).

        tags (optional) are merged into the event dict to allow grouping.
        """
        if not self.should_log_metric(metric_name):
            return

        entry = {
            'timestamp': datetime.now(),
            'round': self.current_round,
            'component_id': component_id,
            'metric': metric_name,
            'value': float(value),
            'unit': unit or ''
        }
        if tags:
            entry.update(tags)

        self.logs.append(entry)

    # ---- introspection & export -------------------------------------------
    def get_available_metrics(self) -> Dict[BenchmarkProfile, Set[str]]:
        return {k: set(v) for k, v in self.profile_metrics.items()}

    def get_active_profiles(self) -> Set[BenchmarkProfile]:
        return set(self.active_profiles)

    def add_profile(self, profile: BenchmarkProfile):
        self.active_profiles.add(profile)
        self.logger.info(f"Added benchmark profile: {profile.name}")

    def remove_profile(self, profile: BenchmarkProfile):
        if profile in self.active_profiles and profile != BenchmarkProfile.ALL:
            self.active_profiles.remove(profile)
            self.logger.info(f"Removed benchmark profile: {profile.name}")

    def get_benchmark_data(self, filter_profile: Optional[BenchmarkProfile] = None) -> pd.DataFrame:
        if not self.logs:
            return pd.DataFrame()
        df = pd.DataFrame(self.logs)
        if filter_profile and filter_profile != BenchmarkProfile.ALL:
            metrics = self.profile_metrics.get(filter_profile, set())
            df = df[df['metric'].isin(metrics)]
        return df

    def export_to_csv(self, filepath: Union[str, Path], filter_profile: Optional[BenchmarkProfile] = None):
        path = Path(filepath)
        df = self.get_benchmark_data(filter_profile)
        if df.empty:
            self.logger.warning("No benchmark data to export")
            return
        df.to_csv(path, index=False)
        self.logger.info(f"Exported benchmark data to {path}")

    # ---- plotting helpers -------------------------------------------------
    def setup_plot_style(self):
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'lines.linewidth': 1.5
        })

    def plot_time_series(self, metric_names: Union[str, List[str]], components: Optional[List[str]] = None,
                         rolling_window: Optional[int] = None, save_path: Optional[Union[str, Path]] = None):
        self.setup_plot_style()
        df = self.get_benchmark_data()
        if df.empty:
            self.logger.warning("No data to plot (time series)")
            return
        if isinstance(metric_names, str):
            metric_names = [metric_names]

        if components:
            df = df[df['component_id'].isin(components)]

        fig, ax = plt.subplots()
        for metric in metric_names:
            mdf = df[df['metric'] == metric].sort_values('timestamp')
            if mdf.empty:
                continue
            # Ensure timestamps are datetimes for proper plotting
            x = pd.to_datetime(mdf['timestamp'])
            series = mdf['value']
            if rolling_window and rolling_window > 1:
                series = series.rolling(rolling_window, min_periods=1).mean()
            y = series.values
            # Plot with square markers and connecting lines for classical look
            ax.plot(x, y, marker='s', linestyle='-', label=f"{metric} ({mdf['unit'].iloc[0]})", markersize=6)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Metric Time Series')
        # Only show legend if there are labeled artists (prevents matplotlib warning)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        fig.autofmt_xdate()
        if save_path:
            fig.savefig(str(save_path))
        plt.close(fig)

    def plot_round_comparison(self, metric_name: str, group_by: str = 'component_id', plot_type: str = 'box',
                              save_path: Optional[Union[str, Path]] = None):
        self.setup_plot_style()
        df = self.get_benchmark_data()
        if df.empty:
            self.logger.warning("No data to plot (round comparison)")
            return
        mdf = df[df['metric'] == metric_name]
        if mdf.empty:
            self.logger.warning(f"No data for metric: {metric_name}")
            return

        fig, ax = plt.subplots()
        # If requested or data contains one sample per round, show a mean-per-round line with markers
        counts = mdf.groupby('round').size()
        use_line = (plot_type == 'line') or (counts.max() <= 1)

        if use_line:
            means = mdf.groupby('round')['value'].mean()
            rounds = list(means.index)
            ax.plot(rounds, means.values, marker='s', linestyle='-', markersize=6)
            ax.set_xticks(rounds)
        else:
            if HAS_SEABORN:
                if plot_type == 'violin':
                    sns.violinplot(data=mdf, x='round', y='value', hue=group_by, ax=ax)
                else:
                    sns.boxplot(data=mdf, x='round', y='value', hue=group_by, ax=ax)
            else:
                # basic matplotlib boxplots grouped by round
                grouped = [group['value'].values for _, group in mdf.groupby('round')]
                labels = [str(r) for r in sorted(mdf['round'].unique())]
                ax.boxplot(grouped, labels=labels)

        ax.set_xlabel('Round')
        ax.set_ylabel(f"{metric_name} ({mdf['unit'].iloc[0] if 'unit' in mdf.columns else ''})")
        ax.set_title(f"{metric_name} Distribution by Round")
        if save_path:
            fig.savefig(str(save_path))
        plt.close(fig)

    def plot_profile_summary(self, profile: BenchmarkProfile, plot_type: str = 'bar',
                             save_path: Optional[Union[str, Path]] = None):
        self.setup_plot_style()
        df = self.get_benchmark_data(profile)
        if df.empty:
            self.logger.warning(f"No data for profile: {profile.name}")
            return

        summary = df.groupby('metric').agg(mean_value=('value', 'mean'), std_value=('value', 'std')).reset_index()
        fig, ax = plt.subplots()
        if plot_type == 'radar' and len(summary) > 2:
            # simple radar plot implementation
            labels = summary['metric'].tolist()
            values = summary['mean_value'].values
            angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            ax = plt.subplot(111, projection='polar')
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
        else:
            ax.bar(summary['metric'], summary['mean_value'], yerr=summary['std_value'])
            plt.xticks(rotation=45, ha='right')

        if save_path:
            fig.savefig(str(save_path))
        plt.close(fig)

    def plot_correlation_matrix(self, profile: Optional[BenchmarkProfile] = None, save_path: Optional[Union[str, Path]] = None):
        self.setup_plot_style()
        df = self.get_benchmark_data(profile)
        if df.empty:
            self.logger.warning('No data for correlation plot')
            return
        # Use round as the aggregation index to ensure metrics collected per round align
        # This increases the chance of having comparable values across metrics for correlation
        if 'round' in df.columns:
            pivot = df.pivot_table(index='round', columns='metric', values='value', aggfunc='mean')
        else:
            pivot = df.pivot_table(index='timestamp', columns='metric', values='value', aggfunc='mean')
        corr = pivot.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        if HAS_SEABORN:
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        else:
            im = ax.imshow(corr.values, cmap='coolwarm', aspect='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='k')

        if save_path:
            fig.savefig(str(save_path))
        plt.close(fig)
