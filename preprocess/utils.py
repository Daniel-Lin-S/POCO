"""Preprocess one neural recording before dataset loading.

Input is a neurons-by-time activity array. process_data_matrix returns
a dictionary for a processed NPZ archive: M holds normalised activity,
PC holds principal-component time series when requested, and
valid_indices marks retained neurons. fit_split and fit_* fields record
the train/validation fit rule. Optional FC_<count> arrays hold
cluster-averaged activity. Plot files are written under figures.
"""

import numpy as np
import os

from analysis import plots
from sklearn.cluster import KMeans
from scipy import signal
from sklearn.decomposition import PCA
from configs.config_global import FIG_DIR
from configs.configs import DatasetConfig
from utils.split_utils import chronological_bounds, train_validation_mask

def bandpass(traces, f_l, f_h, sampling_freq=1):
    """
    Apply a bandpass filter to the input traces.

    Parameters:
        traces (np.ndarray): Input traces to be filtered.
        f_l (float): Lower cutoff frequency in Hz.
        f_h (float): Upper cutoff frequency in Hz.
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        filtered (np.ndarray): Filtered traces.

    """
    cut_off_h = f_h * sampling_freq / 2 ## in units of sampling_freq/2
    cut_off_l = f_l * sampling_freq / 2 ## in units of sampling_freq/2
    #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.
    
    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered

FILTER_BANDS = {
    'lowpass': (1e-6, 0.2),
    'highpass': (0.001, 0.99),
    'bandpass': (0.001, 0.2),
}


def filter_split_segments(data: np.ndarray, config: DatasetConfig,
                          mode: str) -> np.ndarray:
    """Filter each chronological split independently.

    Parameters
    ----------
    data : np.ndarray
        Activity with shape ``(neurons, time)``.
    config : DatasetConfig
        Chronological split fractions.
    mode : str
        One of lowpass, highpass, or bandpass.

    Returns
    -------
    np.ndarray
        Filtered activity with shape ``(neurons, time)``.
    """
    if mode not in FILTER_BANDS:
        raise ValueError(f'Unknown filter mode: {mode}.')
    train_end, val_end = chronological_bounds(
        data.shape[1], config.train_split, config.val_split,
        config.test_split
    )
    result = np.empty_like(data)
    for left, right in ((0, train_end),
                        (train_end, val_end),
                        (val_end, data.shape[1])):
        result[:, left:right] = bandpass(
            data[:, left:right], *FILTER_BANDS[mode]
        )
    if not np.isfinite(result).all():
        raise ValueError('Filtered activity contains non-finite values.')
    return result

def run_pca(
    data: np.ndarray,
    fit_mask: np.ndarray,
    exp_name: str = '',
    n_components: int = 2048,
    fig_dir: str = 'preprocess',
    plot_diagnostics: bool = True,
    solver: str = 'full',
    random_state: int | None = None,
) -> np.ndarray:
    """Fit PCA on train and validation samples and transform all samples.

    Parameters
    ----------
    data : np.ndarray
        Activity with shape ``(neurons, time)``.
    fit_mask : np.ndarray
        Boolean mask with shape ``(time,)``, excluding test samples.
    exp_name : str, optional
        Name used for the variance plot, default ''.
    n_components : int, optional
        Requested principal components, default 2048.
    fig_dir : str, optional
        Directory for the variance plot, default 'preprocess'.
    plot_diagnostics : bool, optional
        Save the variance plot, default True.
    solver : str, optional
        PCA solver, default full.
    random_state : int or None, optional
        Seed for a randomized solver, default None.

    Returns
    -------
    np.ndarray
        Transformed activity with shape ``(components, time)``.
    """
    if fit_mask.shape != (data.shape[1],):
        raise ValueError(
            f"Expected fit mask shape {(data.shape[1],)}, "
            f"got {fit_mask.shape}."
        )
    fit_data = data[:, fit_mask]
    if not np.isfinite(fit_data).all():
        raise ValueError("PCA fit data contains non-finite values.")
    n_components = min(n_components, fit_data.shape[1], data.shape[0])
    if n_components < 1:
        raise ValueError("PCA requires at least one fit sample and neuron.")
    pca = PCA(
        n_components=n_components, svd_solver=solver,
        random_state=random_state,
    )
    pca.fit(fit_data.T)
    act = pca.transform(data.T)
    if not np.isfinite(act).all():
        raise ValueError("PCA transformed activity contains non-finite values.")
    if plot_diagnostics:
        print('EV 10:', pca.explained_variance_ratio_[:10])
        cumulated = np.cumsum(pca.explained_variance_ratio_)
        plots.error_plot(
            np.arange(1, len(cumulated) + 1),
            [cumulated],
            x_label='Number of PCs', y_label='Explained variance',
            save_dir=fig_dir,
            fig_name=f'{exp_name}_explained_variance',
            errormode='none',
            mode='errorshade',
            yticks=[0, 1]
        )
    return act.T

def get_clustered_data(
    data: np.ndarray,
    fit_mask: np.ndarray,
    n_clusters: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Cluster neurons using train and validation correlations.

    Parameters
    ----------
    data : np.ndarray
        Activity with shape ``(neurons, time)``.
    fit_mask : np.ndarray
        Boolean mask with shape ``(time,)`` excluding test samples.
    n_clusters : int, optional
        Number of functional clusters, default 500.
    seed : int, optional
        KMeans random seed, default 0.

    Returns
    -------
    np.ndarray
        Cluster activity with shape ``(clusters, time)``.
    """
    corr_matrix = np.corrcoef(data[:, fit_mask])
    print("Correlation Matrix Shape: ", corr_matrix.shape)
    if not np.isfinite(corr_matrix).all():
        raise ValueError(
            'Train/validation correlation matrix is non-finite.'
        )
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(corr_matrix)
    labels = kmeans.labels_
    
    # compute the mean of time traces in each cluster
    clustered_data = np.zeros((n_clusters, data.shape[1]))
    for i in range(n_clusters):
        clustered_data[i] = data[labels == i].mean(axis=0)
    return clustered_data

def plot_delta_F(data, exp_name='', fig_dir='preprocess', suffix=''):
    """
    Plot the difference between adjacent frames
    data: n_neurons * T
    """
    # plot distribution of delta F
    plots.distribution_plot(
        [[(data[:, 1: ] - data[:, : -1]).reshape(-1)]],
        ['All ROIs', ],
        left=-4.95, right=5, 
        interval=0.1,
        x_label='Delta F', y_label='Frequency',
        plot_dir=fig_dir,
        plot_name=f'{exp_name}_delta_F{suffix}',
        errormode='none',
        mode='errorshade',
        legend=False,
    )

def process_data_matrix(
    data: np.ndarray,
    fig_dir: str,
    roi_indices: np.ndarray | None = None,
    divide_baseline: bool = False,
    normalize_mode: str = 'zscore',
    n_clusers: list[int] | None = None,
    plot_max_min: bool = False,
    exp_name: str = '',
    pc_dim: int = 2048,
    plot_window: int = 64,
    filter_mode: str = 'none',
    plot_diagnostics: bool = True,
    pca_solver: str = 'full',
    pca_random_state: int | None = None,
) -> dict[str, np.ndarray | str | float | bool]:
    """Normalize a recording and optionally fit PCA on train and validation.

    Parameters
    ----------
    data : np.ndarray
        Neural activity with shape ``(neurons, time)``.
    fig_dir : str
        Figure subdirectory under the configured artifact root.
    roi_indices : np.ndarray or None, optional
        Boolean neuron mask with shape ``(neurons,)``, default None.
    divide_baseline : bool, optional
        Subtract and divide by the train/validation median, default False.
    normalize_mode : str, optional
        ``zscore``, ``max``, or ``none``, default ``zscore``.
    n_clusers : list[int] or None, optional
        Cluster counts for FC arrays, default None.
    plot_max_min : bool, optional
        Save min/max diagnostics, default False.
    exp_name : str, optional
        Recording label for diagnostic figures, default empty.
    pc_dim : int, optional
        Requested PCA component count, default 2048; zero skips PCA.
    plot_window : int, optional
        Diagnostic trace window in samples, default 64.
    filter_mode : str, optional
        Split-local filter mode, default ``none``.
    plot_diagnostics : bool, optional
        Save diagnostic figures, default True.
    pca_solver : str, optional
        Scikit-learn PCA solver, default ``full``.
    pca_random_state : int or None, optional
        PCA random seed, default None.

    Returns
    -------
    dict[str, np.ndarray | str | float | bool]
        ``M`` has shape ``(neurons, time)``; ``PC`` has shape
        ``(components, time)`` when requested. ``valid_indices`` has
        shape ``(neurons,)``. Split-fit metadata are scalar values.
    """
    data_dict = {}
    n, T = data.shape
    split_config = DatasetConfig()
    fit_mask = train_validation_mask(
        T, split_config.train_split, split_config.val_split,
        split_config.test_split,
    )
    sub_fig_dir = os.path.join(FIG_DIR, fig_dir, exp_name)

    if roi_indices is None:
        roi_indices = np.ones(n, dtype=bool)
    
    if divide_baseline:
        baseline = np.median(data[:, fit_mask], axis=1, keepdims=True)
        roi_indices &= baseline.reshape(-1) > 1
        data = (data - baseline) / (baseline + 1e-6)

    if plot_max_min:
        # min F
        plots.distribution_plot(
            [[data_dict['M'].min(axis=1)], [data_dict['M'].min(axis=1)[roi_indices]]],
            ['All ROIs', 'Filtered ROIs'],
            left=-100, right=100, 
            interval=10,
            x_label='Min F', y_label='Frequency',
            plot_dir=sub_fig_dir,
            plot_name=f'min_F',
            errormode='none'
        )

        # max F
        plots.distribution_plot(
            [[data_dict['M'].max(axis=1)], [data_dict['M'].max(axis=1)[roi_indices]]],
            ['All ROIs', 'Filtered ROIs'],
            left=-50, right=2000, 
            interval=10,
            x_label='Max F', y_label='Frequency',
            plot_dir=sub_fig_dir,
            plot_name=f'max_F',
            errormode='none'
        )

    if normalize_mode == 'zscore':
        fit_data = data[:, fit_mask]
        mu = np.mean(fit_data, axis=1, keepdims=True)
        std = np.std(fit_data, axis=1, keepdims=True)
        normalized = (data - mu) / (std + 1e-6)
    elif normalize_mode == 'max':
        max_abs = np.max(np.abs(data[:, fit_mask]), axis=1, keepdims=True)
        normalized = data / (max_abs + 1e-6)
    else:
        normalized = data
    if not np.isfinite(normalized).all():
        raise ValueError(
            "Normalized activity contains non-finite values after "
            "fitting on train and validation samples."
        )

    if filter_mode != 'none':
        normalized = filter_split_segments(
            normalized, split_config, filter_mode
        )

    print(f"Data shape: {data.shape}, Valid indices: {roi_indices.sum()}")
    data_dict['fit_split'] = 'chronological_train_val'
    data_dict['fit_train_split'] = split_config.train_split
    data_dict['fit_val_split'] = split_config.val_split
    data_dict['fit_filter_isolated'] = True
    data_dict['valid_indices'] = roi_indices
    data_dict['M'] = normalized

    if pc_dim == 0:
        return data_dict
    
    if not roi_indices.all():
        normalized = normalized[roi_indices]
    if normalized.shape[0] == 0:
        raise ValueError("No valid neurons remain for PCA.")
    if plot_diagnostics:
        plot_delta_F(normalized, fig_dir=sub_fig_dir, exp_name=exp_name)
    data_dict['PC'] = run_pca(
        normalized, fit_mask, fig_dir=sub_fig_dir,
        exp_name=exp_name, n_components=pc_dim,
        plot_diagnostics=plot_diagnostics,
        solver=pca_solver, random_state=pca_random_state,
    )
    if plot_diagnostics:
        plot_delta_F(
            data_dict['PC'], fig_dir=sub_fig_dir,
            suffix='_PC', exp_name=exp_name,
        )

        # plot first 10 PCs
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 10))
        for i in range(10):
            ax = plt.subplot(10, 2, i * 2 + 1)
            ax.plot(data_dict['PC'][i])
            ax.title.set_text(f'PC {i}, full trace')
            # plot a random plot_window-step window
            ax = plt.subplot(10, 2, i * 2 + 2)
            t = np.random.randint(0, data_dict['PC'].shape[1] - plot_window)
            ax.plot(data_dict['PC'][i, t: t + plot_window])
            ax.title.set_text(f'PC {i}, {plot_window}-step window')
        plt.tight_layout()
        plt.savefig(os.path.join(sub_fig_dir, f'{exp_name}_first_10_PCs.pdf'))
        plt.close()

        # plot 10 random neurons
        plt.figure(figsize=(7, 10))
        for i in range(10):
            idx = np.random.randint(0, normalized.shape[0])
            ax = plt.subplot(10, 2, i * 2 + 1)
            ax.plot(normalized[idx])
            ax.title.set_text(f'Neuron {idx}, full trace')
            # plot a random plot_window-step window
            ax = plt.subplot(10, 2, i * 2 + 2)
            t = np.random.randint(0, normalized.shape[1] - plot_window)
            ax.plot(normalized[idx, t: t + plot_window])
            ax.title.set_text(f'Neuron {idx}, {plot_window}-step window')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                sub_fig_dir, f'{exp_name}_10_random_neurons.pdf'
            )
        )
        plt.close()

    for n_cluster in n_clusers or ():
        data_dict[f'FC_{n_cluster}'] = get_clustered_data(
            normalized, fit_mask, n_cluster
        )
    return data_dict
