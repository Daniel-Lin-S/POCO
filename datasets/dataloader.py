"""Create train, validation, and test iterators for POCO fits.

Input is a resolved dataset configuration and phase. Output is one DataLoader
per dataset label. Training drops partial batches to keep examples per step
fixed; validation and test retain every window.
"""

from torch.utils.data import DataLoader

from configs.configs import NeuralPredictionConfig, DatasetConfig
import datasets

class DatasetIters(object):

    def __init__(self, config: NeuralPredictionConfig,
                 phase: str) -> None:
        """Build phase-specific iterators for all configured datasets.

        Parameters
        ----------
        config : NeuralPredictionConfig
            Resolved dataset and batch settings.
        phase : str
            One of train, val, or test.
        """
        if phase not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown dataset phase: {phase}.')
        if type(config.dataset) is list or type(config.dataset) is tuple:
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            dataset_list = [config.dataset]
        else:
            raise ValueError(
                f'Expected dataset names, got {type(config.dataset)}.'
            )

        self.num_datasets = len(dataset_list)
        self.phase = phase
        if self.num_datasets == 0:
            raise ValueError('No datasets were configured for this fit.')
        if (len(config.mod_w) != self.num_datasets or
                len(config.dataset_label) != self.num_datasets):
            raise ValueError(
                'Dataset labels and weights must match dataset count '
                f'{self.num_datasets}.'
            )

        self.data_iters = []
        self.iter_lens = []
        self.min_iter_len = None

        self.data_loaders = []
        self.input_sizes = []
        self.unit_types = []

        for dataset, label in zip(dataset_list, config.dataset_label):
            data_loader, input_size, unit_type = init_single_dataset(dataset, phase, config.dataset_config[label])
            self.data_loaders.append(data_loader)
            self.input_sizes.append(input_size)
            self.unit_types.append(unit_type)

        self.reset()

    def reset(self) -> None:
        """Start a fresh epoch of phase-specific batches."""
        self.data_iters = []
        self.iter_lens = []
        for data_l in self.data_loaders:
            data_iter = iter(data_l)
            self.data_iters.append(data_iter)
            length = len(data_iter)
            if length == 0:
                raise ValueError(
                    f'{self.phase} dataset has no full batches.'
                )
            self.iter_lens.append(length)
        self.min_iter_len = min(self.iter_lens)

    def get_baselines(self, key: str = 'session_copy_mse') -> list:
        """Return per-session baselines for validation or test."""
        if self.phase == 'train':
            raise ValueError(
                'Baselines require validation or test data.'
            )
        all_baselines = []
        for loader in self.data_loaders:
            all_baselines.extend(loader.dataset.baseline[key])
        return all_baselines

def init_single_dataset(dataset_name: str, phase: str,
                        config: DatasetConfig) -> tuple:
    """Construct one phase-specific dataset and batch loader.

    Parameters
    ----------
    dataset_name : str
        Registered recording family.
    phase : str
        One of train, val, or test.
    config : DatasetConfig
        Resolved session and batch settings.

    Returns
    -------
    tuple
        DataLoader, per-session unit counts, and unit-type arrays.
    """
    train_flag = phase == 'train'
    if config.batch_size is None or config.batch_size < 1:
        raise ValueError(
            f'Expected a positive batch size, got {config.batch_size}.'
        )

    if dataset_name == 'zebrafish':
        dataset = datasets.Zebrafish(config, phase=phase)
    elif dataset_name == 'zebrafishahrens':
        dataset = datasets.ZebrafishAhrens(config, phase=phase)
    elif dataset_name == 'simulation':
        dataset = datasets.Simulation(config, phase=phase)
    elif dataset_name == 'celegans':
        dataset = datasets.Celegans(config, phase=phase)
    elif dataset_name == 'celegansflavell':
        dataset = datasets.CelegansFlavell(config, phase=phase)
    elif dataset_name == 'mice':
        dataset = datasets.Mice(config, phase=phase)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}.')

    data_loader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=train_flag or config.shuffle_test,
        drop_last=train_flag, num_workers=config.num_workers,
        collate_fn=dataset.collate_fn
    )

    return data_loader, dataset.input_size, dataset.unit_types
