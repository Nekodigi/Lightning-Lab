from os import cpu_count

from torch.utils.data import DataLoader, Dataset

from confs.conf import BaseTrainerCfg

WORKER_DIV = 1


def make_dataloader(dataset: Dataset, cfg: BaseTrainerCfg, shuffle: bool = True):
    cpu_cnt = cpu_count()
    assert cpu_cnt is not None
    return DataLoader(
        dataset,  # type: ignore
        batch_size=cfg.batch_size,
        num_workers=1 if cfg.dry else cpu_cnt // (2 * WORKER_DIV),
        prefetch_factor=1 if cfg.dry else cpu_cnt // (1 * WORKER_DIV),
        persistent_workers=False if cfg.dry else True,
        pin_memory=False if cfg.dry else True,
        shuffle=shuffle,
    )
