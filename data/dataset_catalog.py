from data.input_ksdd import KSDDDataset
from data.input_dagm import DagmDataset
from data.input_steel import SteelDataset
from data.input_ksdd2 import KSDD2Dataset
from config import Config
from typing import Optional
import tensorflow as tf

def get_dataset(kind: str, cfg: Config) -> Optional[tf.data.Dataset]:
    if kind == "VAL" and not cfg.VALIDATE:
        return None
    if kind == "VAL" and cfg.VALIDATE_ON_TEST:
        kind = "TEST"
    if cfg.DATASET == "KSDD":
        ds = KSDDDataset.new(cfg.DATASET_PATH, cfg, kind)
    elif cfg.DATASET == "DAGM":
        ds = DagmDataset.new(cfg.DATASET_PATH, cfg, kind)
    elif cfg.DATASET == "STEEL":
        ds = SteelDataset.new(cfg.DATASET_PATH, cfg, kind)
    elif cfg.DATASET == "KSDD2":
        ds = KSDD2Dataset.new(cfg.DATASET_PATH, cfg, kind)
    else:
        raise Exception(f"Unknown dataset {cfg.DATASET}")

    shuffle = kind == "TRAIN"
    batch_size = cfg.BATCH_SIZE if kind == "TRAIN" else 1

    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
