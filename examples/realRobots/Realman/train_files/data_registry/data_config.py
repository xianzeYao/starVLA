"""Registry entry for the Realman hanger CoT dataset."""

from starVLA.dataloader.realman_cot_lerobot_datasets import RealmanCoTDataConfig


ROBOT_TYPE_CONFIG_MAP = {"realman_cot": RealmanCoTDataConfig()}
DATASET_NAMED_MIXTURES = {
    "realman_cot_hanger": [("realman_cot_hanger", 1.0, "realman_cot")],
}
