"""Registry entries for the ARX dual-arm CoT dataset."""

from starVLA.dataloader.arx_cot_lerobot_datasets import ArxCoTDataConfig


ROBOT_TYPE_CONFIG_MAP = {
    "arx_cot": ArxCoTDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    "arx_cot_sweep": [
        ("arx_cot_sweep_lerobot", 1.0, "arx_cot"),
    ],
    "arx_cot_sweep_v2": [
        ("arx_cot_sweep_v2_lerobot", 1.0, "arx_cot"),
    ],
    "cot_cube_v3": [
        ("cot_cube_v3_lerobot", 1.0, "arx_cot"),
    ],
    "arx_cot_box": [
        ("arx_cot_box", 1.0, "arx_cot"),
    ],
}
