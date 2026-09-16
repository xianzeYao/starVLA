"""Regression tests for the process-wide OmegaConf compatibility patch."""

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_with_tracker(body, wrapped):
    # Isolate the global monkey patch from other tests and their import order.
    script = (
        "from dataclasses import dataclass\n"
        "from enum import Enum\n"
        "from omegaconf import OmegaConf, SCMode\n"
        "from omegaconf.errors import MissingMandatoryValue\n"
        "from starVLA.training.trainer_utils.config_tracker import wrap_config\n"
        f"wrap = wrap_config if {wrapped!r} else lambda cfg: cfg\n"
        + body
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("wrapped", [False, True])
def test_to_object_instantiates_structured_configs_after_tracking_import(wrapped):
    run_with_tracker('''
@dataclass
class Settings:
    width: int = 8
    alias: int = "${width}"
cfg = wrap(OmegaConf.structured(Settings))
result = OmegaConf.to_object(cfg)
assert isinstance(result, Settings)
assert result.width == 8 and result.alias == 8
''', wrapped)


@pytest.mark.parametrize("wrapped", [False, True])
def test_missing_value_policy_survives_tracking_patch(wrapped):
    run_with_tracker('''
cfg = wrap(OmegaConf.create({"required": "???"}))
assert OmegaConf.to_container(cfg, throw_on_missing=False) == {"required": "???"}
for convert in (OmegaConf.to_object,
                lambda c: OmegaConf.to_container(c, throw_on_missing=True)):
    try:
        convert(cfg)
    except MissingMandatoryValue:
        pass
    else:
        raise AssertionError("mandatory-value error was swallowed")
''', wrapped)


@pytest.mark.parametrize("wrapped", [False, True])
def test_conversion_preserves_resolution_and_enum_options(wrapped):
    run_with_tracker('''
class Choice(Enum):
    RGB = "rgb"
cfg = wrap(OmegaConf.create({"size": 8, "alias": "${size}", "mode": Choice.RGB}))
assert OmegaConf.to_container(cfg, resolve=False, enum_to_str=True,
                              throw_on_missing=True) == {
    "size": 8, "alias": "${size}", "mode": "RGB"}
assert OmegaConf.to_container(cfg, resolve=True, enum_to_str=False,
                              throw_on_missing=True) == {
    "size": 8, "alias": 8, "mode": Choice.RGB}
# Keep the tracker's existing resolve=True default for established callers.
assert OmegaConf.to_container(cfg)["alias"] == 8
''', wrapped)


@pytest.mark.parametrize("wrapped", [False, True])
def test_structured_constructor_typeerror_is_not_retried_as_dict(wrapped):
    run_with_tracker('''
@dataclass
class InvalidSettings:
    width: int = 8
    def __post_init__(self):
        raise TypeError("invalid structured settings")
cfg = wrap(OmegaConf.structured(InvalidSettings))
try:
    OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE)
except TypeError as exc:
    assert "invalid structured settings" in str(exc)
else:
    raise AssertionError("constructor TypeError was swallowed")
''', wrapped)
