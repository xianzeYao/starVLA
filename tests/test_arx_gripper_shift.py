from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.realRobots.ARX.train_files.shift_arx_grippers import (
    install_staged_migration,
    migrate_dataset,
    verify_migration,
)


POSITION_FIELDS = (
    "observation.state",
    "observation.eef",
    "observation.eef_tcp",
    "action",
)


def _vector(start: float) -> list[np.float32]:
    return list(np.arange(start, start + 14, dtype=np.float32))


def _statistics(*, include_quantiles: bool) -> dict:
    result = {}
    for field_index, field in enumerate(POSITION_FIELDS):
        base = float(field_index * 20)
        entry = {
            "min": _vector(base),
            "max": _vector(base + 2),
            "mean": _vector(base + 1),
            "std": [np.float32(0.5)] * 14,
            "count": [2],
        }
        if include_quantiles:
            entry["q01"] = _vector(base + 0.25)
            entry["q99"] = _vector(base + 1.75)
        if field != "observation.state":
            state_bases = {
                "min": 0,
                "max": 2,
                "mean": 1,
                "q01": 0.25,
                "q99": 1.75,
            }
            for statistic, state_base in state_bases.items():
                if statistic not in entry:
                    continue
                for index in (6, 13):
                    entry[statistic][index] = np.float32(state_base + index)
        result[field] = entry
    result["observation.qvel"] = {
        "min": _vector(-50),
        "max": _vector(-48),
        "mean": _vector(-49),
        "std": [np.float32(0.25)] * 14,
        "count": [2] * 14,
    }
    return result


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_fixture(root: Path, *, malformed: bool = False) -> None:
    data_dir = root / "data/chunk-000"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    arrays = {}
    for field_index, field in enumerate(POSITION_FIELDS):
        width = 13 if malformed and field == "action" else 14
        rows = [
            list(
                np.arange(
                    field_index * 20,
                    field_index * 20 + width,
                    dtype=np.float32,
                )
            ),
            list(
                np.arange(
                    field_index * 20 + 100,
                    field_index * 20 + 100 + width,
                    dtype=np.float32,
                )
            ),
        ]
        if field != "observation.state":
            for row_index, row in enumerate(rows):
                for gripper_index in (6, 13):
                    if gripper_index < width:
                        row[gripper_index] = np.float32(
                            gripper_index + 100 * row_index
                        )
        arrays[field] = pa.array(rows, type=pa.list_(pa.float32()))
    arrays["observation.qvel"] = pa.array(
        [_vector(-50), _vector(-36)], type=pa.list_(pa.float32())
    )
    arrays["timestamp"] = pa.array([0.0, 0.1], type=pa.float64())
    arrays["frame_index"] = pa.array([0, 1], type=pa.int64())
    table = pa.table(arrays).replace_schema_metadata({b"fixture": b"preserve"})
    pq.write_table(table, data_dir / "episode_000000.parquet", compression="snappy")

    episode = {
        "episode_index": 0,
        "stats": _statistics(include_quantiles=False),
    }
    (meta_dir / "episodes_stats.jsonl").write_text(
        json.dumps(_json_ready(episode), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    global_stats = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": _statistics(include_quantiles=True),
    }
    (meta_dir / "stats_gr00t.json").write_text(
        json.dumps(_json_ready(global_stats), indent=4) + "\n",
        encoding="utf-8",
    )


class ArxGripperShiftTest(unittest.TestCase):
    def test_migration_shifts_only_gripper_values_and_location_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            migrated = Path(temp_dir) / "migrated"
            _write_fixture(source)

            migration = migrate_dataset(source, migrated, offset=6.2)
            verification = verify_migration(source, migrated, offset=6.2)

            self.assertEqual(migration.parquet_files, 1)
            self.assertEqual(migration.rows, 2)
            self.assertEqual(verification.parquet_files, 1)
            self.assertEqual(verification.rows, 2)
            original = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
            shifted = pq.read_table(migrated / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(original.schema, shifted.schema)
            for column in original.column_names:
                if column not in POSITION_FIELDS:
                    self.assertTrue(original[column].equals(shifted[column]), column)
                    continue
                before = np.asarray(original[column].to_pylist(), dtype=np.float32)
                after = np.asarray(shifted[column].to_pylist(), dtype=np.float32)
                np.testing.assert_array_equal(after[:, :6], before[:, :6])
                np.testing.assert_array_equal(after[:, 7:13], before[:, 7:13])
                np.testing.assert_array_equal(
                    after[:, 6], before[:, 6] + np.float32(6.2)
                )
                np.testing.assert_array_equal(
                    after[:, 13], before[:, 13] + np.float32(6.2)
                )

            old_global = json.loads((source / "meta/stats_gr00t.json").read_text())
            new_global = json.loads((migrated / "meta/stats_gr00t.json").read_text())
            self.assertEqual(
                old_global["__format_version"], new_global["__format_version"]
            )
            self.assertEqual(old_global["__cache_config"], new_global["__cache_config"])
            for field in POSITION_FIELDS:
                old_stats = old_global["statistics"][field]
                new_stats = new_global["statistics"][field]
                self.assertEqual(old_stats["std"], new_stats["std"])
                self.assertEqual(old_stats["count"], new_stats["count"])
                for stat in ("min", "max", "mean", "q01", "q99"):
                    for index in range(14):
                        expected = old_stats[stat][index]
                        if index in (6, 13):
                            expected = float(
                                np.float32(old_stats[stat][index]) + np.float32(6.2)
                            )
                        self.assertEqual(new_stats[stat][index], expected)

                for index in (6, 13):
                    raw_before = np.float32(old_stats["mean"][index])
                    raw_after = np.float32(new_stats["mean"][index])
                    norm_before = 2 * (
                        (raw_before - np.float32(old_stats["min"][index]))
                        / (
                            np.float32(old_stats["max"][index])
                            - np.float32(old_stats["min"][index])
                        )
                    ) - 1
                    norm_after = 2 * (
                        (raw_after - np.float32(new_stats["min"][index]))
                        / (
                            np.float32(new_stats["max"][index])
                            - np.float32(new_stats["min"][index])
                        )
                    ) - 1
                    self.assertAlmostEqual(
                        float(norm_before), float(norm_after), places=6
                    )

            self.assertEqual(
                old_global["statistics"]["observation.qvel"],
                new_global["statistics"]["observation.qvel"],
            )

    def test_install_replaces_only_staged_files_and_keeps_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            staged = Path(temp_dir) / "staged"
            installed = Path(temp_dir) / "installed"
            _write_fixture(source)
            shutil.copytree(source, installed)
            unrelated = installed / "meta/keep.bin"
            unrelated.write_bytes(b"unchanged")
            migrate_dataset(source, staged, offset=6.2)

            replaced = install_staged_migration(staged, installed)

            self.assertEqual(len(replaced), 3)
            self.assertEqual(unrelated.read_bytes(), b"unchanged")
            verify_migration(source, installed, offset=6.2)

    def test_migration_rejects_bad_vector_width_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            migrated = Path(temp_dir) / "migrated"
            _write_fixture(source, malformed=True)

            with self.assertRaisesRegex(ValueError, r"action.*14"):
                migrate_dataset(source, migrated, offset=6.2)

            self.assertFalse(any(migrated.glob("data/*/*.parquet")))


if __name__ == "__main__":
    unittest.main()
