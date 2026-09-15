#!/usr/bin/env python3
"""Stage and verify the ARX v1 continuous-gripper offset migration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


POSITION_FIELDS = (
    "observation.state",
    "observation.eef",
    "observation.eef_tcp",
    "action",
)
GRIPPER_INDICES = (6, 13)
LOCATION_STATISTICS = ("min", "max", "mean", "q01", "q99")
EPISODE_STATS_PATH = Path("meta/episodes_stats.jsonl")
GLOBAL_STATS_PATH = Path("meta/stats_gr00t.json")


@dataclass(frozen=True)
class MigrationReport:
    source_root: str
    output_root: str
    parquet_files: int
    rows: int
    offset: float


@dataclass(frozen=True)
class VerificationReport:
    source_root: str
    migrated_root: str
    parquet_files: int
    rows: int
    offset: float
    left_gripper_range: tuple[float, float]
    right_gripper_range: tuple[float, float]


def _parquet_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("data/*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet files found under {root / 'data'}")
    return paths


def _load_vectors(table: pa.Table, field: str, source: Path) -> np.ndarray:
    if field not in table.column_names:
        raise ValueError(f"{source}: missing required field {field!r}")
    column = table[field]
    if not (
        pa.types.is_list(column.type)
        and pa.types.is_float32(column.type.value_type)
    ):
        raise ValueError(
            f"{source}: {field} must be list<float32>, got {column.type}"
        )
    values = column.to_pylist()
    if any(value is None or len(value) != 14 for value in values):
        widths = sorted(
            {None if value is None else len(value) for value in values},
            key=str,
        )
        raise ValueError(f"{source}: {field} must have width 14; got {widths}")
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (table.num_rows, 14):
        raise ValueError(
            f"{source}: {field} must have shape [{table.num_rows},14], "
            f"got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{source}: {field} contains NaN or Inf")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: contains no records")
    return records


def _validate_stats(stats: dict[str, Any], source: Path) -> None:
    for field in POSITION_FIELDS:
        if field not in stats or not isinstance(stats[field], dict):
            raise ValueError(f"{source}: missing statistics for {field!r}")
        field_stats = stats[field]
        for statistic in ("min", "max", "mean", "std"):
            values = field_stats.get(statistic)
            if not isinstance(values, list) or len(values) != 14:
                raise ValueError(
                    f"{source}: {field}.{statistic} must have width 14"
                )
        for statistic in ("q01", "q99"):
            values = field_stats.get(statistic)
            if values is not None and (
                not isinstance(values, list) or len(values) != 14
            ):
                raise ValueError(
                    f"{source}: {field}.{statistic} must have width 14"
                )
        count = field_stats.get("count")
        if count is not None and (
            not isinstance(count, list) or len(count) != 1
        ):
            raise ValueError(f"{source}: {field}.count must have width 1")


def _shift_float32(value: float, offset: np.float32) -> float:
    return float(np.float32(value) + offset)


def _shift_stats(stats: dict[str, Any], offset: np.float32) -> dict[str, Any]:
    shifted = copy.deepcopy(stats)
    for field in POSITION_FIELDS:
        for statistic in LOCATION_STATISTICS:
            values = shifted[field].get(statistic)
            if values is None:
                continue
            for index in GRIPPER_INDICES:
                values[index] = _shift_float32(values[index], offset)
    return shifted


def _atomic_write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=4)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight(
    source_root: Path,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    parquet_paths = _parquet_paths(source_root)
    for path in parquet_paths:
        table = pq.read_table(path)
        for field in POSITION_FIELDS:
            _load_vectors(table, field, path)

    episode_records = _read_jsonl(source_root / EPISODE_STATS_PATH)
    for record in episode_records:
        stats = record.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"{source_root / EPISODE_STATS_PATH}: missing stats")
        _validate_stats(stats, source_root / EPISODE_STATS_PATH)

    global_payload = _read_json(source_root / GLOBAL_STATS_PATH)
    global_stats = global_payload.get("statistics")
    if not isinstance(global_stats, dict):
        raise ValueError(f"{source_root / GLOBAL_STATS_PATH}: missing statistics")
    _validate_stats(global_stats, source_root / GLOBAL_STATS_PATH)
    return parquet_paths, episode_records, global_payload


def migrate_dataset(
    source_root: Path | str,
    output_root: Path | str,
    *,
    offset: float = 6.2,
) -> MigrationReport:
    """Write a validated, changed-files-only migration into ``output_root``."""
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if source == output:
        raise ValueError("source_root and output_root must be different")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output root is not empty: {output}")

    parquet_paths, episode_records, global_payload = _preflight(source)
    shift = np.float32(offset)
    rows = 0
    for source_path in parquet_paths:
        table = pq.read_table(source_path)
        rows += table.num_rows
        for field in POSITION_FIELDS:
            values = _load_vectors(table, field, source_path).copy()
            values[:, list(GRIPPER_INDICES)] += shift
            column_index = table.schema.get_field_index(field)
            shifted_column = pa.array(values.tolist(), type=table[field].type)
            table = table.set_column(
                column_index,
                table.schema.field(column_index),
                shifted_column,
            )
        relative = source_path.relative_to(source)
        _atomic_write_table(output / relative, table)

    shifted_episodes = []
    for record in episode_records:
        shifted_record = copy.deepcopy(record)
        shifted_record["stats"] = _shift_stats(shifted_record["stats"], shift)
        shifted_episodes.append(shifted_record)
    _atomic_write_jsonl(output / EPISODE_STATS_PATH, shifted_episodes)

    shifted_global = copy.deepcopy(global_payload)
    shifted_global["statistics"] = _shift_stats(
        shifted_global["statistics"], shift
    )
    _atomic_write_json(output / GLOBAL_STATS_PATH, shifted_global)
    return MigrationReport(
        source_root=str(source),
        output_root=str(output),
        parquet_files=len(parquet_paths),
        rows=rows,
        offset=float(shift),
    )


def install_staged_migration(
    staged_root: Path | str,
    dataset_root: Path | str,
) -> list[Path]:
    """Atomically replace only the Parquet and statistics files in a stage."""
    staged = Path(staged_root).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    staged_paths = _parquet_paths(staged) + [
        staged / EPISODE_STATS_PATH,
        staged / GLOBAL_STATS_PATH,
    ]
    relatives = [path.relative_to(staged) for path in staged_paths]
    targets = [dataset / relative for relative in relatives]
    for source_path, target_path in zip(staged_paths, targets):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if not target_path.is_file():
            raise FileNotFoundError(target_path)

    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for source_path, target_path in zip(staged_paths, targets):
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                suffix=".installing",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            shutil.copy2(source_path, temporary)
            temporary_paths.append((temporary, target_path))
        for temporary, target_path in temporary_paths:
            os.replace(temporary, target_path)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)
    return relatives


def verify_migration(
    source_root: Path | str,
    migrated_root: Path | str,
    *,
    offset: float = 6.2,
) -> VerificationReport:
    """Prove that a staged migration changed only the intended values."""
    source = Path(source_root).expanduser().resolve()
    migrated = Path(migrated_root).expanduser().resolve()
    source_paths = _parquet_paths(source)
    migrated_paths = _parquet_paths(migrated)
    source_relatives = [path.relative_to(source) for path in source_paths]
    migrated_relatives = [path.relative_to(migrated) for path in migrated_paths]
    if source_relatives != migrated_relatives:
        raise ValueError("source and migrated Parquet path sets differ")

    shift = np.float32(offset)
    rows = 0
    grippers: list[list[np.ndarray]] = [[], []]
    for relative in source_relatives:
        original = pq.read_table(source / relative)
        changed = pq.read_table(migrated / relative)
        if original.schema != changed.schema:
            raise ValueError(f"{relative}: Arrow schema changed")
        if original.num_rows != changed.num_rows:
            raise ValueError(f"{relative}: row count changed")
        rows += original.num_rows
        for column in original.column_names:
            if column not in POSITION_FIELDS:
                if not original[column].equals(changed[column]):
                    raise ValueError(f"{relative}: unrelated column {column!r} changed")
                continue
            before = _load_vectors(original, column, source / relative)
            after = _load_vectors(changed, column, migrated / relative)
            expected = before.copy()
            expected[:, list(GRIPPER_INDICES)] += shift
            if not np.array_equal(after, expected):
                raise ValueError(f"{relative}: {column} differs beyond gripper shift")
            if column == "observation.state":
                for hand, index in enumerate(GRIPPER_INDICES):
                    grippers[hand].append(after[:, index])

        state = _load_vectors(changed, "observation.state", migrated / relative)
        action = _load_vectors(changed, "action", migrated / relative)
        if not np.array_equal(
            state[:, list(GRIPPER_INDICES)],
            action[:, list(GRIPPER_INDICES)],
        ):
            raise ValueError(f"{relative}: state/action grippers are not equal")

    old_episodes = _read_jsonl(source / EPISODE_STATS_PATH)
    new_episodes = _read_jsonl(migrated / EPISODE_STATS_PATH)
    if len(old_episodes) != len(new_episodes):
        raise ValueError("episode statistics record count changed")
    for index, (old_record, new_record) in enumerate(zip(old_episodes, new_episodes)):
        expected_record = copy.deepcopy(old_record)
        expected_record["stats"] = _shift_stats(expected_record["stats"], shift)
        if new_record != expected_record:
            raise ValueError(
                f"{EPISODE_STATS_PATH}: record {index} differs beyond gripper shift"
            )

    old_global = _read_json(source / GLOBAL_STATS_PATH)
    new_global = _read_json(migrated / GLOBAL_STATS_PATH)
    expected_global = copy.deepcopy(old_global)
    expected_global["statistics"] = _shift_stats(
        expected_global["statistics"], shift
    )
    if new_global != expected_global:
        raise ValueError(f"{GLOBAL_STATS_PATH}: differs beyond gripper shift")

    left = np.concatenate(grippers[0])
    right = np.concatenate(grippers[1])
    return VerificationReport(
        source_root=str(source),
        migrated_root=str(migrated),
        parquet_files=len(source_paths),
        rows=rows,
        offset=float(shift),
        left_gripper_range=(float(left.min()), float(left.max())),
        right_gripper_range=(float(right.min()), float(right.max())),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offset", type=float, default=6.2)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing staged migration without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verify_only:
        report = migrate_dataset(
            args.source_root,
            args.output_root,
            offset=args.offset,
        )
        print(json.dumps(asdict(report), indent=2))
    verification = verify_migration(
        args.source_root,
        args.output_root,
        offset=args.offset,
    )
    print(json.dumps(asdict(verification), indent=2))


if __name__ == "__main__":
    main()
