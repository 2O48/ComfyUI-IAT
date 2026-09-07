from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from py.nodes.dataset_repository import (  # noqa: E402
    DatasetError,
    dataset_bundle_is_current,
    dataset_bundle_matches_source,
    get_dataset_index,
    load_dataset_record,
    write_dataset_bundle,
)
from py.nodes.embedding_adapters import unload_embedding_adapters  # noqa: E402


DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve automotive CMF training samples that best match the user's text or reference image."
)
DEFAULT_DOCUMENT_INSTRUCTION = "Represent an automotive CMF training sample for retrieval."


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except Exception as exc:
        raise RuntimeError(f"Could not read config `{path}`: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Config `{path}` must contain a YAML object.")
    return value


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _safe_filename(value: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip().rstrip(". ")
    return safe or "dataset"


def _source_directories(input_path: Path) -> List[Path]:
    if not input_path.is_dir():
        raise RuntimeError(f"Input path is not a directory: `{input_path}`")
    if (input_path / "dataset.json").is_file():
        return [input_path]
    candidates = sorted(
        (path.parent for path in input_path.rglob("dataset.json") if "compiled" not in path.parts),
        key=lambda path: path.as_posix().lower(),
    )
    if not candidates:
        raise RuntimeError(f"No dataset.json files were found under `{input_path}`")
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile IAT image/caption datasets into portable .iatdb files.")
    parser.add_argument("input", type=Path, help="One dataset directory or a root containing multiple datasets")
    parser.add_argument("--config", type=Path, default=PLUGIN_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, help="Output .iatdb path; valid only for one dataset")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated .iatdb files")
    parser.add_argument("--model-path", type=str, help="Local Qwen3-VL-Embedding or Chinese CLIP directory")
    parser.add_argument("--provider", choices=("auto", "qwen3_vl", "chinese_clip"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--force", action="store_true", help="Rebuild even when the bundle is current")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    dataset_config = config.get("datasets") or {}
    if not isinstance(dataset_config, dict):
        raise RuntimeError("`datasets` in config.yaml must be an object.")

    raw_model_path = args.model_path or str(dataset_config.get("embedding_model_path") or "").strip()
    if not raw_model_path:
        raise RuntimeError("No embedding model is configured. Set datasets.embedding_model_path or --model-path.")
    model_path = _resolve_config_path(config_path, raw_model_path)
    if not model_path.is_dir():
        raise RuntimeError(f"Local embedding model directory does not exist: `{model_path}`")

    provider = args.provider or str(dataset_config.get("embedding_provider") or "auto")
    device = args.device or str(dataset_config.get("embedding_device") or "cuda")
    batch_size = args.batch_size or int(dataset_config.get("embedding_batch_size") or 1)
    dimension = args.dimension if args.dimension is not None else int(dataset_config.get("embedding_dimension") or 0)
    query_instruction = str(
        dataset_config.get("embedding_query_instruction") or DEFAULT_QUERY_INSTRUCTION
    ).strip()
    document_instruction = str(
        dataset_config.get("embedding_document_instruction") or DEFAULT_DOCUMENT_INSTRUCTION
    ).strip()
    if batch_size <= 0:
        raise RuntimeError("Embedding batch size must be positive.")
    if dimension < 0:
        raise RuntimeError("Embedding dimension cannot be negative.")

    input_path = args.input.expanduser().resolve()
    sources = _source_directories(input_path)
    if args.output and len(sources) != 1:
        raise RuntimeError("--output can only be used when compiling one dataset.")
    if args.output and args.output_dir:
        raise RuntimeError("Use either --output or --output-dir, not both.")
    default_output_dir = (input_path.parent if len(sources) == 1 else input_path) / "compiled"
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir.resolve()
    cache_dir = output_dir / ".iat_index"

    print(f"Embedding model: {model_path}")
    print(f"Provider/device: {provider}/{device}; dimension={dimension or 'model default'}; batch={batch_size}")
    built = 0
    skipped = 0
    seen_names = set()
    try:
        for source in sources:
            record = load_dataset_record(source)
            if record.dataset_name in seen_names:
                raise RuntimeError(f"Duplicate dataset_name in build input: `{record.dataset_name}`")
            seen_names.add(record.dataset_name)
            output_path = (
                args.output.expanduser().resolve()
                if args.output
                else output_dir / f"{_safe_filename(record.dataset_name)}.iatdb"
            )
            print(f"\n[{record.dataset_name}] {len(record.entries)} chunks")
            if record.metadata.get("sample_type_counts"):
                print(f"Sample types: {record.metadata['sample_type_counts']}")
            if not args.force and dataset_bundle_matches_source(
                record,
                output_path,
                str(model_path),
                provider,
                dimension,
                query_instruction,
                document_instruction,
            ):
                print(f"Up to date: {output_path}")
                skipped += 1
                continue
            index = get_dataset_index(
                record,
                cache_dir,
                embedding_model_path=str(model_path),
                require_embeddings=True,
                embedding_device=device,
                embedding_batch_size=batch_size,
                embedding_provider=provider,
                embedding_dimension=dimension,
                embedding_query_instruction=query_instruction,
                embedding_document_instruction=document_instruction,
            )
            if not args.force and dataset_bundle_is_current(index, output_path):
                print(f"Up to date: {output_path}")
                skipped += 1
                continue
            write_dataset_bundle(index, output_path)
            print(f"Built: {output_path}")
            if record.warnings:
                print(f"Non-fatal source warnings: {len(record.warnings)}")
            built += 1
    finally:
        unload_embedding_adapters()

    print(f"\nDone. Built={built}, skipped={skipped}, total={len(sources)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DatasetError, RuntimeError) as exc:
        print(f"[IAT] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
