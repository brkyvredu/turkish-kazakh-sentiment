#!/usr/bin/env python3
"""SAUCIS revision: offline, non-destructive JupyterLab preflight.

Run from the project root:
    python 00_revision_preflight.py --project .

Creates only a NEW _revision_audit/<timestamp>/ report directory. Does not
load/train models, initialize CUDA, download datasets, install packages,
remove caches, or deserialize pickle/joblib/checkpoint contents.
Uses the Python standard library; installed pyarrow is optional for schemas.
File enumeration is bounded and incomplete scans are explicitly reported.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

VERSION = "1.0.0"
DATASET_REPOS = (
    "maydogan/Turkish_SentimentAnalysis_TRSAv1", "issai/kazsandra",
)
MODEL_REPOS = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE", "BAAI/bge-m3",
    "FacebookAI/xlm-roberta-base",
)
PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "datasets",
    "huggingface-hub", "pyarrow", "torch", "transformers",
    "sentence-transformers", "tokenizers", "safetensors", "accelerate",
    "joblib", "PyYAML", "matplotlib", "tabulate",
)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "_revision_audit"}
DATA_SUFFIXES = {".csv", ".tsv", ".arrow", ".parquet"}
# Explicit allowlist. No token/password values or general environment dump.
ENV_KEYS = (
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE",
    "XDG_CACHE_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
)
SECRET_NAMES = {"token", "stored_tokens", ".env", ".netrc", ".git-credentials", "credentials"}


def public_string(value: object) -> str:
    text = str(value)
    text = re.sub(r"hf_[A-Za-z0-9]{24,}", "[HF_TOKEN_REDACTED]", text)
    # A credential accidentally placed in a URL should not leak through paths/errors.
    text = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)
    return text


def small_text(path: Path, max_bytes: int = 1024 * 1024) -> str:
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Metadata exceeds permitted size")
    return data.decode("utf-8-sig")


def command(argv: list[str], timeout: int = 15) -> dict:
    """Run a fixed read-only utility, never shell=True."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="replace", check=False)
        return {"returncode": result.returncode,
                "stdout": public_string(result.stdout[:40000]),
                "stderr": public_string(result.stderr[:4000])}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": public_string(exc)}


def disk_info(path: Path) -> dict:
    try:
        actual = path
        while not actual.exists() and actual != actual.parent:
            actual = actual.parent
        usage = shutil.disk_usage(actual)
        entry = {"requested_path": str(path), "measured_path": str(actual),
                 "total_bytes": usage.total, "used_bytes": usage.used,
                 "free_bytes": usage.free, "device_id": actual.stat().st_dev,
                 "writable_permission_check": os.access(actual, os.W_OK)}
        if hasattr(os, "statvfs"):
            fs = os.statvfs(actual)
            entry.update(inodes_total=fs.f_files, inodes_available=fs.f_favail)
        return entry
    except OSError as exc:
        return {"requested_path": str(path), "error": public_string(exc)}


def resources(project: Path) -> dict:
    info = {"python_executable": sys.executable, "python_version": sys.version,
            "cpu_count": os.cpu_count(), "project_disk": disk_info(project)}
    if hasattr(os, "sched_getaffinity"):
        info["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    memory_path = Path("/proc/meminfo")
    if memory_path.exists():
        mem = {}
        for line in small_text(memory_path).splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"}:
                mem[key] = value.strip()
        info["host_memory"] = mem
    cg_paths = [Path("/sys/fs/cgroup")]
    try:
        for line in small_text(Path("/proc/self/cgroup")).splitlines():
            if line.startswith("0::"):
                relative = line[3:].lstrip("/")
                if relative and ".." not in Path(relative).parts:
                    cg_paths.append(Path("/sys/fs/cgroup") / relative)
    except (OSError, ValueError):
        pass
    cg = []
    for root in dict.fromkeys(cg_paths):
        entry = {"path": str(root)}
        for name in ("memory.max", "memory.current", "cpu.max", "cpuset.cpus.effective"):
            try:
                entry[name] = small_text(root / name, 4096).strip()
            except (OSError, ValueError):
                pass
        if len(entry) > 1:
            cg.append(entry)
    info["cgroup_v2_visible_limits"] = cg
    info["cgroup_note"] = "Host memory is not an allocation guarantee; ancestor/container limits may apply."
    info["packages"] = {}
    for package in PACKAGES:
        try:
            info["packages"][package] = md.version(package)
        except md.PackageNotFoundError:
            info["packages"][package] = None
    info["selected_environment"] = {k: public_string(os.environ[k]) for k in ENV_KEYS if k in os.environ}
    info["gpu_query"] = command([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    info["torch_runtime_probed"] = False
    info["gpu_note"] = "nvidia-smi query only: no GPU context, tensor allocation or CUDA compatibility test."
    return info


def relevant_dataset_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return any(token in normalized for token in ("kazsandra", "trsav1", "turkishsentimentanalysis"))


def discover_roots(project: Path, extra_data: list[str], extra_hf: list[str]) -> tuple[list[dict], list[dict]]:
    """Inspect only known project subfolders and matching HF repositories."""
    targets, discovery = [], []
    seen = set()

    def add(path: Path, kind: str, repo: str | None = None):
        path = Path(os.path.abspath(path.expanduser()))
        identity = (str(path), kind)
        if identity not in seen:
            seen.add(identity)
            targets.append({"path": str(path), "kind": kind, "repo_id": repo})

    for name in ("data", "external", "results", "checkpoints", "models"):
        add(project / name, "project_" + name)
    for item in extra_data:
        add(Path(item), "extra_data")

    default_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"
    active_home = Path(os.environ.get("HF_HOME", str(default_home))).expanduser()
    homes = list(dict.fromkeys([active_home, default_home, Path.home() / ".cache/huggingface",
                               *[Path(p).expanduser() for p in extra_hf]]))
    hub_roots = [Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(active_home / "hub")))).expanduser()]
    dataset_roots = [Path(os.environ.get("HF_DATASETS_CACHE", str(active_home / "datasets"))).expanduser()]
    hub_roots += [h / "hub" for h in homes]
    dataset_roots += [h / "datasets" for h in homes]
    if os.environ.get("TRANSFORMERS_CACHE"):
        hub_roots.append(Path(os.environ["TRANSFORMERS_CACHE"]).expanduser())

    for root in dict.fromkeys(hub_roots):
        status = {"kind": "hf_hub_root", "path": str(root), "exists": root.is_dir(),
                  "enumeration_complete": True, "matches": []}
        if root.is_dir():
            try:
                with os.scandir(root) as entries:
                    for number, ent in enumerate(entries):
                        if number >= 3000:
                            status["enumeration_complete"] = False
                            break
                        if not ent.is_dir(follow_symlinks=True):
                            continue
                        for repo in DATASET_REPOS + MODEL_REPOS:
                            prefix = "datasets--" if repo in DATASET_REPOS else "models--"
                            if ent.name.lower() == (prefix + repo.replace("/", "--")).lower():
                                add(Path(ent.path), "hf_dataset" if repo in DATASET_REPOS else "hf_model", repo)
                                status["matches"].append(ent.name)
            except OSError as exc:
                status.update(error=public_string(exc), enumeration_complete=False)
        discovery.append(status)

    for root in dict.fromkeys(dataset_roots):
        status = {"kind": "hf_datasets_root", "path": str(root), "exists": root.is_dir(),
                  "enumeration_complete": True, "matches": []}
        if root.is_dir():
            try:
                with os.scandir(root) as entries:
                    for number, ent in enumerate(entries):
                        if number >= 3000:
                            status["enumeration_complete"] = False
                            break
                        if ent.is_dir(follow_symlinks=True) and relevant_dataset_name(ent.name):
                            add(Path(ent.path), "hf_prepared_dataset")
                            status["matches"].append(ent.name)
            except OSError as exc:
                status.update(error=public_string(exc), enumeration_complete=False)
        discovery.append(status)
    return targets, discovery


def bounded_inventory(target: dict, seconds: float, max_entries: int) -> tuple[dict, list[Path]]:
    root = Path(target["path"])
    report = dict(target, exists=root.exists(), scan_complete=True, examined_entries=0,
                  files_seen=0, logical_bytes=0, unique_allocated_bytes=0,
                  extension_counts={}, directories_not_followed=[], errors=[], files=[])
    if not root.is_dir():
        report["status"] = "not_found_at_checked_path" if not root.exists() else "not_a_directory"
        return report, []
    deadline = time.monotonic() + seconds
    stack, files, inodes, largest = [(root, 0)], [], set(), []
    while stack:
        folder, depth = stack.pop()
        if depth > 14:
            report["scan_complete"] = False
            report["directories_not_followed"].append(str(folder))
            continue
        if time.monotonic() >= deadline or report["examined_entries"] >= max_entries:
            report["scan_complete"] = False
            report["stop_reason"] = "time_or_entry_budget"
            break
        try:
            with os.scandir(folder) as entries:
                for ent in entries:
                    if time.monotonic() >= deadline or report["examined_entries"] >= max_entries:
                        report["scan_complete"] = False
                        report["stop_reason"] = "time_or_entry_budget"
                        stack.clear()
                        break
                    report["examined_entries"] += 1
                    if ent.name.lower() in SECRET_NAMES or ent.name.endswith(".lock"):
                        continue
                    p = Path(ent.path)
                    try:
                        if ent.is_symlink() and ent.is_dir(follow_symlinks=True):
                            report["directories_not_followed"].append(str(p.relative_to(root)))
                            report["scan_complete"] = False
                            continue
                        if ent.is_dir(follow_symlinks=False):
                            if ent.name not in SKIP_DIRS:
                                stack.append((p, depth + 1))
                            continue
                        st = ent.stat(follow_symlinks=True)
                        if not stat.S_ISREG(st.st_mode):
                            continue
                    except OSError as exc:
                        report["scan_complete"] = False
                        if len(report["errors"]) < 30:
                            report["errors"].append({"path": str(p), "error": public_string(exc)})
                        continue
                    report["files_seen"] += 1
                    report["logical_bytes"] += st.st_size
                    inode_key = (st.st_dev, st.st_ino)
                    if inode_key not in inodes:
                        inodes.add(inode_key)
                        report["unique_allocated_bytes"] += getattr(st, "st_blocks", 0) * 512
                    suffix = p.suffix.lower() or "[no extension]"
                    report["extension_counts"][suffix] = report["extension_counts"].get(suffix, 0) + 1
                    entry = {"relative_path": str(p.relative_to(root)), "bytes": st.st_size,
                             "symlink": ent.is_symlink()}
                    files.append(p)
                    if len(report["files"]) < 400:
                        report["files"].append(entry)
                    largest.append(entry)
        except OSError as exc:
            report["scan_complete"] = False
            if len(report["errors"]) < 30:
                report["errors"].append({"path": str(folder), "error": public_string(exc)})
    report["listing_truncated"] = report["files_seen"] > len(report["files"])
    report["largest_files"] = sorted(largest, key=lambda r: r["bytes"], reverse=True)[:12]
    report["note"] = "Sizes are inventory estimates, not reclaimable space. Totals from different roots may overlap."
    return report, files


def inspect_schema(path: Path) -> dict:
    """Schema only: never report rows, review text, schema metadata or labels."""
    result = {"path": str(path), "kind": path.suffix.lower(), "columns": []}
    try:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            # Only a bounded first physical line; no row samples exported.
            with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
                first = handle.readline(65537)
            if len(first) > 65536:
                raise ValueError("CSV header too long; skipped")
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                delimiter = csv.Sniffer().sniff(first, delimiters=",\t;|").delimiter
            except csv.Error:
                pass
            columns = next(csv.reader([first], delimiter=delimiter))
            # Reject data-like first rows rather than dumping possible review text.
            if not columns or not all(re.fullmatch(r"[\w .:/()\-]{0,100}", c or "", flags=re.UNICODE) for c in columns):
                raise ValueError("Not a safe, recognizable header; skipped")
            recognized = {"text", "text_cleaned", "review", "comment", "sentence", "content", "body",
                          "label", "label_raw", "label_id", "score", "rating", "sentiment", "stars",
                          "id", "custom_id", "sample_id", "review_id", "domain", "language", "source", "n"}
            if len({c.lower() for c in columns} & recognized) < 2:
                raise ValueError("Cannot confirm dataset header; content withheld")
            result.update(columns=columns, delimiter=delimiter, row_count=None)
        elif suffix in {".arrow", ".parquet"}:
            import pyarrow as pa  # optional local import; no model or dataset loading
            if suffix == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(str(path))
                result.update(columns=pf.schema_arrow.names, row_count=pf.metadata.num_rows)
            else:
                with pa.memory_map(str(path), "r") as source:
                    try:
                        reader = pa.ipc.open_file(source)
                    except pa.ArrowInvalid:
                        source.seek(0)
                        reader = pa.ipc.open_stream(source)
                    result.update(columns=reader.schema.names, row_count=None)
        else:
            raise ValueError("Unsupported schema format")
        lower = {c.lower() for c in result["columns"]}
        result["domain_column_candidates"] = sorted(lower & {"domain", "category", "source", "site", "platform"})
        result["id_column_candidates"] = sorted(lower & {"id", "custom_id", "review_id", "sample_id"})
        result["label_column_candidates"] = sorted(lower & {"label", "label_raw", "score", "rating", "sentiment", "stars"})
        result["status"] = "schema_read"
    except Exception as exc:
        result.update(status="not_read", error=public_string(f"{type(exc).__name__}: {exc}"))
    return result


def inspect_npy_header(path: Path) -> dict:
    """Read only the NumPy header; never load payload or allow pickle."""
    result = {"path": str(path)}
    try:
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise ValueError("Invalid NPY magic")
            version = tuple(handle.read(2))
            if version not in {(1, 0), (2, 0), (3, 0)}:
                raise ValueError("Unsupported NPY version")
            count = 2 if version == (1, 0) else 4
            data = handle.read(count)
            if len(data) != count:
                raise ValueError("Truncated header length")
            n = struct.unpack("<H" if count == 2 else "<I", data)[0]
            if n > 131072:
                raise ValueError("Oversized NPY header")
            raw = handle.read(n)
            if len(raw) != n:
                raise ValueError("Truncated NPY header")
            parsed = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1"))
            shape = parsed.get("shape")
            if not isinstance(shape, tuple) or not all(isinstance(v, int) and v >= 0 for v in shape):
                raise ValueError("Invalid NPY shape")
            result.update(shape=list(shape), dtype=parsed.get("descr"),
                          fortran_order=parsed.get("fortran_order"))
        candidates = [path.with_suffix(".json"), path.with_suffix(".manifest.json"),
                      Path(str(path) + ".json"), Path(str(path) + ".manifest.json")]
        result["nearby_metadata_files"] = [str(p) for p in candidates if p.is_file()]
        result["alignment_verified"] = False
        result["note"] = "Header/row count alone never authorizes embedding reuse."
    except Exception as exc:
        result["error"] = public_string(f"{type(exc).__name__}: {exc}")
    return result


def dataset_metadata(path: Path) -> dict:
    result = {"path": str(path)}
    try:
        info = json.loads(small_text(path, 2 * 1024 * 1024))
        result["columns"] = list(info.get("features", {}).keys())
        splits = info.get("splits", {})
        result["split_counts"] = {str(k): {field: v[field] for field in ("num_examples", "num_bytes") if field in v}
                                  for k, v in splits.items() if isinstance(v, dict)}
        result["builder_name"] = info.get("builder_name")
        result["config_name"] = info.get("config_name")
    except Exception as exc:
        result["error"] = public_string(f"{type(exc).__name__}: {exc}")
    return result


def summary(report: dict) -> str:
    lines = ["SAUCIS REVIZYONU — YEREL ON DENETIM", "=" * 60,
             "Bu rapor veri dogrulama/model deneyi degildir; mevcut dosyalarin envanteridir.",
             "Indirme, egitim, GPU ayirma, dosya silme veya cache tasima yapilmadi.", "",
             f"Proje: {report['project']}", f"Python: {report['resources']['python_executable']}",
             f"CPU affinity: {report['resources'].get('cpu_affinity_count', 'bilinmiyor')}"]
    free = report["resources"]["project_disk"].get("free_bytes")
    if free is not None:
        lines.append(f"Bos disk (anlik): {free / 2**30:.2f} GiB — yalnizca bilgi, deney kapsami kisitlanmadi.")
    lines += ["", "GPU ENVANTERI (index, UUID, name, driver, total/used/free MiB):",
              report["resources"]["gpu_query"].get("stdout") or str(report["resources"]["gpu_query"])]
    lines += ["", "ILGILI KLASORLER:"]
    for scan in report["inventories"]:
        note = "tam" if scan["scan_complete"] else "KISMI"
        if not scan["exists"]:
            lines.append(f"- {scan['path']}: bu yolda bulunamadi")
        else:
            lines.append(f"- {scan['path']}: {scan['files_seen']} dosya goruldu; tarama {note}")
    lines += ["", "VERI SEMALARI (icerik/yorum metni yok):"]
    for item in report["schemas"]:
        lines.append(f"- {item['path']}")
        lines.append("  " + (", ".join(item["columns"]) if item["status"] == "schema_read" else item.get("error", "okunamadi")))
    lines += ["", "DATASET_INFO METADATA:"]
    for item in report["dataset_metadata"]:
        lines.append(f"- {item['path']}: columns={item.get('columns', [])}; splits={item.get('split_counts', {})}")
    lines += ["", "EMBEDDING CACHE BASLIKLARI:"]
    for item in report["embedding_headers"]:
        lines.append(f"- {item['path']}: shape={item.get('shape')}, dtype={item.get('dtype')}; hizalama DOGRULANMADI")
    lines += ["", "SINIRLAR / SONRAKI KONTROL:",
              "- Dosyanin/kolonun bulunmasi tamlik, domain kalitesi veya etiket dogrulugu kaniti degildir.",
              "- Snapshot adi mevcut olsa da butun gerekli parcalar inmemis olabilir.",
              "- Sonuclar tam kapsamli veri profili ve split denetimiyle dogrulanacak.",
              "- Bu taramada bulunmayan veri baska bir klasorde bulunabilir.",
              "- Kontrol edilen HF kokleri ve kisimli taramalar JSON raporundadir.",
              "- Ham yorumlar ve erisim token degerleri rapora eklenmez.",
              "- GPU secimi/uyumluluk ve checkpoint-resume testi bir sonraki asamadir."]
    lines += ["", "RAPOR KLASORU: " + report["output_directory"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".")
    parser.add_argument("--output-root", default="_revision_audit")
    parser.add_argument("--extra-data-dir", action="append", default=[], help="Additional explicit data folder; may repeat.")
    parser.add_argument("--extra-hf-home", action="append", default=[], help="Additional explicit HF home folder; may repeat.")
    parser.add_argument("--scan-seconds", type=float, default=15, help="Budget per target folder.")
    parser.add_argument("--max-entries", type=int, default=6000, help="Budget per target folder.")
    parser.add_argument("--max-schema-files", type=int, default=40, help="Read only this many local schema headers.")
    args = parser.parse_args()
    if args.scan_seconds <= 0 or args.max_entries <= 0 or args.max_schema_files < 0:
        parser.error("Scan limits must be positive (schema count may be zero).")
    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        parser.error(f"Project folder not found: {project}")
    # These flags affect this diagnostic process only; no login/config changes.
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    report = {"schema_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "project": str(project), "resources": resources(project),
              "scope": "read-only targeted inventory; writes only new reports",
              "runtime_cuda_compatibility_tested": False}
    targets, discovery = discover_roots(project, args.extra_data_dir, args.extra_hf_home)
    report["cache_root_discovery"] = discovery
    inventories, candidates, cache_npy, meta_candidates = [], [], [], []
    for target in targets:
        print(f"[SCAN] {target['path']}", flush=True)
        inventory, files = bounded_inventory(target, args.scan_seconds, args.max_entries)
        inventories.append(inventory)
        if target["kind"] not in {"hf_model", "project_results", "project_checkpoints", "project_models"}:
            candidates.extend(p for p in files if p.suffix.lower() in DATA_SUFFIXES)
            meta_candidates.extend(p for p in files if p.name == "dataset_info.json")
        if target["kind"] == "project_results":
            cache_npy.extend(p for p in files if p.suffix.lower() == ".npy")
    # HF and explicit raw data schemas first; processed project files second.
    def priority(path: Path):
        text = str(path).lower()
        return ("processed" in path.parts, not ("snapshots" in path.parts), text)
    candidates = sorted(dict.fromkeys(candidates), key=priority)
    report["inventories"] = inventories
    report["schema_candidates_seen"] = len(candidates)
    report["schema_inspection_truncated"] = len(candidates) > args.max_schema_files
    report["schemas"] = [inspect_schema(p) for p in candidates[:args.max_schema_files]]
    report["dataset_metadata"] = [dataset_metadata(p) for p in list(dict.fromkeys(meta_candidates))[:40]]
    report["embedding_headers"] = [inspect_npy_header(p) for p in cache_npy[:40]]
    report["embedding_header_inspection_truncated"] = len(cache_npy) > 40
    report["limitations"] = [
        "No full data validation, label audit, cache alignment verification or checksum validation of large files.",
        "Search limited to listed folders; absence here does not prove absence on the server.",
        "Directory symlinks below roots not followed; errors and caps make scans partial.",
        "CUDA runtime and kernel compatibility not tested; nvidia-smi information only.",
        "Report includes local paths; review before public distribution.",
    ]
    report["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_root = Path(args.output_root).expanduser()
    if not out_root.is_absolute():
        out_root = project / out_root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = out_root / stamp
    # Exclusive timestamp directory: no overwriting of earlier reports or results.
    out.mkdir(parents=True, exist_ok=False)
    report["output_directory"] = str(out)
    json_text = public_string(json.dumps(report, ensure_ascii=False, indent=2))
    txt = public_string(summary(report))
    with (out / "preflight_report.json").open("x", encoding="utf-8") as handle:
        handle.write(json_text + "\n")
    with (out / "preflight_summary.txt").open("x", encoding="utf-8") as handle:
        handle.write(txt)
    print("\n" + txt)
    print("[DONE] Send preflight_report.json and preflight_summary.txt; not the datasets/results directory.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] Audit interrupted. No source files were modified.", file=sys.stderr)
        raise SystemExit(130)
    except OSError as exc:
        print(f"[ERROR] {public_string(exc)}. No source files were modified.", file=sys.stderr)
        raise SystemExit(1)
