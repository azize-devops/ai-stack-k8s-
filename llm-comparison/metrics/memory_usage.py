"""
Memory usage tracking for LLM benchmarking.

Provides system-level memory measurement via ``psutil`` (with a ``/proc``
fallback) and optional Kubernetes pod-level memory queries through the
Kubernetes Metrics API.
"""

from __future__ import annotations

import logging
import subprocess
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import psutil; fall back to /proc on Linux.
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False
    logger.debug("psutil not installed -- falling back to /proc/meminfo")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryMetrics:
    """Structured container for memory measurements."""

    total_mb: float = 0.0
    available_mb: float = 0.0
    used_mb: float = 0.0
    percent_used: float = 0.0
    process_rss_mb: Optional[float] = None
    pod_memory_mb: Optional[float] = None
    pod_memory_limit_mb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key, val in d.items():
            if isinstance(val, float):
                d[key] = round(val, 2)
        return d


# ---------------------------------------------------------------------------
# System memory
# ---------------------------------------------------------------------------

def get_system_memory() -> MemoryMetrics:
    """Return current system memory usage.

    Prefers ``psutil`` when available; otherwise parses ``/proc/meminfo``
    (Linux only).

    Returns
    -------
    MemoryMetrics
        Populated with system-level fields (``total_mb``, ``available_mb``,
        ``used_mb``, ``percent_used``) and the current process RSS.
    """
    if _HAS_PSUTIL:
        return _memory_via_psutil()
    return _memory_via_proc()


def _memory_via_psutil() -> MemoryMetrics:
    """Gather memory stats through psutil."""
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024 * 1024)

    return MemoryMetrics(
        total_mb=vm.total / (1024 * 1024),
        available_mb=vm.available / (1024 * 1024),
        used_mb=vm.used / (1024 * 1024),
        percent_used=vm.percent,
        process_rss_mb=rss,
    )


def _memory_via_proc() -> MemoryMetrics:
    """Parse /proc/meminfo as a psutil fallback (Linux only)."""
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        logger.warning("/proc/meminfo not found -- returning zeroed metrics")
        return MemoryMetrics()

    info: dict[str, int] = {}
    with meminfo_path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                # Values in /proc/meminfo are in kB.
                info[key] = int(parts[1])

    total_kb = info.get("MemTotal", 0)
    available_kb = info.get("MemAvailable", info.get("MemFree", 0))
    used_kb = total_kb - available_kb

    # Attempt to read own RSS from /proc/self/status.
    rss_mb: Optional[float] = None
    status_path = Path("/proc/self/status")
    if status_path.exists():
        with status_path.open() as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    rss_mb = rss_kb / 1024
                    break

    total_mb = total_kb / 1024
    percent = (used_kb / total_kb * 100) if total_kb > 0 else 0.0

    return MemoryMetrics(
        total_mb=total_mb,
        available_mb=available_kb / 1024,
        used_mb=used_kb / 1024,
        percent_used=percent,
        process_rss_mb=rss_mb,
    )


# ---------------------------------------------------------------------------
# Kubernetes pod memory
# ---------------------------------------------------------------------------

def get_pod_memory(
    pod_name: str,
    namespace: str = "ai-stack",
    container: Optional[str] = None,
) -> MemoryMetrics:
    """Query the Kubernetes Metrics API for a specific pod's memory usage.

    Requires ``kubectl`` to be available and configured on the host.
    Falls back gracefully if ``kubectl`` is missing or the cluster is
    unreachable.

    Parameters
    ----------
    pod_name:
        Exact pod name or a label-selector prefix (e.g. ``"localai"``).
    namespace:
        Kubernetes namespace.  Defaults to ``"ai-stack"``.
    container:
        Optional container name within the pod.

    Returns
    -------
    MemoryMetrics
        With ``pod_memory_mb`` populated.  ``pod_memory_limit_mb`` is
        populated when resource limits are defined on the container spec.
    """
    metrics = MemoryMetrics()

    # --- Fetch current memory usage via `kubectl top` ---
    try:
        cmd = [
            "kubectl", "top", "pod", pod_name,
            "--namespace", namespace,
            "--no-headers",
        ]
        if container:
            cmd += ["--containers", container]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Example output:  "localai-0   245m   1842Mi"
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                # Memory is the last column (or second-to-last with containers flag).
                mem_str = parts[-1] if not container else parts[2] if len(parts) > 2 else parts[-1]
                metrics.pod_memory_mb = _parse_k8s_memory(mem_str)
                break  # take the first matching line
        else:
            logger.warning(
                "kubectl top failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
    except FileNotFoundError:
        logger.info("kubectl not found -- skipping pod memory query")
    except subprocess.TimeoutExpired:
        logger.warning("kubectl top timed out")
    except Exception as exc:
        logger.warning("Unexpected error querying pod memory: %s", exc)

    # --- Fetch memory limit from pod spec ---
    try:
        cmd_spec = [
            "kubectl", "get", "pod", pod_name,
            "--namespace", namespace,
            "-o", "json",
        ]
        result_spec = subprocess.run(
            cmd_spec, capture_output=True, text=True, timeout=10,
        )
        if result_spec.returncode == 0:
            pod_json = json.loads(result_spec.stdout)
            containers = pod_json.get("spec", {}).get("containers", [])
            target = container or (containers[0]["name"] if containers else None)
            for c in containers:
                if c["name"] == target or target is None:
                    limit = (
                        c.get("resources", {})
                         .get("limits", {})
                         .get("memory", "")
                    )
                    if limit:
                        metrics.pod_memory_limit_mb = _parse_k8s_memory(limit)
                    break
    except Exception:
        pass  # non-critical

    return metrics


# ---------------------------------------------------------------------------
# Before / after comparison
# ---------------------------------------------------------------------------

def measure_memory_delta(
    before: MemoryMetrics,
    after: MemoryMetrics,
) -> dict[str, float]:
    """Compare two memory snapshots and return deltas.

    Useful for measuring the memory impact of loading a model or running
    inference.

    Returns
    -------
    dict
        Keys: ``delta_used_mb``, ``delta_percent``, ``delta_process_rss_mb``,
        ``delta_pod_memory_mb``.  Missing values are ``None``.
    """
    delta: dict[str, Optional[float]] = {
        "delta_used_mb": round(after.used_mb - before.used_mb, 2),
        "delta_percent": round(after.percent_used - before.percent_used, 2),
    }

    if before.process_rss_mb is not None and after.process_rss_mb is not None:
        delta["delta_process_rss_mb"] = round(
            after.process_rss_mb - before.process_rss_mb, 2,
        )
    else:
        delta["delta_process_rss_mb"] = None

    if before.pod_memory_mb is not None and after.pod_memory_mb is not None:
        delta["delta_pod_memory_mb"] = round(
            after.pod_memory_mb - before.pod_memory_mb, 2,
        )
    else:
        delta["delta_pod_memory_mb"] = None

    return delta  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_k8s_memory(value: str) -> float:
    """Parse a Kubernetes memory string (e.g. ``'1842Mi'``, ``'2Gi'``) to MB."""
    value = value.strip()
    multipliers = {
        "Ki": 1 / 1024,
        "Mi": 1.0,
        "Gi": 1024.0,
        "Ti": 1024.0 * 1024,
        "K": 1 / 1024,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * mult
    # Bare number = bytes.
    try:
        return float(value) / (1024 * 1024)
    except ValueError:
        logger.warning("Unable to parse memory value: %s", value)
        return 0.0
