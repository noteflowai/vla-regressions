"""Release unused allocator pages in this process; never target another process."""
import ctypes
from datetime import datetime, timezone
from pathlib import Path
import time


def snapshot():
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    def value(mapping, key):
        return int(mapping[key].split()[0]) * 1024 if key in mapping else None
    return {
        "process_rss_bytes": value(status, "VmRSS"),
        "anonymous_rss_bytes": value(status, "RssAnon"),
        "file_rss_bytes": value(status, "RssFile"),
        "locked_bytes": value(status, "VmLck"),
        "pinned_bytes": value(status, "VmPin"),
        "host_available_bytes": value(memory, "MemAvailable"),
    }


def trim_own_cpu_allocator():
    library = ctypes.CDLL(None)
    function = getattr(library, "malloc_trim", None)
    if function is None:
        raise RuntimeError("This Linux collector requires glibc malloc_trim")
    function.argtypes = [ctypes.c_size_t]
    function.restype = ctypes.c_int
    before = snapshot()
    started = time.perf_counter()
    result = function(0)
    elapsed = time.perf_counter() - started
    after = snapshot()
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(), "implementation": "glibc_malloc_trim_0",
        "returned": result, "elapsed_seconds": elapsed, "before": before, "after": after,
        "scope": "After model close, gc, CUDA synchronization and cache release. Only this "
                 "process's unused allocator pages; no live tensor, other process or budget "
                 "is modified. Host-memory delta can include concurrent external activity.",
    }
