"""Process-local Windows hybrid CPU policy; other platforms retain scheduler defaults."""
import ctypes
import logging
import os
import sys

logger = logging.getLogger("tts-service")

def select_performance_cpus(records, allowed):
    """Respect existing affinity and leave homogeneous CPU sets unchanged."""
    available = [(cpu, efficiency) for group, cpu, efficiency in records
                 if group == 0 and cpu in allowed]
    if any(group != 0 for group, _, _ in records):
        return list(allowed)
    if not available or len({efficiency for _, efficiency in available}) < 2:
        return list(allowed)
    fastest = max(efficiency for _, efficiency in available)
    return sorted({cpu for cpu, efficiency in available if efficiency == fastest})

def windows_cpu_records():
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True).GetSystemCpuSetInformation
    api.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                    wintypes.HANDLE, wintypes.ULONG]
    api.restype = wintypes.BOOL
    needed = wintypes.ULONG()
    api(None, 0, ctypes.byref(needed), None, 0)
    if not needed.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(needed.value)
    if not api(buffer, len(buffer), ctypes.byref(needed), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    records = []
    offset = 0
    while offset + 8 <= needed.value:
        size = ctypes.c_uint32.from_buffer(buffer, offset).value
        kind = ctypes.c_uint32.from_buffer(buffer, offset + 4).value
        if size < 8 or offset + size > needed.value:
            raise ValueError("Invalid Windows CPU set record")
        if kind == 0 and size >= 20:
            group = ctypes.c_uint16.from_buffer(buffer, offset + 12).value
            cpu = buffer.raw[offset + 14]
            efficiency = buffer.raw[offset + 18]
            records.append((group, cpu, efficiency))
        offset += size
    return records

def configure_cpu_policy():
    policy = os.getenv("TTS_CPU_POLICY", "performance").lower()
    if policy == "system" or sys.platform != "win32":
        logger.info("CPU policy: %s (scheduler defaults retained)", policy)
        return
    if policy != "performance":
        logger.warning("Unknown TTS_CPU_POLICY=%s; retaining current affinity", policy)
        return
    try:
        import psutil
        process = psutil.Process()
        before = process.cpu_affinity()
        selected = select_performance_cpus(windows_cpu_records(), before)
        if selected != before:
            process.cpu_affinity(selected)
        logger.info("CPU policy: performance; affinity=%s (previous=%s)",
                    process.cpu_affinity(), before)
    except Exception:
        logger.exception("CPU policy unavailable; retaining scheduler settings")

