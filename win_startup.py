import sys as _sys
import platform as _platform

if _sys.platform == "win32":
    _wv = _sys.getwindowsversion()
    _platform.win32_ver = lambda *a, **k: (
        str(_wv.major), f"{_wv.major}.{_wv.minor}.{_wv.build}", "", ""
    )

    def _no_wmi(*args, **kwargs):
        raise OSError("WMI disabled to avoid startup hang")

    if hasattr(_platform, "_wmi_query"):
        _platform._wmi_query = _no_wmi
