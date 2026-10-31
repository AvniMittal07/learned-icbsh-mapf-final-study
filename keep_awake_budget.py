"""Budget runner with a process-lifetime Windows idle-standby prevention request.

Power settings are not changed. The request is released in finally, and Windows
also releases it if this process terminates. Keep the display/system awake so
Modern Standby cannot invalidate a short wall-clock benchmark.
"""
import os
from mapf.compute_budget import main


if __name__=='__main__':
    setter=None
    if os.name=='nt':
        import ctypes
        setter=ctypes.WinDLL('kernel32',use_last_error=True).SetThreadExecutionState
        setter.argtypes=[ctypes.c_uint];setter.restype=ctypes.c_uint
        if not setter(0x80000003):
            raise ctypes.WinError(ctypes.get_last_error())
    try:main()
    finally:
        if setter is not None:setter(0x80000000)
