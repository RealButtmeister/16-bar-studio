"""Keep generated songs outside a one-file executable's temporary resources."""
from pathlib import Path
import sys


def documents_directory():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        import uuid

        class GUID(ctypes.Structure):
            _fields_ = [('Data1', wintypes.DWORD), ('Data2', wintypes.WORD),
                        ('Data3', wintypes.WORD), ('Data4', ctypes.c_ubyte * 8)]

        folder_id = GUID.from_buffer_copy(uuid.UUID('FDD39AD0-238F-46AF-ADB4-6C85480369C7').bytes_le)
        shell = ctypes.WinDLL('shell32')
        ole = ctypes.WinDLL('ole32')
        shell.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD,
                                              wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
        shell.SHGetKnownFolderPath.restype = ctypes.c_long
        ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole.CoTaskMemFree.restype = None
        result = ctypes.c_void_p()
        if shell.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(result)) == 0:
            try:
                return Path(ctypes.wstring_at(result))
            finally:
                ole.CoTaskMemFree(result)
    return Path.home() / 'Documents'


def output_base(source_base):
    if getattr(sys, 'frozen', False):
        return documents_directory() / '16 Bar Studio'
    return Path(source_base)
