"""Read an API key without copying saved PC credentials into UI or files."""
from __future__ import annotations

from dataclasses import dataclass, field
import os

try:
    import winreg
except ImportError:  # The source also runs on non-Windows systems.
    winreg = None


@dataclass(frozen=True)
class APIKey:
    value: str = field(default='', repr=False)
    source: str = ''


def _saved_windows_key(hive, path):
    """Read only OPENAI_API_KEY; never modify the registry or environment."""
    try:
        with winreg.OpenKey(hive, path, 0, winreg.KEY_READ) as handle:
            value, kind = winreg.QueryValueEx(handle, 'OPENAI_API_KEY')
        if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str):
            return value.strip()
    except OSError:
        pass
    return ''


def resolve_api_key(override=''):
    """Resolve freshly: typed override, process, saved Windows user, system.

    Reading saved Windows variables also works when an Explorer shortcut was
    launched before a key was assigned through PowerShell.
    """
    value = override.strip()
    if value:
        return APIKey(value, 'entered')
    value = os.environ.get('OPENAI_API_KEY', '').strip()
    if value:
        return APIKey(value, 'process')
    if winreg is not None:
        for hive, path, source in (
            (winreg.HKEY_CURRENT_USER, 'Environment', 'user'),
            (winreg.HKEY_LOCAL_MACHINE,
             r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'machine'),
        ):
            value = _saved_windows_key(hive, path)
            if value:
                return APIKey(value, source)
    return APIKey()


def key_status(override=''):
    """Return presence/source only; never display any portion of a key."""
    credential = resolve_api_key(override)
    if credential.source == 'entered':
        return 'Using the key entered here for this session. Clear it to use the PC key.'
    labels = {'process': 'environment', 'user': 'Windows user', 'machine': 'Windows system'}
    if credential.source in labels:
        return f'PC API key detected ({labels[credential.source]}). Leave the key field blank to use it.'
    return 'No PC API key found. Enter a key to use AI.'
