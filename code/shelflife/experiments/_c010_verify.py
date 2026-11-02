"""C010 automated bibliography extraction and verification, loaded from the panel task's own code.

The paper (Walters & Wilder 2023) verified every citation by hand; there is no original code. The only automated
scorer in this project is the extract/verify pipeline in ``shelflife/experiments/helpers/C010``: v1 = ``extract.py`` + ``verify.py``, v2 =
``extract_v2.py`` (wraps v1) + ``verify_v2.py`` (replaces ``verify.parse`` and adds lookup-failure handling).

We do not copy or edit that code. Each version is loaded into a *private* module object (so v1 stays pristine when v2
monkeypatches ``verify.parse``/``verify.http_get``), and its HTTP layer is replaced by a reader that

  * reads the panel's response cache (``tasks/C010/cache``) read-only,
  * falls back to a writable cache outside ``experiments/`` (``code/.cache/c010_verify``) for new lookups, and
  * in ``offline`` mode never touches the network: a cache miss is reported as a failed lookup (status
    ``'error:offline'``), which the v2 rules turn into ``unverifiable/lookup_failed`` rather than ``fabricated``.
    Every such miss is counted per citation (``verify_tracked``): a lookup that was never attempted is not evidence
    about the citation, so callers must not turn it into a fabrication rate (see C010 ``aggregate``).

Offline re-verification of a citation whose every lookup URL is cached is exactly what the panel computed.
"""
import functools
import hashlib
import json
import sys
import threading
import types
from pathlib import Path

from ..paths import CODE, HELPERS

TASK = HELPERS / 'C010'
PANEL_CACHE = TASK / 'cache'
LOCAL_CACHE = CODE / '.cache' / 'c010_verify'


def _load(path, name, inject=None):
    """Execute ``path`` as a fresh module object. Compiled from source text (no importlib loader), so no
    __pycache__ is ever written next to the read-only experiment code."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    saved = {k: sys.modules.get(k) for k in (inject or {})}
    sys.modules.update(inject or {})
    try:
        exec(compile(Path(path).read_text(), str(path), 'exec'), mod.__dict__)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


class _NoSleep(types.ModuleType):
    def __init__(self, real):
        super().__init__('time')
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def sleep(_):
        return None


_tls = threading.local()   # per-thread count of lookups answered 'error:offline' (never attempted)


def _cached_reader(verify_mod, offline):
    original = verify_mod.http_get   # writes into verify_mod.CACHE, which we point at LOCAL_CACHE

    def http_get(url, source):
        key = hashlib.sha256(url.encode()).hexdigest()
        for root in (PANEL_CACHE, LOCAL_CACHE):
            path = root / source / key[:2] / f'{key}.json'
            if path.exists():
                rec = json.loads(path.read_text())
                return rec['status'], rec['body']
        if offline:
            _tls.offline_misses = getattr(_tls, 'offline_misses', 0) + 1
            return 'error:offline', None
        return original(url, source)
    return http_get


@functools.lru_cache(maxsize=None)
def pipeline(version='v2', offline=True):
    """Return a namespace with ``extract(text)`` and ``verify(citation)`` for panel version 'v1' or 'v2'."""
    import time as _time
    v = _load(TASK / 'verify.py', f'_c010_verify_{version}_{offline}')
    v.CACHE = LOCAL_CACHE
    if offline:
        v.time = _NoSleep(_time)
    v.http_get = _cached_reader(v, offline)
    ext1 = _load(TASK / 'extract.py', f'_c010_extract_{version}_{offline}')
    if version == 'v1':
        return types.SimpleNamespace(version='v1', extract=ext1.extract, verify=v.verify, parse=v.parse, module=v)
    if version != 'v2':
        raise ValueError(version)
    v2 = _load(TASK / 'verify_v2.py', f'_c010_verify2_{offline}', inject={'verify': v})
    if offline:
        v2.time = _NoSleep(_time)
    ext2 = _load(TASK / 'extract_v2.py', f'_c010_extract2_{offline}', inject={'extract': ext1})
    return types.SimpleNamespace(version='v2', extract=ext2.extract_v2, verify=v2.verify, parse=v2.parse, module=v)


def verify_tracked(p, citation):
    """``p.verify(citation)`` plus ``offline_cache_misses``: how many catalogue lookups the verdict needed but
    could not make because the URL was absent from every cache in offline mode (0 when online or fully cached)."""
    _tls.offline_misses = 0
    v = dict(p.verify(citation))
    v['offline_cache_misses'] = _tls.offline_misses
    return v

