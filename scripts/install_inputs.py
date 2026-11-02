"""Download and verify the input files that are not stored in git.

Every file is listed in scripts/inputs.json with its path, size, sha256 and where it comes from:

  git    a file of a public GitHub repository at a pinned commit (fetched from raw.githubusercontent.com)
  hf     a file of a Hugging Face dataset at a pinned revision
  url    a plain download (the original papers' releases on GitHub, OSF, ...); zip_member entries name the file
         inside that archive

Run from the repository root (only the Python standard library is needed):

    python scripts/install_inputs.py                    # download everything that is missing or invalid
    python scripts/install_inputs.py --check            # report which files are present and valid
    python scripts/install_inputs.py --claims C7 C10    # only the files those claims read
    python scripts/install_inputs.py --from DIR         # copy from a directory or tar that already holds them

A file is written only after its sha256 matches the manifest; existing valid files are left alone.
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'scripts' / 'inputs.json'
CACHE = ROOT / '.cache' / 'inputs'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def status(entries):
    missing, bad = [], []
    for e in entries:
        p = ROOT / e['path']
        if not p.exists():
            missing.append(e)
        elif p.stat().st_size != e['bytes'] or sha256(p) != e['sha256']:
            bad.append(e)
    return missing, bad


def fetch(url, dest=None):
    """Download url into dest (cached) and return its path. A transfer that stops early is resumed with a Range
    request, up to 8 attempts; dest defaults to a cache file named after the url."""
    dest = dest or CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.partial')
    if tmp.exists():
        tmp.unlink()
    total, last = None, None
    for attempt in range(8):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {'User-Agent': 'shelflife-install-inputs/1.0'}
        if have:
            headers['Range'] = f'bytes={have}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
                if have and r.status != 206:      # server ignored the range: start over
                    have = 0
                    tmp.unlink()
                if total is None:
                    length = r.headers.get('Content-Length')
                    total = (have + int(length)) if length else None
                with open(tmp, 'ab' if have else 'wb') as out:
                    shutil.copyfileobj(r, out)
        except Exception as e:  # noqa: BLE001  (transient network errors: resume)
            last = e
            continue
        if total is None or tmp.stat().st_size == total:
            tmp.replace(dest)
            return dest
        if tmp.stat().st_size > total:
            tmp.unlink()
        last = OSError(f'truncated transfer ({tmp.stat().st_size if tmp.exists() else 0} of {total} bytes)')
    if tmp.exists():
        tmp.unlink()
    raise OSError(f'download failed after 8 attempts: {url} ({last})')


def open_source(e):
    """A readable binary stream for entry e, fetched from where it lives."""
    kind = e['kind']
    if kind == 'git':
        url = f"https://raw.githubusercontent.com/{e['repo']}/{e['commit']}/{urllib.parse.quote(e['upstream_path'])}"
        return open(fetch(url), 'rb')
    if kind == 'hf':
        url = f"https://huggingface.co/datasets/{e['repo']}/resolve/{e['revision']}/{urllib.parse.quote(e['upstream_path'])}"
        return open(fetch(url), 'rb')
    if kind == 'url':
        archive = fetch(e['url'])
        if e.get('zip_member'):
            with zipfile.ZipFile(archive) as z:
                return io.BytesIO(z.read(e['zip_member']))
        return open(archive, 'rb')
    sys.exit(f"unknown kind {kind!r} for {e['path']}")


def cache_path(e):
    kind = e['kind']
    if kind == 'git':
        return CACHE / hashlib.sha256(f"https://raw.githubusercontent.com/{e['repo']}/{e['commit']}/{urllib.parse.quote(e['upstream_path'])}".encode()).hexdigest()[:24]
    if kind == 'hf':
        return CACHE / hashlib.sha256(f"https://huggingface.co/datasets/{e['repo']}/resolve/{e['revision']}/{urllib.parse.quote(e['upstream_path'])}".encode()).hexdigest()[:24]
    return CACHE / hashlib.sha256(e['url'].encode()).hexdigest()[:24]


def put(e, src):
    """Write src to e['path'] if its sha256 matches; otherwise leave nothing and return False."""
    dest = ROOT / e['path']
    tmp = dest.with_name(dest.name + '.partial')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src, open(tmp, 'wb') as out:
        shutil.copyfileobj(src, out)
    if sha256(tmp) != e['sha256']:
        tmp.unlink()
        return False
    tmp.replace(dest)
    return True


def download(e):
    """Fetch and verify one entry; a truncated transfer is retried with a fresh download."""
    for attempt in range(3):
        try:
            if put(e, open_source(e)):
                return True
        except (OSError, EOFError, zipfile.BadZipFile, KeyError) as err:  # truncated or unreadable download
            print(f'  retrying after: {err}', flush=True)
        cached = cache_path(e)
        if cached.exists():
            cached.unlink()
    return False


def install_from(entries, source):
    source = Path(source)
    if source.is_dir():
        def opener(rel):
            p = source / rel
            return open(p, 'rb') if p.exists() else None
    elif tarfile.is_tarfile(source):
        tar = tarfile.open(source)
        members = {m.name.removeprefix('./'): m for m in tar.getmembers() if m.isfile()}

        def opener(rel):
            m = members.get(rel)
            return tar.extractfile(m) if m else None
    else:
        sys.exit(f'{source} is neither a directory nor a tar archive')
    done = 0
    for e in entries:
        dest = ROOT / e['path']
        if dest.exists() and sha256(dest) == e['sha256']:
            continue
        src = opener(e['path'])
        if src is None:
            print(f"not in bundle: {e['path']}")
            continue
        put(e, src)
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='only report; download nothing')
    ap.add_argument('--from', dest='source', help='directory or tar archive that already holds the files')
    ap.add_argument('--claims', nargs='*', help="only files read by these claims (post IDs such as C7, or config IDs such as C086)")
    a = ap.parse_args()
    entries = json.loads(MANIFEST.read_text())['files']
    if a.claims:
        wanted = set(a.claims)
        entries = [e for e in entries if wanted & set(e.get('claims', []))]
    missing, bad = status(entries)
    if a.check:
        for e in missing:
            print(f"missing: {e['path']}")
        for e in bad:
            print(f"invalid: {e['path']}")
        ok = len(entries) - len(missing) - len(bad)
        print(f'{ok}/{len(entries)} input files present and verified')
        sys.exit(0 if ok == len(entries) else 1)
    todo = missing + bad
    if not todo:
        print(f'all {len(entries)} input files present and verified')
        return
    if a.source:
        done = install_from(todo, a.source)
    else:
        done, failed = 0, []
        for i, e in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {e['path']}", flush=True)
            if download(e):
                done += 1
            else:
                failed.append(e['path'])
        for path in failed:
            print(f'checksum mismatch after 3 downloads, not written: {path}')
    missing, bad = status(entries)
    print(f'installed {done}; {len(entries) - len(missing) - len(bad)}/{len(entries)} input files present and verified')
    sys.exit(0 if not (missing or bad) else 1)


if __name__ == '__main__':
    main()
