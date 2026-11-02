"""Verbatim prompt templates with provenance.

A PromptTemplate holds the exact text the original paper used and where it came from. When a template is declared
with an ``extract`` spec, the text is read from the official repository clone at import time, and
``tests/test_registry.py`` re-extracts it and compares hashes, so a template cannot silently drift from its source.

Substitution styles mirror how the original code filled the prompt; choose the one the source code used, because
they treat braces and missing keys differently:
  'format'   str.format(**fields)                (Python .format / f-string templates)
  'percent'  text % fields                        (old-style % templates)
  'replace'  literal token replacement            (e.g. '[QUESTION]' -> text); tokens given as {token: field}
  'literal'  no substitution
"""
import ast
import functools
import hashlib
import json
import string
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PROJECT


@functools.lru_cache(maxsize=None)
def git_commit(repo):
    try:
        out = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass(frozen=True)
class Source:
    """Where a prompt, parser or dataset came from.

    repo: path to the official repository clone (absolute, or relative to the project root)
    path: file inside the repo; locator: function/variable/line range; url: public link; note: free text
    commit: expected git commit; checked by the registry tests when given
    """
    repo: str = None
    path: str = None
    locator: str = None
    url: str = None
    commit: str = None
    note: str = None

    @property
    def file(self):
        if self.repo is None or self.path is None:
            return None
        root = Path(self.repo)
        return (root if root.is_absolute() else PROJECT / root) / self.path

    def describe(self):
        parts = [p for p in (self.path, self.locator) if p]
        where = ':'.join(parts) if parts else (self.url or self.note or 'unspecified')
        return f'{where} @ {self.commit[:12]}' if self.commit else where


def extract_text(source, kind, var=None, lines=None, json_path=None):
    """Read verbatim text from a source file.

    kind='py_var'  module-level (or class-level) string assignment ``var = '...'``, evaluated with ast.literal_eval
    kind='py_dict' ``var`` is 'VAR[key]' for a dict literal assignment
    kind='file'    whole file
    kind='lines'   1-based inclusive line range (a, b), joined with newlines, no trailing newline
    kind='json'    json_path is a list of keys/indices
    """
    text = source.file.read_text()
    if kind == 'file':
        return text
    if kind == 'lines':
        a, b = lines
        return '\n'.join(text.splitlines()[a - 1:b])
    if kind == 'json':
        obj = json.loads(text)
        for k in json_path:
            obj = obj[k]
        return obj
    if kind in ('py_var', 'py_dict'):
        name, key = var, None
        if kind == 'py_dict':
            var, key = name.split('[', 1)
            key = ast.literal_eval(key.rstrip(']'))
        for node in ast.walk(ast.parse(text)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if var in targets and node.value is not None:
                value = _literal(node.value)
                return value if key is None else value[key]
        raise KeyError(f'{name} not found in {source.file}')
    raise ValueError(f'unknown extract kind {kind}')


def _literal(node):
    """ast.literal_eval plus the string idioms prompt files use: '...'.strip()/lstrip()/rstrip() and
    textwrap.dedent('...') / dedent('...') (optionally chained)."""
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {'strip', 'lstrip', 'rstrip'} and not node.args:
            return getattr(_literal(func.value), func.attr)()
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
        if name == 'dedent' and len(node.args) == 1:
            import textwrap
            return textwrap.dedent(_literal(node.args[0]))
    return ast.literal_eval(node)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    style: str = 'format'
    tokens: dict = field(default_factory=dict)   # for style='replace': {literal token: field name}
    source: Source = None
    extract: dict = None                        # extract_text kwargs used to obtain `text` from `source`
    note: str = None

    @classmethod
    def from_source(cls, name, source, style='format', tokens=None, note=None, **extract):
        return cls(name=name, text=extract_text(source, **extract), style=style, tokens=tokens or {},
                   source=source, extract=extract, note=note)

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()

    def fields(self):
        if self.style == 'format':
            return {f for _, f, _, _ in string.Formatter().parse(self.text) if f}
        if self.style == 'replace':
            return set(self.tokens.values())
        return set()

    def render(self, **values):
        if self.style == 'literal':
            if values:
                raise ValueError(f'{self.name}: literal template takes no fields')
            return self.text
        if self.style == 'format':
            missing = self.fields() - set(values)
            if missing:
                raise KeyError(f'{self.name}: missing fields {sorted(missing)}')
            return self.text.format(**values)
        if self.style == 'percent':
            return self.text % values
        if self.style == 'replace':
            out = self.text
            for token, fname in self.tokens.items():
                if token not in self.text:
                    raise KeyError(f'{self.name}: token {token!r} absent from template')
                out = out.replace(token, str(values[fname]))
            return out
        raise ValueError(f'unknown style {self.style}')

    def with_text(self, text, name=None):
        """A modified copy (for prompt-variation experiments). Provenance is dropped: it is no longer verbatim."""
        return PromptTemplate(name=name or self.name + '*', text=text, style=self.style, tokens=dict(self.tokens),
                              source=None, extract=None, note=f'modified from {self.name}')

    def verify_source(self):
        """True when re-extracting from the source still yields the stored text (None if no extract spec)."""
        if self.extract is None:
            return None
        return extract_text(self.source, **self.extract) == self.text

    def manifest(self):
        return {'name': self.name, 'sha256': self.sha256, 'style': self.style,
                'source': None if self.source is None else self.source.describe(), 'note': self.note}
