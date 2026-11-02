"""Load pure functions and constants from an original repository file without running its module body.

Original scripts often parse CLI flags, import retired SDKs, or set API keys at import time. ``load_functions``
compiles only the named top-level function definitions (plus named constant assignments) into a fresh namespace,
so tests can call the authors' own prompt builders and parsers.
"""
import ast
from pathlib import Path

# Standard-library modules whose imports are kept; anything else (SDKs, CLI parsing) is dropped.
SAFE_MODULES = {'re', 'json', 'math', 'string', 'collections', 'itertools', 'random', 'typing', 'functools',
                'unicodedata', 'dataclasses', 'statistics', 'operator', 'copy', 'enum'}


def load_functions(path, names, constants=(), extra_globals=None):
    path = Path(path)
    tree = ast.parse(path.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.Import) and all(a.name.split('.')[0] in SAFE_MODULES for a in node.names):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or '').split('.')[0] in SAFE_MODULES:
            keep.append(node)
    namespace = dict(extra_globals or {})
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), 'exec'), namespace)
    missing = set(names) - set(namespace)
    if missing:
        raise KeyError(f'{sorted(missing)} not defined in {path}')
    return namespace
