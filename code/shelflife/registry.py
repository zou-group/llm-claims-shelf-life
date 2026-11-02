"""Experiment registry: one registered config per claim.

An Experiment declares
  * variants        named prompt/strategy/parameter settings. The paper's protocol is a variant with
                    faithful=True; prompt or strategy changes are new variants (``Variant.override``) and are
                    marked faithful=False automatically, so a modified run can never be mistaken for a faithful one;
  * items()         the dataset rows for a variant, loaded from the official release;
  * program()       the prompting strategy (shelflife.program) that turns an item into model calls;
  * score()         per-output scoring that follows the original scorer;
  * aggregate()     the paper's metric over scored rows;
  * targets         numbers from the paper, with tolerance and an offline way to recompute them where possible;
  * prior_runs      earlier runs in experiments/ that this config should reproduce exactly (prompt parity, rescoring).

Register with the ``@register`` class decorator inside a module in ``shelflife/experiments/``.
"""
import dataclasses
import hashlib
import importlib
import json
import pkgutil
from dataclasses import dataclass, field
from typing import Callable

from .prompts import PromptTemplate, Source

REGISTRY = {}
IMPORT_ERRORS = {}   # module name -> error; one broken claim module must not hide the others


def _jsonable(obj):
    if isinstance(obj, PromptTemplate):
        return obj.manifest()
    if isinstance(obj, Source):
        return obj.describe()
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if callable(obj):
        return getattr(obj, '__qualname__', repr(obj))
    return obj


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    prompts: dict = field(default_factory=dict)     # role -> PromptTemplate
    params: dict = field(default_factory=dict)      # original generation parameters (temperature, max_tokens, ...)
    strategy: str = 'single_turn'                   # label of the prompting strategy the program implements
    options: dict = field(default_factory=dict)     # experiment-specific knobs (subset, conditions, n per item, ...)
    repeats: int = 1                                # independent samples per item in the original protocol
    faithful: bool = True                           # matches the paper's protocol (prompts, strategy, params)
    deviations: tuple = ()                          # documented unavoidable deviations even when faithful
    parent: str = None

    def manifest(self):
        return _jsonable(self)

    @property
    def fingerprint(self):
        text = json.dumps(self.manifest(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()

    def override(self, name, *, description=None, prompts=None, params=None, options=None, strategy=None,
                 repeats=None, deviations=()):
        """A new variant with changed prompts/params/options/strategy. Prompt and strategy changes make it
        non-faithful; so do parameter changes. Option changes (e.g. a different subset size) keep the parent's
        faithfulness unless ``deviations`` are declared."""
        new_prompts = {**self.prompts, **(prompts or {})}
        for role, template in list(new_prompts.items()):
            if isinstance(template, str):
                new_prompts[role] = self.prompts[role].with_text(template)
        changed_protocol = (new_prompts != self.prompts or (params is not None and {**self.params, **params} != self.params)
                            or (strategy is not None and strategy != self.strategy))
        return dataclasses.replace(
            self, name=name, description=description or f'{self.name} with overrides',
            prompts=new_prompts, params={**self.params, **(params or {})},
            options={**self.options, **(options or {})}, strategy=strategy or self.strategy,
            repeats=self.repeats if repeats is None else repeats,
            faithful=self.faithful and not changed_protocol and not deviations,
            deviations=tuple(self.deviations) + tuple(deviations), parent=self.name)


OFFLINE_CHECKS = {'archived_outputs', 'prior_run', 'released_data'}


@dataclass(frozen=True)
class PaperTarget:
    """A number reported in the paper that the implementation should recover.

    check: how it is verified
      'archived_outputs'  our scorer/metric applied to outputs the authors released (fully offline)
      'prior_run'         our scorer/metric applied to saved outputs of an earlier original-model run in experiments/
      'released_data'     a paper number computable from released data alone (dataset sizes/statistics, human
                          baselines, the paper's own arithmetic); anchors the data loading, not the model scoring
      'live'              needs new API calls on the original model (opt-in, costs money)
      'unverifiable'      no released outputs and the original model is unreachable; documented only
    compute(exp) -> float returns the reproduced value for offline checks.
    known_discrepancy: when the reproduced value is known NOT to match (e.g. the only reachable snapshot is a proxy
    that behaves differently), say why. The target test then fails if the value unexpectedly falls inside the
    tolerance, so the documentation cannot go stale silently. Never widen a tolerance to absorb a miss.
    """
    name: str
    value: float
    tolerance: float
    where: str
    model: str
    variant: str
    check: str
    compute: Callable = None
    note: str = None
    known_discrepancy: str = None   # documented, investigated failure to reproduce; the test then expects a miss

    def __post_init__(self):
        if self.check not in {'archived_outputs', 'prior_run', 'released_data', 'live', 'unverifiable'}:
            raise ValueError(f'bad check kind {self.check}')
        if self.check in OFFLINE_CHECKS and self.compute is None:
            raise ValueError(f'{self.name}: offline targets need compute()')


@dataclass(frozen=True)
class PriorRun:
    """An earlier run whose requests this config must reproduce.

    path: run directory relative to the project root; variant: our variant name that should match it.
    cases: cases file the run sent (relative to project), if any. known_deviations: documented, intentional
    differences between our faithful rendering and what that run sent (each one a short justification).
    """
    path: str
    variant: str
    description: str = ''
    cases: str = None
    known_deviations: tuple = ()
    no_requests: bool = False      # run directory exists but sent nothing (e.g. aborted); listed for completeness


class Experiment:
    claim_id: str = None
    title: str = None
    claim: str = None
    paper: dict = {}
    source: Source = None
    default_variant: str = None
    targets: tuple = ()
    prior_runs: tuple = ()
    excluded_runs: dict = {}   # {run dir relative to project: reason} for run dirs deliberately not registered
    run_manifest: str = None  # optional explicit project-relative JSON file, separate from execution source
    maturity: str = 'ready'    # 'draft' while integrating: generic tests skip it unless SHELFLIFE_INCLUDE_DRAFTS=1

    # --- to implement -------------------------------------------------------------------------
    def build_variants(self):
        """Return {name: Variant}."""
        raise NotImplementedError

    def items(self, variant, limit=None):
        """Dataset rows for this variant. Every row needs a unique, stable 'id'."""
        raise NotImplementedError

    def program(self, variant):
        raise NotImplementedError

    def score(self, item, output, variant):
        """Score one program output (dict from Program.forward). Return a JSON-serializable dict."""
        raise NotImplementedError

    def aggregate(self, rows, variant):
        """rows: list of {'item','output','score','model','repeat'} for one model. Return metric dict."""
        raise NotImplementedError

    # --- provided ---------------------------------------------------------------------------------
    def all_prior_runs(self):
        """Legacy declarations plus validated registrations from the explicit run manifest."""
        from .run_manifest import merged_runs
        return merged_runs(self)[0]

    def all_excluded_runs(self):
        """Legacy exclusions plus validated exclusions from the explicit run manifest."""
        from .run_manifest import merged_runs
        return merged_runs(self)[1]

    @property
    def variants(self):
        if not hasattr(self, '_variants'):
            built = self.build_variants()
            for name, v in built.items():
                if v.name != name:
                    raise ValueError(f'variant key {name} != name {v.name}')
            self._variants = built
        return self._variants

    def variant(self, name=None):
        return self.variants[name or self.default_variant]

    def add_variant(self, variant):
        """Register a runtime variant (e.g. from Variant.override) so runners and tests can address it by name."""
        if variant.name in self.variants and self.variants[variant.name] != variant:
            raise ValueError(f'variant {variant.name} already exists with different settings')
        self.variants[variant.name] = variant
        return variant

    def cases(self, variant, limit=None):
        """Legacy-format cases for single-turn programs (what cases.jsonl held in earlier runs)."""
        program = self.program(variant)
        if not hasattr(program, 'case'):
            raise TypeError(f'{type(program).__name__} is not single-turn; cases depend on responses')
        return [program.case(item) for item in self.items(variant, limit)]

    # --- rescoring saved runs (override the two hooks when record ids or outputs differ) -------------
    def record_item(self, record, items_by_id):
        """Dataset row a saved record answers. Default: the record's case id (or runner item id)."""
        return items_by_id.get(record.get('case_id') or record.get('item_id'))

    def record_output(self, record):
        """Program-style output for a saved record. Runner rows already hold one; legacy rows hold content."""
        if 'output' in record and 'trace' in record:
            return record['output']
        return {'content': record.get('content') or '', 'finish_reason': record.get('finish_reason'),
                'status': record.get('status')}

    def rescore(self, run_dir, variant=None, model=None, streams=None, allow_unmatched=False):
        """Score a saved run with this config and aggregate per model: {model: aggregate(...)}.

        Reads every response stream (shelflife.replay.run_records). Records that map to no item raise, unless
        allow_unmatched, because a silent mismatch would change the denominator.
        """
        from .replay import model_key, run_records
        variant = self.variant(variant) if isinstance(variant, str) or variant is None else variant
        items = {x['id']: x for x in self.items(variant)}
        rows, unmatched = {}, []
        for rec in run_records(run_dir, streams):
            m = model_key(rec)
            if model is not None and m != model:
                continue
            item = self.record_item(rec, items)
            if item is None:
                unmatched.append(rec.get('case_id') or rec.get('item_id'))
                continue
            usable = rec.get('status') in {'ok', 'truncated', 'empty', 'complete'}
            output = self.record_output(rec) if usable else None
            rows.setdefault(m, []).append({'item': item, 'output': output, 'model': m, 'repeat': rec.get('repeat', 0),
                                           'status': rec.get('status'),
                                           'score': None if output is None else self.score(item, output, variant)})
        if unmatched and not allow_unmatched:
            raise ValueError(f'{len(unmatched)} saved records match no item of {variant.name}: {unmatched[:3]}')
        return {m: self.aggregate(r, variant) for m, r in sorted(rows.items())}

    def manifest(self):
        return {'claim_id': self.claim_id, 'title': self.title, 'claim': self.claim, 'paper': self.paper,
                'source': None if self.source is None else self.source.describe(),
                'default_variant': self.default_variant,
                'variants': {k: v.manifest() for k, v in self.variants.items()},
                'targets': [_jsonable(t) for t in self.targets],
                'prior_runs': [_jsonable(p) for p in self.prior_runs]}


def register(cls):
    exp = cls()
    if not exp.claim_id:
        raise ValueError(f'{cls.__name__} lacks claim_id')
    if exp.claim_id in REGISTRY and type(REGISTRY[exp.claim_id]).__qualname__ != cls.__qualname__:
        raise ValueError(f'duplicate registration for {exp.claim_id}')
    REGISTRY[exp.claim_id] = exp
    return cls


def load_all():
    from . import experiments
    for mod in pkgutil.iter_modules(experiments.__path__):
        if not mod.name.startswith('_') and mod.name not in IMPORT_ERRORS:
            try:
                importlib.import_module(f'{experiments.__name__}.{mod.name}')
            except Exception as e:  # surfaced by tests/test_registry.py::test_all_modules_import
                IMPORT_ERRORS[mod.name] = f'{type(e).__name__}: {e}'
    return REGISTRY


def get(claim_id):
    load_all()
    return REGISTRY[claim_id]


def list_experiments(include_drafts=True):
    exps = load_all()
    return sorted(c for c, e in exps.items() if include_drafts or e.maturity != 'draft')


def skip_if_draft(exp):
    """For generic tests: drafts are skipped unless SHELFLIFE_INCLUDE_DRAFTS=1 (set by `shelflife check`)."""
    import os
    import pytest
    if exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1':
        pytest.skip(f'{exp.claim_id} is a draft (run `python -m shelflife.cli check {exp.claim_id}`)')
