"""An optional, validated headline for new experiments; existing aggregate dicts still work.

The helper records scientific choices; it does not choose a scorer, denominator,
confidence interval, missing-data policy, or reliability threshold for the paper.
Return the normal aggregate dictionary from an experiment::

    return {'shelflife': Headline(
        metric='paired_accuracy_drop', value=0.12, ci=(0.04, 0.20),
        n=100, direction=1, unit='fraction',
        denominator='100 base questions with both conditions scored',
        null_value=0, paper_value=0.18, missing=3,
        missingness_note='3 additional questions lack a matched control',
        headline_reliable=True,
        reliability_note='All scored responses pass the preregistered parser checks',
    ).to_dict()}

Use ``value=None, ci=None`` when no estimate is available (never NaN or zero).
An unknown paper or null anchor is independently ``None``; a missing anchor does
not make an observed estimate unavailable. Use ``role='sensitivity'`` for a
separately labeled sensitivity result. Additional claim-specific diagnostics can
remain ordinary dictionary fields alongside these results.
"""
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Literal


def _finite(name: str, value: Real | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number or None, got {value!r}')
    return float(value)


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer, got {value!r}')
    return int(value)


@dataclass(frozen=True, kw_only=True)
class Headline:
    """A paper-defined metric serialized into the existing ``shelflife`` shape.

    ``direction`` is +1 if larger values support the claim, -1 if smaller do.
    ``n`` counts the observations represented by the estimate; ``denominator``
    explains their unit and inclusion rule (pairs, items, calls, macro groups,
    etc.). A weighted metric should describe its weights there. ``missing`` is
    an optional count of unavailable observations in the intended sample; it is
    not automatically subtracted from ``n`` because papers handle failures
    differently. Explain those rules in ``missingness_note``.

    All values and anchors use ``unit``. No universal range is imposed: rates,
    percentages, differences, correlations and log ratios are all supported.
    CIs must be ordered but need not contain the point estimate (some valid
    interval estimators do not). A missing CI means uncertainty is unavailable,
    not zero. Reliability is an explicit scientific assessment: None means it
    has not been assessed, not that the result is reliable.
    """

    metric: str
    value: float | None
    n: int
    direction: Literal[-1, 1]
    unit: str
    denominator: str
    ci: tuple[float, float] | None = None
    null_value: float | None = None
    paper_value: float | None = None
    missing: int | None = None
    missingness_note: str | None = None
    headline_reliable: bool | None = None
    reliability_note: str | None = None
    role: Literal['primary', 'sensitivity', 'diagnostic'] = 'primary'

    def __post_init__(self):
        for name in ('metric', 'unit', 'denominator'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('missingness_note', 'reliability_note'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} must be a nonempty string or None')
        if (isinstance(self.direction, bool) or not isinstance(self.direction, Real)
                or self.direction not in (-1, 1)):
            raise ValueError('direction must be +1 or -1')
        object.__setattr__(self, 'direction', int(self.direction))
        for name in ('value', 'null_value', 'paper_value'):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        object.__setattr__(self, 'n', _count('n', self.n))
        if self.missing is not None:
            object.__setattr__(self, 'missing', _count('missing', self.missing))
        if self.value is not None and self.n == 0:
            raise ValueError('an observed value requires n > 0; use value=None when unavailable')
        if self.ci is not None:
            if not isinstance(self.ci, (tuple, list)) or len(self.ci) != 2:
                raise ValueError('ci must contain exactly two finite endpoints or be None')
            lo, hi = (_finite('ci endpoint', v) for v in self.ci)
            if lo is None or hi is None or lo > hi:
                raise ValueError('ci endpoints must be finite and ordered low <= high')
            if self.value is None:
                raise ValueError('ci must be None when value is unavailable')
            object.__setattr__(self, 'ci', (lo, hi))
        if self.headline_reliable is not None and type(self.headline_reliable) is not bool:
            raise ValueError('headline_reliable must be True, False, or None')
        if self.role not in ('primary', 'sensitivity', 'diagnostic'):
            raise ValueError('role must be primary, sensitivity, or diagnostic')

    def to_dict(self) -> dict:
        """Return an independent, strict-JSON-compatible ``aggregate()['shelflife']`` value."""
        result = asdict(self)
        result['ci'] = list(self.ci) if self.ci is not None else None
        return result
