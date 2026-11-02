"""The optional headline contract catches ambiguous or unserializable results."""
import json

import numpy as np
import pytest

from shelflife.results import Headline


def headline(**changes):
    fields = dict(metric='accuracy_drop', value=0.12, ci=(0.04, 0.20),
                  n=100, direction=1, unit='fraction',
                  denominator='base questions with both conditions scored')
    return Headline(**(fields | changes))


def test_existing_shape_and_strict_json():
    metric = headline(null_value=0, paper_value=0.18, missing=3,
                      missingness_note='Unpaired questions excluded',
                      headline_reliable=False, reliability_note='Parser validity is below 95%')
    aggregate = {'shelflife': metric.to_dict(), 'parser_validity': 0.9}
    restored = json.loads(json.dumps(aggregate, allow_nan=False))
    assert restored['shelflife']['value'] == 0.12
    assert restored['shelflife']['ci'] == [0.04, 0.2]
    assert restored['shelflife']['null_value'] == 0
    assert restored['shelflife']['paper_value'] == 0.18
    assert restored['shelflife']['headline_reliable'] is False
    assert restored['shelflife']['missing'] == 3
    assert restored['shelflife']['role'] == 'primary'
    assert restored['parser_validity'] == 0.9


def test_missing_estimate_and_unknown_anchors_are_independent():
    observed = headline().to_dict()
    assert observed['value'] == 0.12
    assert observed['paper_value'] is None
    assert observed['null_value'] is None
    assert observed['headline_reliable'] is None
    assert observed['missing'] is None
    unavailable = headline(value=None, ci=None, n=0, paper_value=0.18, missing=100).to_dict()
    assert unavailable['value'] is None
    assert unavailable['ci'] is None
    assert unavailable['paper_value'] == 0.18
    json.dumps(unavailable, allow_nan=False)
    # Observations can exist even when a reference/control needed by the estimator is missing.
    assert headline(value=None, ci=None, n=100).n == 100


@pytest.mark.parametrize('field', ['value', 'paper_value', 'null_value'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True, '0.1'])
def test_nonfinite_or_nonnumeric_values_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        headline(**{field: value})


@pytest.mark.parametrize('ci', [(0.3, 0.1), (0, float('nan')), (None, None),
                              (0, float('inf')), (0,), (0, 1, 2), '01', (False, 1)])
def test_invalid_intervals_rejected(ci):
    with pytest.raises(ValueError, match='ci'):
        headline(ci=ci)


def test_missing_value_cannot_have_interval_or_empty_sample_have_value():
    with pytest.raises(ValueError, match='unavailable'):
        headline(value=None)
    with pytest.raises(ValueError, match='n > 0'):
        headline(n=0)


@pytest.mark.parametrize('field', ['n', 'missing'])
@pytest.mark.parametrize('value', [-1, 1.5, True, '3'])
def test_invalid_counts_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        headline(**{field: value})


@pytest.mark.parametrize('field', ['metric', 'unit', 'denominator'])
@pytest.mark.parametrize('value', ['', '  ', None, 7])
def test_scientific_definition_required(field, value):
    with pytest.raises(ValueError, match=field):
        headline(**{field: value})


@pytest.mark.parametrize('direction', [0, 2, True, '1', None, 1 + 0j])
def test_invalid_direction_rejected(direction):
    with pytest.raises(ValueError, match='direction'):
        headline(direction=direction)


def test_does_not_impose_rate_range_or_interval_estimator():
    assert headline(value=-2, ci=(-1, 0), direction=-1, unit='log ratio').value == -2
    assert headline(value=120, ci=None, unit='tokens').ci is None
    # Missing can exceed n for a heavily filtered sample; paper-specific policy owns this decision.
    assert headline(n=2, missing=50, role='sensitivity').missing == 50


@pytest.mark.parametrize('fields', [{'headline_reliable': 1}, {'role': 'main'},
                                  {'missingness_note': ''}, {'reliability_note': False}])
def test_invalid_annotations_rejected(fields):
    with pytest.raises(ValueError):
        headline(**fields)


def test_normalizes_numpy_numbers_and_returns_independent_dictionary():
    metric = headline(value=np.float64(0.12), n=np.int64(100), missing=np.int64(3),
                      ci=[np.float64(0.04), np.float64(0.2)])
    first = metric.to_dict()
    json.dumps(first, allow_nan=False)
    first['ci'][0] = 100
    assert metric.to_dict()['ci'] == [0.04, 0.2]
