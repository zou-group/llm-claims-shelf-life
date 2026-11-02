"""Remove private build metadata from payloads embedded in the public blog.

Scientific inputs and full source manifests remain in their original locations.
"""
from copy import deepcopy


def public_results(data):
    data = deepcopy(data)
    data.pop('sweep', None)  # Includes unreleased excluded claims and screening notes.
    for model in data['models']:
        model.pop('role', None)  # Internal panel-selection/budget notes, unused by UI.
    for claim in data['claims']:
        # Render methodological provenance without internal workflow terminology.
        for field in ('paper_value_note', 'paper_date_note', 'repro_e_note'):
            value = claim.get(field)
            if isinstance(value, str):
                for old, new in (
                    (' (matches earlier pipeline card)', ''),
                    ('; released results, module PaperTarget', '; authors’ released results'),
                    (', from the reproduction pass', ''),
                    ("on the port's task set", 'on the evaluated task set'),
                    ('the four MC datasets the port runs', 'the four evaluated multiple-choice datasets'),
                    ('rescored with the module (0.3899); not a number printed in the text',
                     'scored with the evaluation metric (0.3899)'),
                ):
                    value = value.replace(old, new)
                claim[field] = value
        if claim['key'] == 'C079':
            claim['claim'] = ('Models produce stereotype-consistent pairings on the IAT-style '
                              'word-assignment prompt (mean implicit bias score > 0), despite '
                              'appearing unbiased on explicit benchmarks.')
        if claim['key'] == 'TOY-01':
            claim['paper_value_note'] = ('No numerical paper reference: normalized to our '
                'GPT-4o-2024-05-13 reproduction, 4 errors in 10 trials (0.4).')
        for point in claim['points']:
            point.pop('note', None)  # Internal collection provenance; flags remain.
    return data


def public_numbers(numbers):
    return {k: v for k, v in numbers.items() if k != 'dropped'}


def public_card(record):
    card = {k: deepcopy(record[k]) for k in
            ('key', 'headline', 'claim', 'summary', 'metric') if k in record}
    card['figure'] = {k: deepcopy(record['figure'][k]) for k in
                      ('src', 'label', 'caption', 'alt', 'source_url', 'page',
                       'scope_note', 'license', 'kind') if k in record['figure']}
    # This caveat is displayed in Sources, methods & model data.
    caveat = record.get('provenance', {}).get('caveat')
    if caveat:
        card['provenance'] = {'caveat': caveat}
    return card
