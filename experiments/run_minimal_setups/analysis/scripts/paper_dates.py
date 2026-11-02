"""Resolve paper release dates from the source-reviewed corpus ledger."""
import json
import re
from pathlib import Path


def _identity(url):
    match = re.search(r'arxiv.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})', url or '')
    return 'arxiv:' + match[1] if match else (url or '').replace('http://', 'https://').rstrip('/')


def apply_release_dates(metadata):
    registry = Path(__file__).resolve().parents[4] / 'data/paper-releases.json'
    papers = json.loads(registry.read_text())['papers']
    lookup = {}
    for paper in papers:
        for url in [paper['source_url'], *paper['aliases']]:
            lookup[_identity(url)] = paper
    resolved = {}
    for key, record in metadata.items():
        paper = lookup.get(_identity(record.get('url')))
        if paper is None:
            raise ValueError(f'{key}: paper needs a reviewed release-date source')
        note = paper['note']
        if paper.get('caveat'):
            note += ' ' + paper['caveat']
        resolved[key] = dict(record, paper_date=paper['first_public_date'], paper_date_note=note,
                             paper_date_source=paper['source_url'], paper_date_kind=paper['date_kind'],
                             paper_date_status=paper['status'])
    return resolved
