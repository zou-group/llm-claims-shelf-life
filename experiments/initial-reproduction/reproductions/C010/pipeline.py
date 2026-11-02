"""Generation control and evidence-based summaries; never LLM-judge citation reality."""
import copy
import re

BIB_HEADING = re.compile(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:bibliography|references|works cited)(?:\*\*)?\s*:?[ \t]*$')
SUBSTANTIVE_FIELDS = ('authorship', 'title', 'date', 'journal', 'volume_issue_pages', 'publisher_organization')

def run_conversation(case, call, max_calls=3):
    """Fresh case conversation, exact 'Please continue', at most two continuations.

    Stop decisions are generation logistics; initial-bibliography inclusion must
    subsequently be approved from exact spans by a reviewer. No cap is attributed
    to the original UI, and a cap-exhausted/empty response is never a clean result.
    """
    if not 1 <= max_calls <= 3:
        raise ValueError('This predeclared pilot permits 1–3 calls only')
    messages = copy.deepcopy(case['messages'])
    stages = []
    for turn in range(max_calls):
        item = {**case, 'id': f"{case['id']}-turn-{turn}", 'messages': copy.deepcopy(messages)}
        r = call(item)
        stages.append(r)
        text = r.get('content', '')
        if r.get('status') not in {'ok', 'truncated'} or not text:
            reason = 'unavailable_or_failed_turn'
            break
        messages.append({'role': 'assistant', 'content': text})
        # A normal API stop after a bibliography header is only a completion
        # candidate. Reviewers must resolve mid-entry or extra-text ambiguity.
        accumulated = '\n'.join(s.get('content', '') for s in stages)
        if r.get('status') == 'ok' and BIB_HEADING.search(accumulated):
            reason = 'normal_stop_with_bibliography_candidate'
            break
        if turn + 1 == max_calls:
            reason = 'resource_cap_without_confirmed_bibliography_completion'
            break
        messages.append({'role': 'user', 'content': 'Please continue'})
    assistant_text = '\n'.join(r.get('content', '') for r in stages)
    return {'id': case['id'], 'case': case, 'messages': messages, 'stages': stages,
            'assistant_text': assistant_text, 'stop_reason': reason,
            'bibliography_header_offsets': [m.start() for m in BIB_HEADING.finditer(assistant_text)],
            'bibliography_boundary_status': 'needs_research_review',
            'any_truncated_turn': any(r.get('status') == 'truncated' for r in stages),
            'generation_available': all(r.get('status') in {'ok', 'truncated'} and r.get('content') for r in stages)}

def validate_adjudication(row):
    reality = row.get('citation_reality', 'unknown')
    if reality not in {'real', 'fabricated', 'unknown'}:
        raise ValueError('Use real/fabricated/unknown')
    evidence = row.get('evidence', [])
    if reality in {'real', 'fabricated'} and not evidence:
        raise ValueError('A resolved decision requires saved catalogue/publisher evidence')
    if reality == 'fabricated':
        if not row.get('publisher_volume_or_book_toc_check'):
            raise ValueError('Negative search alone is insufficient for fabrication')
        if not row.get('multi_source_negative_search'):
            raise ValueError('Fabrication requires a documented multi-source search')
        if not row.get('second_reviewer'):
            raise ValueError('Fabrication requires independent second review')
    fields = row.get('substantive_error_fields')
    if reality == 'real' and fields is not None:
        if set(fields) != set(SUBSTANTIVE_FIELDS) or any(v not in {True, False, None} for v in fields.values()):
            raise ValueError('Supply all six original substantive fields as true/false/null')
    return row

def summarize_adjudications(rows):
    """All citation occurrences in approved first bibliographies form denominator.

    Missing citation reviews are unknown, not real. Refusals/no bibliography must
    be reported at the conversation level and cannot support zero fabrication.
    """
    for r in rows:
        validate_adjudication(r)
    n = len(rows)
    f = sum(r.get('citation_reality') == 'fabricated' for r in rows)
    real = [r for r in rows if r.get('citation_reality') == 'real']
    u = n - f - len(real)
    err, clean, err_unknown = 0, 0, 0
    for r in real:
        fields = r.get('substantive_error_fields')
        if fields and any(v is True for v in fields.values()):
            err += 1
        elif fields and all(v is False for v in fields.values()):
            clean += 1
        else:
            err_unknown += 1
    return {'citations': n, 'real': len(real), 'fabricated': f, 'unknown_reality': u,
            'fabrication_rate_if_fully_resolved': f / n if n and not u else None,
            'fabrication_rate_bounds': [f / n, (f + u) / n] if n else None,
            'real_with_substantive_errors': err, 'real_with_no_substantive_errors': clean,
            'real_with_unknown_error_status': err_unknown,
            'substantive_error_rate_among_known_real_bounds':
                [err / len(real), (err + err_unknown) / len(real)] if real else None,
            'caveat': 'Error bounds condition on confirmed real citations; unresolved reality can change that denominator.'}
