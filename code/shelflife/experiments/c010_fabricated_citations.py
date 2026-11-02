"""C010 — Fabricated bibliographic citations (Walters & Wilder 2023, Scientific Reports 13:14045).

Claim: asked to write ~2,000-word literature reviews with scholarly references, ChatGPT fabricated 55% of GPT-3.5's
222 and 18% of GPT-4's 414 cited works, and 43% / 24% of the real works carry substantive bibliographic errors
(Tables 3-4). There is no code: the 42 topics (Appendix 1), the 84 complete transcripts (Appendix 2) and the
author-coded 636 citations (Appendix 3 workbook) are the release. They are read here directly from the Office XML
supplements saved by initial-reproduction (``reproductions/C010/sources``; EuropePMC supplementaryFiles).

Protocol (the program)
  A fresh user-only conversation per topic with the literal Appendix-2 prompt; "Please continue" (exact) while the
  answer is incomplete; only the *first* bibliography is evaluated. The paper gives no numeric cap on continuations
  ("sometimes more than once"); every archived conversation used one or two. The project protocol (initial
  reproduction ``pipeline.run_conversation``, reused verbatim by the panel driver) allows at most 3 calls and stops
  after a normally finished response once a bibliography heading has appeared. That stop rule (``BIB_HEADING``,
  loaded from the prior pipeline) is generation logistics, not scoring.

Variants
  paper_chatgpt_ui  default (faithful up to declared API adaptations): literal prompts, max_tokens 8192, <=3 calls.
                    Initial-reproduction pilot / main36 / resume700 runs.
  paper_chatgpt_ui_verify_online  paper_chatgpt_ui with online catalogue lookups (see "online" below).
  cap4096_verify_online           paper_chatgpt_ui_verify_online at max_tokens 4096 (gpt-3.5-turbo output limit).

Scoring (deviation: the paper's scoring was manual)
  extraction  panel ``extract_v2.extract_v2`` (first bibliography; prose dropped, merged entries split).
  reality     panel ``verify_v2.verify``: Crossref (DOI, bibliographic, title+author), Open Library, Semantic
              Scholar; paper's rule real = title match/near-match AND >=1 author surname. Not-found legal / web /
              report / classical / organisation-authored entries and any 'fabricated' call resting on a failed
              lookup are ``unverifiable`` (reported as an upper bound, never as fabricated). Both are loaded from the
              panel task (``_c010_verify``); offline mode reads only the saved response cache.
  online      variant option ``verify_online`` (default False; opt in with the ``*_verify_online`` variant or
              ``Variant.override(..., options={'verify_online': True})``) lets the verifier make new free catalogue
              lookups (Crossref / Open Library / Semantic Scholar; cached under ``code/.cache/c010_verify``, not
              version-pinned). Offline, a lookup absent from the saved cache is never attempted; such a citation is
              ``unverifiable/lookup_failed`` *and* flagged ``lookup_not_attempted``, and ``aggregate`` then reports
              ``fabricated_share`` (and its CI) as None with status ``undetermined_offline_cache_miss`` plus the
              counts and bounds, never a numeric point estimate (a new run scored offline would otherwise read ~0%).
  errors      the verifier's coarse ``exists_with_errors`` flags (date/journal/volume/authorship). NOT the paper's
              six-field substantive-error coding and not validated; reported as ``auto_error_share_among_real``.

Robustness:
  parse       scoring option 'paper' (default: extract_v2 only) or 'robust': extract_v2 first; only when it finds no
              entry in an available conversation, extract_v2 is re-run on cleaned copies of the text (``robust_texts``:
              <think> blocks removed, a duplicated trailing in-text citation "(Author, 2017)" removed from entry lines,
              a few extra bibliography headings such as "Literature Cited" / "Sources" mapped to "References", then
              the same with ``_robust.normalize``). The verifier and the reality rule are unchanged. aggregate reports
              parsed_paper / parsed_robust / unparsed (conversations) and, under 'robust' only,
              headline_reliable = _robust.reliable(conversations, unparsed, missing_generation) with a validity rate.
  robust vs paper on conversations extract_v2 DID parse: none (the fallback runs only when extract_v2 finds no
              entry; tested over the saved conversations).
  robust      paper_chatgpt_ui_verify_online (the panel generation variant) with parse='robust' + reasoning-model cap 32768 through the policy option min_cap_reasoning (reasoning models
              spent the whole 8192 cap on hidden reasoning and returned no text: M196 35/42, M190 5/42 conversations;
              non-reasoning models keep max_tokens 8192), restart at 4096 when a route rejects 8192 (panel-driver
              mechanism), and a stop rule that also recognises the extractor's bibliography headings
              (e.g. "**References:**", which BIB_HEADING misses) so a finished paper is not sent an unneeded
              "Please continue".

Metric (per model, citation-weighted as in Table 3): fabricated share = fabricated / all first-bibliography
entries; upper bound counts unverifiable as fabricated; 95% CI = topic-cluster bootstrap (METHODOLOGY.md).
Conversations without usable generation are missing (reported, never zero-fabrication).
"""
import collections
import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile

from ..paths import INITIAL
from ..program import CallFailed, Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPRO = INITIAL / 'reproductions' / 'C010'
SOURCES = REPRO / 'sources'
MOESM = '41598_2023_41032_MOESM{}_ESM.{}'
PAPER_URL = 'https://www.nature.com/articles/s41598-023-41032-5'
PILOT_TOPICS = (1, 8, 14, 25, 35, 42)
FINAL = REPRO / 'remaining_work/continuation/resume700/final_complete'

PREFIX = ('I want you to act as an academic researcher. Your task is to write a paper of approximately 2000 words '
          'with parenthetical citations and a bibliography that includes at least 5 scholarly resources such as '
          'journal articles and scholarly books. The paper should respond to this question: ')
_APPENDIX2 = Source(repo=str(SOURCES), path=MOESM.format(2, 'docx'), locator='Supplementary Appendix 2, first paragraph after each "GPT-x, TOPIC n" heading',
                    url=PAPER_URL, note='identical introductory text for all 84 conversations; the Methods quote '
                                        "(‘[paper topic].’) differs in punctuation from what was actually sent")
USER = PromptTemplate('user', PREFIX + '"{question}"', source=_APPENDIX2,
                      note='question = the topic question as typed in Appendix 2 (single-spaced; Appendix 1 has double spaces)')
CONTINUE = PromptTemplate('continue', 'Please continue', style='literal', source=Source(
    url=PAPER_URL, locator='Methods, "Paper topics and prompts"; 116 literal turns in Appendix 2'))


# --------------------------------------------------------------------------------------------- Office XML sources
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def docx_paragraphs(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
    return [''.join(t.text or '' for t in p.iter(W + 't')) for p in root.iter(W + 'p')]


def xlsx_sheet(path, index):
    """Rows of worksheet ``index`` (1-based) as {column letter: string}, header row included."""
    with zipfile.ZipFile(path) as z:
        shared = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(S + 'si')]
        root = ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml'))
    rows = []
    for row in root.iter(S + 'row'):
        cells = {}
        for c in row.findall(S + 'c'):
            col = re.sub(r'\d+', '', c.attrib['r'])
            if c.attrib.get('t') == 'inlineStr':
                cells[col] = ''.join(c.find(S + 'is').itertext())
            elif (v := c.find(S + 'v')) is not None:
                cells[col] = shared[int(v.text)] if c.attrib.get('t') == 's' else v.text
        rows.append({'source_row': int(row.attrib['r']), 'cells': cells})
    return rows


def archive_transcripts():
    """The 84 archived conversations of Appendix 2: {(ui_model '3.5'|'4', topic): {'prompt', 'turns', ...}}."""
    paras = docx_paragraphs(SOURCES / MOESM.format(2, 'docx'))
    heads = [(i, m) for i, p in enumerate(paras) if (m := re.fullmatch(r'GPT-(3\.5|4), TOPIC (\d+)', p.strip()))]
    assert len(heads) == 84
    out = {}
    for j, (start, m) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(paras)
        section = [p for p in paras[start + 1:end] if p.strip()]
        turns, current = [{'role': 'user', 'content': section[0]}], []
        for p in section[1:]:
            if p.strip() == 'Please continue':
                turns += [{'role': 'assistant', 'content': '\n'.join(current)}, {'role': 'user', 'content': 'Please continue'}]
                current = []
            else:
                current.append(p)
        if current:
            turns.append({'role': 'assistant', 'content': '\n'.join(current)})
        out[(m[1], int(m[2]))] = {'prompt': section[0], 'turns': turns, 'paragraphs': section,
                                  'assistant_text': '\n'.join(t['content'] for t in turns if t['role'] == 'assistant')}
    return out


def appendix1_topics():
    out = []
    for p in docx_paragraphs(SOURCES / MOESM.format(1, 'docx')):
        m = re.match(r'^(\d+)\.\s*(.*)', p)
        if m and len(out) < 42:
            assert int(m[1]) == len(out) + 1
            out.append(m[2])
    assert len(out) == 42
    return out


def coded_citations():
    """Appendix 3 'Works cited in those papers' (636 rows) with the authors' coding."""
    rows = xlsx_sheet(SOURCES / MOESM.format(3, 'xlsx'), 2)[1:]
    assert len(rows) == 636
    out = []
    for r in rows:
        c = r['cells']
        real = c['N'] == '0'
        out.append({'id': f"historical-gpt{c['A']}-sheetrow{r['source_row']}", 'ui_model': c['A'], 'topic': int(c['C']),
                    'citation': c['E'], 'type': c['I'], 'reality': 'fabricated' if c['N'] == '1' else 'real',
                    # Table 4 counts the six fields P-U; O (publication type) is separate (initial-reproduction report)
                    'substantive_error': any(c.get(k) == '1' for k in 'PQRSTU') if real else None,
                    'error_fields': {k: c.get(k) for k in 'OPQRSTU'}, 'cells': c})
    return out


def coded_papers():
    rows = xlsx_sheet(SOURCES / MOESM.format(3, 'xlsx'), 1)[1:]
    assert len(rows) == 84
    return [r['cells'] for r in rows]


# ------------------------------------------------------------------------------------------- prior project code
def prior_pipeline():
    """initial-reproduction reproductions/C010/pipeline.py: BIB_HEADING (the stop rule)."""
    return load_functions(REPRO / 'pipeline.py', [], constants=('BIB_HEADING',))


def _verifier(version='v2', offline=True):
    from . import _c010_verify
    return _c010_verify.pipeline(version, offline)


# ----------------------------------------------------------------------------------------------------- program
class _EitherHeading:
    """Stop-rule matcher for stop_rule='robust': the prior BIB_HEADING or the extractor's HEAD regex."""

    def __init__(self, *patterns):
        self.patterns = patterns

    def search(self, text):
        return next((m for m in (p.search(text) for p in self.patterns) if m), None)


ROBUST_MIN_CAP_REASONING = 32768   # 'robust': reasoning-model output cap (policy min_cap_reasoning)


class ContinueUntilBibliography(Program):
    """initial-reproduction ``pipeline.run_conversation`` as a program (panel ``generate.run_conversation`` is
    the same function). Steps are 'turn-0', 'turn-1', ...; with ``cap_fallback`` (panel driver) a first turn
    rejected for its output cap restarts the conversation at that cap with steps 'cap4096-turn-k'."""
    name = 'continue_until_bibliography'
    CAP_REJECTED = re.compile(r'max_tokens|max_completion_tokens|maximum.*tokens|too large', re.I)

    def __init__(self, first_messages, params, max_calls=3, cont='Please continue', cap_fallback=None,
                 stop_rule='prior'):
        if not 1 <= max_calls <= 3:
            raise ValueError('the project protocol permits 1-3 calls')
        if stop_rule not in ('prior', 'robust'):
            raise ValueError(f'unknown stop_rule {stop_rule!r}')
        self.first_messages, self.params, self.max_calls = first_messages, dict(params), max_calls
        self.cont, self.cap_fallback = cont, cap_fallback
        self.bib_heading = prior_pipeline()['BIB_HEADING']
        if stop_rule == 'robust':   # also the extractor's heading regex ("**References:**", "## 7. References", ...)
            self.bib_heading = _EitherHeading(self.bib_heading, _verifier('v1').extract.__globals__['HEAD'])

    def preview(self, item):
        return self.first_messages(item)

    def _conversation(self, item, lm, params, prefix):
        messages = self.first_messages(item)
        stages, reason = [], None
        for turn in range(self.max_calls):
            c = lm(messages, params, step=f'{prefix}turn-{turn}')
            stages.append(c)
            if c.status not in {'ok', 'truncated'} or not c.content:
                reason = 'unavailable_or_failed_turn'
                break
            messages = messages + [{'role': 'assistant', 'content': c.content}]
            if c.status == 'ok' and self.bib_heading.search('\n'.join(s.content for s in stages)):
                reason = 'normal_stop_with_bibliography_candidate'
                break
            if turn + 1 == self.max_calls:
                reason = 'resource_cap_without_confirmed_bibliography_completion'
                break
            messages = messages + [{'role': 'user', 'content': self.cont}]
        return {'assistant_text': '\n'.join(s.content for s in stages), 'stop_reason': reason, 'messages': messages,
                'stage_status': [s.status for s in stages], 'stage_finish': [s.finish_reason for s in stages],
                'any_truncated_turn': any(s.status == 'truncated' for s in stages),
                'generation_available': all(s.status in {'ok', 'truncated'} and s.content for s in stages)}

    def forward(self, item, lm):
        try:
            out = self._conversation(item, lm, self.params, '')
            out['cap_reduced'] = None
        except CallFailed as e:
            first = len(lm.trace) == 1
            if not (self.cap_fallback and first and self.CAP_REJECTED.search(e.completion.error or '')):
                raise
            out = self._conversation(item, lm, {**self.params, 'max_tokens': self.cap_fallback}, f'cap{self.cap_fallback}-')
            out['cap_reduced'] = self.cap_fallback
        return out


# ------------------------------------------------------------------------------------- robust parse fallback
_THINK_BLOCK = re.compile(r'<(think|thinking|reasoning)>.*?</\1>|<(?:think|thinking|reasoning)>.*\Z', re.S | re.I)
_YEAR = r'(?:1[5-9]\d\d|20[0-3]\d)'
# an entry line (author-date "(2017)" or ". 2017." early in the line) that ends with a duplicated in-text citation
# "(Boudreaux et al., 2017)": v1's INTEXT_END test then rejects it as body prose (M066 topic 29, the panel)
_TRAILING_INTEXT = re.compile(
    r'^(?P<entry>[^\n]{0,300}?(?:\(' + _YEAR + r'[a-z]?\)|\.\s' + _YEAR + r'[a-z]?\.)[^\n]{10,}?)'
    r'[ \t]*\([A-Z][^()\n]{1,80}?,?\s' + _YEAR + r'[a-z]?\)\.?[ \t]*$', re.M)
_ALT_HEADING = re.compile(
    r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*(?:(?:\d+|[IVXLC]+)\.?\s*)?'
    r'(?:literature cited|cited literature|cited works|works consulted|sources|scholarly sources|citations|reference|'
    r'bibliographic references|selected references|reference list and further reading)'
    r'[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$')


def robust_texts(text):
    """Cleaned copies of a response for the robust fallback, in the order tried. Only reached when extract_v2
    finds no entry in the original text."""
    t = _THINK_BLOCK.sub(' ', text or '')
    t = _ALT_HEADING.sub('References', t)
    t = _TRAILING_INTEXT.sub(lambda m: m.group('entry'), t)
    n = _robust.normalize(text)
    n = _ALT_HEADING.sub('References', n)
    return [t, _TRAILING_INTEXT.sub(lambda m: m.group('entry'), n)]


def extract_bibliography(text, version='v2', parse='paper'):
    """(extraction dict, path): the paper extractor first; under parse='robust' the cleaned copies only when it
    finds no entry. path is 'paper', 'robust' or None (no entry found)."""
    ext = _verifier(version).extract
    e = ext(text or '')
    if e['entries'] or parse != 'robust':
        return e, ('paper' if e['entries'] else None)
    for t in robust_texts(text):
        r = ext(t)
        if r['entries']:
            return r, 'robust'
    return e, None


# ------------------------------------------------------------------------------------------------- experiment
def _sha16(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')
_ABORTED_WAVE1 = ('wave-1 run aborted 2026-10-04 (OpenRouter credit exhausted, local connect timeouts) and moved '
                  'aside; created before the relaunch core edits and superseded by the tagged reruns; the collector '
                  'skips _aborted*')
_SMOKE = ('pre-launch route/policy smoke test (runs/smoke*, about $0.02) made before the M204 capabilities fix and '
          'policy changes, so its requests no longer replay; not used by the analysis')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


@register
class C010FabricatedCitations(Experiment):
    claim_id = 'C010'
    title = 'Fabricated and erroneous citations in LLM-written literature reviews'
    claim = ("ChatGPT literature reviews fabricate many cited works (GPT-3.5 55% of 222, GPT-4 18% of 414) and real "
             "works carry substantive bibliographic errors (43% / 24%).")
    paper = {'title': 'Fabrication and errors in the bibliographic citations generated by ChatGPT',
             'authors': 'Walters & Wilder', 'year': 2023, 'url': PAPER_URL}
    source = _APPENDIX2
    default_variant = 'paper_chatgpt_ui'

    def build_variants(self):
        base = Variant(
            'paper_chatgpt_ui', 'Literal Appendix-2 prompt, fresh conversation, exact "Please continue", <=3 calls, '
                                'first bibliography scored', prompts={'user': USER, 'continue': CONTINUE},
            params={'max_tokens': 8192}, strategy='continue_until_bibliography',
            options={'max_calls': 3, 'cap_fallback': None, 'extractor': 'v2', 'verifier': 'v2', 'verify_online': False},
            deviations=(
                'original: ChatGPT web UI (GPT-3.5 / GPT-4), first week of April 2023; snapshot, hidden system text, '
                'sampling and UI length limit unknown. API call with no system message, provider-default sampling',
                'max_tokens 8192 is an API resource cap, not the UI response limit (initial-reproduction protocol.md)',
                'at most 3 calls (initial + 2 "Please continue"); the paper gives no number, archive max is 2 continuations',
                'stop rule = prior pipeline BIB_HEADING after a normal finish; a model whose heading the regex misses '
                '(e.g. "**References:**") receives continuations it did not need; only the first bibliography is scored',
                'scoring is automated (panel extract_v2 + verify_v2 catalogue lookups) instead of the authors\' manual '
                'searches; substantive errors are coarse automated flags, not the paper\'s six-field coding'))
        online = base.override(
            'paper_chatgpt_ui_verify_online', options={'verify_online': True},
            description='paper_chatgpt_ui scored with new (network) catalogue lookups for uncached citations')
        return {
            'paper_chatgpt_ui': base,
            # explicit opt-in to network-backed scoring for new generations (same generation protocol)
            'paper_chatgpt_ui_verify_online': online,
            # reproductions (G03): GPT-3.5 API snapshots reject max_tokens 8192 (output limit 4096);
            # the April-2023 ChatGPT-3.5 UI itself had a 4k-token window, so 4096 is the closest served cap.
            'cap4096_verify_online': base.override(
                'cap4096_verify_online', params={'max_tokens': 4096}, options={'verify_online': True},
                description='paper_chatgpt_ui at max_tokens 4096 (GPT-3.5-turbo output limit), online catalogue lookups',
                deviations=('max_tokens 4096 instead of 8192: gpt-3.5-turbo snapshots cap output at 4096 tokens',)),
            # robustness for large panels: same requests as the panel generation variant
            'robust': online.override(
                'robust',
                options={'parse': 'robust', 'cap_fallback': 4096, 'stop_rule': 'robust',
                         'min_cap_reasoning': ROBUST_MIN_CAP_REASONING},
                description='robust_parse + reasoning-model cap 32768 (policy min_cap_reasoning) + restart at 4096 if '
                            'a route rejects 8192 + robust stop rule, for new panel runs',
                deviations=('scoring: robust bibliography fallback and 10% reliability rule (as robust_parse)',
                            f'reasoning models: output cap {ROBUST_MIN_CAP_REASONING} (panel policy min_cap_reasoning, '
                            'case_extra) instead of 8192: they spent the whole 8192 cap on hidden reasoning and '
                            'returned no text (panel: M196 35/42, M190 5/42 conversations). Non-reasoning models '
                            'keep the paper max_tokens 8192',
                            'a first turn rejected for its output cap (e.g. a 4k-output route) restarts the '
                            'conversation at 4096 (steps cap4096-turn-k), as the panel driver does',
                            'stop rule also accepts the extractor\'s bibliography headings (e.g. "**References:**"), '
                            'so a finished paper is not sent an unneeded "Please continue"')),
        }

    # ---------------------------------------------------------------------------------------------- data
    def items(self, variant, limit=None):
        archive = archive_transcripts()
        areas = appendix1_topics()
        rows = []
        for n in range(1, 43):
            if limit is not None and len(rows) >= limit:
                break
            prompt = archive[('3.5', n)]['prompt']
            assert prompt == archive[('4', n)]['prompt'] and prompt.startswith(PREFIX + '"') and prompt.endswith('"')
            rows.append({'id': f'C010-topic-{n:02d}', 'topic_number': n, 'question': prompt[len(PREFIX) + 1:-1],
                         'appendix1_topic': areas[n - 1],
                         'academic_area': 'H' if n <= 8 else 'S' if n <= 33 else 'N', 'pilot': n in PILOT_TOPICS})
        return rows

    def program(self, variant):
        user = variant.prompts['user']
        prog = ContinueUntilBibliography(
            lambda item: [{'role': 'user', 'content': user.render(question=item['question'])}], variant.params,
            max_calls=variant.options['max_calls'], cont=variant.prompts['continue'].render(),
            cap_fallback=variant.options['cap_fallback'], stop_rule=variant.options.get('stop_rule', 'prior'))
        cap = variant.options.get('min_cap_reasoning')
        if cap:   # robust only: the panel policy raises reasoning models' cap to this (case_extra on every turn)
            prog.case_extra = {'min_cap_reasoning': cap}
        return prog

    # ------------------------------------------------------------------------------------------- scoring
    def verify_citations(self, citations, version='v2', online=False):
        """Verify citation strings. ``online=False`` (default) reads only saved catalogue responses; a verdict
        that needed an uncached lookup and is not 'real' is flagged ``lookup_not_attempted`` (a found match is
        final, so a 'real' verdict cannot be changed by a missing lookup)."""
        from ._c010_verify import verify_tracked
        p = _verifier(version, offline=not online)
        out = []
        for c in citations:
            v = verify_tracked(p, c)
            out.append({'citation': c, 'sha': _sha16(c), 'decision': v['decision'], 'reality': v['reality'],
                        'type_guess': v.get('type_guess'), 'auto_error': v['decision'] == 'exists_with_errors',
                        'lookup_failed': v.get('unverifiable_reason') == 'lookup_failed',
                        'offline_cache_misses': v['offline_cache_misses'],
                        'lookup_not_attempted': bool(v['offline_cache_misses']) and v['reality'] != 'real'})
        return out

    def score(self, item, output, variant):
        """Extract the first bibliography and verify each entry."""
        extractor = variant.options['extractor']
        verifier = variant.options['verifier']
        parse = variant.options.get('parse', 'paper')
        available = bool(output.get('generation_available'))
        base = {'generation_available': available, 'stop_reason': output.get('stop_reason'),
                'cap_reduced': output.get('cap_reduced')}
        ext, path = extract_bibliography(output.get('assistant_text') or '', extractor, parse)
        if parse == 'robust':
            base['parse_path'] = path if available else None
        if not available:
            # e.g. a bibliography the BIB_HEADING stop rule missed, then a failed unneeded continuation (audit
            # minor 3): still missing, but counted so the cost of unneeded turns is visible
            return {**base, 'has_bibliography': None, 'citations': [],
                    'unavailable_with_extractable_bibliography': bool(ext['entries'])}
        return {**base, 'has_bibliography': ext['has_bibliography'],
                'citations': self.verify_citations(ext['entries'], verifier, online=bool(variant.options.get('verify_online'))),
                'verify_online': bool(variant.options.get('verify_online'))}

    def aggregate(self, rows, variant=None, n_boot=2000):
        """rows: [{'item', 'score'}]; score['citations'] entries carry 'reality' (real|fabricated|unverifiable),
        optional 'substantive_error' (author coding) / 'auto_error' (verifier flags) and a type."""
        scored = [r for r in rows if (r.get('score') or {}).get('generation_available') or (r.get('score') or {}).get('citations')]
        cits = [dict(c, _topic=r['item']['id']) for r in scored for c in r['score']['citations']]
        n = len(cits)
        count = collections.Counter(c['reality'] for c in cits)
        F, U, R = count['fabricated'], count['unverifiable'], count['real']
        share = lambda cs: sum(c['reality'] == 'fabricated' for c in cs) / len(cs) if cs else float('nan')
        not_attempted = sum(bool(c.get('lookup_not_attempted')) for c in cits)
        if not_attempted:   # offline cache misses: the point estimate would count never-checked works as not fabricated
            lo = hi = None
            status = 'undetermined_offline_cache_miss'
        else:
            _, lo, hi = stats.cluster_bootstrap(cits, lambda c: c['_topic'], share, n_boot=n_boot, seed=20260930)
            status = 'determined' if n else 'no_citations'
        real = [c for c in cits if c['reality'] == 'real']
        out = {
            'conversations': len(rows),
            'generation_available': sum(bool((r.get('score') or {}).get('generation_available')) for r in rows),
            'missing_generation': sum(not (r.get('score') or {}).get('generation_available') for r in rows),
            'missing_with_extractable_bibliography': sum(bool((r.get('score') or {}).get('unavailable_with_extractable_bibliography')) for r in rows),
            'topics_with_bibliography': sum(bool((r.get('score') or {}).get('has_bibliography')) for r in rows),
            'citations': n, 'fabricated': F, 'unverifiable': U, 'real': R,
            'lookup_failed': sum(bool(c.get('lookup_failed')) for c in cits),
            'lookup_not_attempted_offline': not_attempted,
            'fabricated_share_status': status,
            'fabricated_share': F / n if n and not not_attempted else None,
            'fabricated_share_ci95_topic_bootstrap': [lo, hi],
            'fabricated_share_bounds_unverifiable': [F / n, (F + U) / n] if n else None,
            'by_type': {t: {'n': len(cs), 'fabricated': sum(c['reality'] == 'fabricated' for c in cs),
                            'fabricated_share': None if any(c.get('lookup_not_attempted') for c in cs) else share(cs)}
                        for t, cs in sorted(_group(cits, lambda c: c.get('type') or c.get('type_guess')).items())},
        }
        if any('substantive_error' in c for c in real):
            known = [c for c in real if c.get('substantive_error') is not None]
            k = sum(c['substantive_error'] for c in known)
            out.update(substantive_error_share_among_real=k / len(known) if known else None,
                       real_with_substantive_error=k, real_with_unknown_error_status=len(real) - len(known))
        if variant is not None and variant.options.get('parse') == 'robust':
            # conversations: parsed by the paper extractor, by the robust fallback, or available with no entry;
            # missing = no usable generation (call failed, or a turn returned nothing, e.g. cap spent on reasoning)
            paths = collections.Counter((r.get('score') or {}).get('parse_path') for r in rows
                                        if (r.get('score') or {}).get('generation_available'))
            unparsed, missing = paths[None], out['missing_generation']
            out.update(parsed_paper=paths['paper'], parsed_robust=paths['robust'], unparsed=unparsed,
                       validity_rate=(paths['paper'] + paths['robust']) / len(rows) if rows else None,
                       headline_reliable=_robust.reliable(len(rows), unparsed, missing)
                       and out['fabricated_share'] is not None,
                       reliability_rule='robust: <=10% of conversations unparsed or missing generation')
        if any('auto_error' in c for c in real):
            out.update(real_with_auto_errors=sum(bool(c.get('auto_error')) for c in real),
                       auto_error_share_among_real=sum(bool(c.get('auto_error')) for c in real) / len(real) if real else None,
                       auto_error_note='coarse automated flags (date/journal/volume/authorship), not the paper\'s coding')
        return out

    # ------------------------------------------------------------------------------ archived (paper) outputs
    def archived_rows(self, ui_model, scorer='author_coding'):
        """Rows for the 42 archived papers of one ChatGPT UI model ('3.5' or '4').

        author_coding  the authors' own labels (Appendix 3), the paper's numbers.
        auto_v2       our automated verifier applied to the authors' 636 citation strings (offline cache).
        """
        coded = [c for c in coded_citations() if c['ui_model'] == ui_model]
        by_topic = _group(coded, lambda c: c['topic'])
        rows = []
        for t in range(1, 43):
            cs = by_topic.get(t, [])
            if scorer == 'author_coding':
                cits = [{'citation': c['citation'], 'reality': c['reality'], 'substantive_error': c['substantive_error'],
                         'type': c['type']} for c in cs]
            else:
                cits = self.verify_citations([c['citation'] for c in cs], scorer.split('_')[1])
                for v, c in zip(cits, cs):
                    v['type'] = c['type']
            rows.append({'item': {'id': f'C010-topic-{t:02d}'}, 'score': {'generation_available': True,
                                                                          'has_bibliography': bool(cits), 'citations': cits}})
        return rows

    def archived_metric(self, ui_model, key, scorer='author_coding', type_=None):
        agg = self.aggregate(self.archived_rows(ui_model, scorer), n_boot=1)
        return agg['by_type'][type_]['fabricated_share'] if type_ else agg[key]

    # ------------------------------------------------------------------------------------- targets
    targets = tuple(
        PaperTarget(name, value, 0.005, where, f'ChatGPT {m} web UI (April 2023)', 'paper_chatgpt_ui', 'archived_outputs',
                    compute=fn, note='authors\' own coding of the archived outputs (Appendix 3); tolerance = rounding to whole %')
        for name, value, where, m, fn in (
            ('gpt35_fabricated_share', 0.55, 'Table 3, all works (n=222)', 'GPT-3.5', lambda e: e.archived_metric('3.5', 'fabricated_share')),
            ('gpt4_fabricated_share', 0.18, 'Table 3, all works (n=414)', 'GPT-4', lambda e: e.archived_metric('4', 'fabricated_share')),
            ('gpt35_articles_fabricated', 0.73, 'Table 3, articles (n=128)', 'GPT-3.5', lambda e: e.archived_metric('3.5', None, type_='A')),
            ('gpt4_articles_fabricated', 0.18, 'Table 3, articles (n=255)', 'GPT-4', lambda e: e.archived_metric('4', None, type_='A')),
            ('gpt35_books_fabricated', 0.23, 'Table 3, books (n=78)', 'GPT-3.5', lambda e: e.archived_metric('3.5', None, type_='B')),
            ('gpt4_books_fabricated', 0.08, 'Table 3, books (n=126)', 'GPT-4', lambda e: e.archived_metric('4', None, type_='B')),
            ('gpt35_chapters_fabricated', 0.70, 'Table 3, chapters (n=10)', 'GPT-3.5', lambda e: e.archived_metric('3.5', None, type_='C')),
            ('gpt4_chapters_fabricated', 0.70, 'Table 3, chapters (n=23)', 'GPT-4', lambda e: e.archived_metric('4', None, type_='C')),
            ('gpt35_websites_fabricated', 0.50, 'Table 3, websites (n=6)', 'GPT-3.5', lambda e: e.archived_metric('3.5', None, type_='W')),
            ('gpt4_websites_fabricated', 0.10, 'Table 3, websites (n=10)', 'GPT-4', lambda e: e.archived_metric('4', None, type_='W')),
            ('gpt35_substantive_error_share', 0.43, 'Table 4 (n=101 real)', 'GPT-3.5', lambda e: e.archived_metric('3.5', 'substantive_error_share_among_real')),
            ('gpt4_substantive_error_share', 0.24, 'Table 4 (n=340 real)', 'GPT-4', lambda e: e.archived_metric('4', 'substantive_error_share_among_real')),
        )) + (
        PaperTarget('gpt35_fabricated_share_auto_verifier', 0.55, 0.005, 'Table 3, all works (n=222)', 'ChatGPT GPT-3.5 web UI',
                    'paper_chatgpt_ui', 'archived_outputs',
                    compute=lambda e: e.archived_metric('3.5', 'fabricated_share', scorer='auto_v2'),
                    note='our automated verifier (panel verify_v2, offline cache) on the authors\' 222 citation strings',
                    known_discrepancy=('automated verifier gives 117/222 = 52.7% vs the authors\' 121/222 (54.5%, '
                                       'reported 55%): 112 agree, 5 author-fabricated found real, 4 unverifiable, 5 '
                                       'author-real called fabricated (sensitivity 0.93, PPV 0.96); upper bound with 10 '
                                       'unverifiable 57.2%. Rules were tuned on this set (in-sample).')),
        PaperTarget('gpt4_fabricated_share_auto_verifier', 0.18, 0.005, 'Table 3, all works (n=414)', 'ChatGPT GPT-4 web UI',
                    'paper_chatgpt_ui', 'archived_outputs',
                    compute=lambda e: e.archived_metric('4', 'fabricated_share', scorer='auto_v2'),
                    note='our automated verifier (panel verify_v2, offline cache) on the authors\' 414 citation strings',
                    known_discrepancy=('automated verifier gives 72/414 = 17.4% vs the authors\' 74/414 (17.9%, '
                                       'reported 18%), 0.6 pp beyond rounding: 61 agree, 9 author-fabricated found '
                                       'real, 4 unverifiable, 11 author-real called fabricated (sensitivity 0.82, PPV '
                                       '0.85); upper bound with 12 unverifiable 20.3%. In-sample tuning.')),
    )


def _group(rows, key):
    out = collections.defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out
