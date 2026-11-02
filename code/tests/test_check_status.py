"""Static and technical success must not imply scientific readiness."""
from shelflife import check


def test_no_tests_is_explicitly_not_a_readiness_verdict(monkeypatch, capsys):
    monkeypatch.setattr(check, 'static_checks', lambda cid: ([('required', 'fixture', True, '')], None))
    assert check.main('FIXTURE', run_pytest=False) == 0
    out = capsys.readouterr().out
    assert 'STATIC CHECKS PASS' in out and 'tests not run' in out
    assert '\nREADY' not in out


def test_technical_checks_do_not_assert_scientific_validity(monkeypatch, capsys):
    class Fixture:
        maturity = 'ready'
    monkeypatch.setattr(check, 'static_checks', lambda cid: ([('required', 'fixture', True, '')], Fixture()))
    monkeypatch.setattr(check, 'run_tests', lambda cid: (0, '1 passed', []))
    assert check.main('FIXTURE') == 0
    out = capsys.readouterr().out
    assert 'TECHNICAL CHECKS PASS' in out and 'requires independent review' in out
    assert '\nREADY' not in out


def test_failed_tests_block_even_when_static_checks_pass(monkeypatch, capsys):
    class Fixture:
        maturity = 'ready'
    monkeypatch.setattr(check, 'static_checks', lambda cid: ([('required', 'fixture', True, '')], Fixture()))
    monkeypatch.setattr(check, 'run_tests', lambda cid: (1, '1 failed', ['FAILED fixture']))
    assert check.main('FIXTURE') == 1
    assert 'NOT READY' in capsys.readouterr().out
