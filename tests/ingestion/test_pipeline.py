"""Traitement des lots : client Binance simulé, Parquet temporaire, table et traces simulées."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ingestion import pipeline
from ingestion.download import DownloadError
from ingestion.pipeline import BatchProcessor
from ingestion.store import ParquetStore
from marketdata import identity, sources
from marketdata.sources import SourceFormatError
from tests.conftest import HOUR_MS, START_MS, raw_rows

JAN = datetime(2025, 1, 1, tzinfo=UTC)
FEB = datetime(2025, 2, 1, tzinfo=UTC)
SHA = 'a' * 64


class FakeClient:
    def __init__(self, archive=None, recent=None, error=None):
        self.archive = archive
        self.recent = recent or []
        self.error = error
        self.calls = []

    def archive_rows(self, symbol, month):
        self.calls.append(('archive', symbol, month))
        if self.error:
            raise self.error
        return None if self.archive is None else (self.archive, SHA)

    def recent_rows(self, symbol, start, end):
        self.calls.append(('recent', symbol, start, end))
        if self.error:
            raise self.error
        return self.recent


class FakeTable:
    def __init__(self, latest=None):
        self.latest = latest
        self.upserted = []

    def upsert(self, frame):
        self.upserted.append(frame)
        return len(frame)

    def latest_open_time(self, symbol):
        return self.latest


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return len(self.calls)


def make(tmp_path, client, table=None):
    recorder = Recorder()
    processor = BatchProcessor(
        client, record=recorder, parquet=ParquetStore(tmp_path), table=table or FakeTable(), flow_run_id='f1'
    )
    return processor, recorder


# -- archives ----------------------------------------------------------------------------------
def test_accepted_month_written_and_traced(tmp_path):
    processor, recorder = make(tmp_path, FakeClient(archive=raw_rows(31 * 24)))
    result = processor.archive_month('BTCUSDT', JAN)
    assert result.accepted and result.rows == 744
    assert processor.parquet.has_month('BTCUSDT', JAN)
    trace = recorder.calls[0]
    assert trace['status'] == 'accepted' and trace['source'] == 'archive'
    assert (trace['period_start'], trace['period_end']) == (JAN, FEB)
    assert trace['source_sha256'] == SHA and trace['prefect_flow_run_id'] == 'f1'
    assert trace['data_sha256'] == identity.fingerprint(processor.parquet.read())
    assert trace['issues'] == []


def test_signalled_issues_do_not_block(tmp_path):
    rows = raw_rows(31 * 24)
    del rows[5]
    processor, recorder = make(tmp_path, FakeClient(archive=rows))
    result = processor.archive_month('BTCUSDT', JAN)
    assert result.accepted
    assert recorder.calls[0]['issues'][0]['rule'] == 'bougies manquantes'
    assert recorder.calls[0]['issues'][0]['blocking'] is False


def test_interrupted_candle_dropped_and_signalled(tmp_path):
    # arrêt du trading en cours d'heure (cas Binance du 2023-03-24) : clôture à +39 min
    rows = raw_rows(31 * 24)
    rows[5][6] = str(int(rows[5][0]) + 39 * 60_000)
    processor, recorder = make(tmp_path, FakeClient(archive=rows))
    result = processor.archive_month('BTCUSDT', JAN)
    assert result.accepted and result.rows == 743
    issues = recorder.calls[0]['issues']
    assert issues[0]['rule'] == 'bougie interrompue écartée' and issues[0]['blocking'] is False
    assert issues[0]['count'] == 1
    assert len(processor.parquet.read()) == 743


def test_blocking_issue_rejects_without_writing(tmp_path):
    rows = raw_rows(31 * 24)
    rows.append(rows[0])
    processor, recorder = make(tmp_path, FakeClient(archive=rows))
    result = processor.archive_month('BTCUSDT', JAN)
    assert result.status == 'rejected'
    assert not processor.parquet.has_month('BTCUSDT', JAN)
    trace = recorder.calls[0]
    assert trace['status'] == 'rejected' and trace['data_sha256'] is None
    assert any(i['blocking'] for i in trace['issues'])


def test_candles_outside_month_rejected(tmp_path):
    processor, _ = make(tmp_path, FakeClient(archive=raw_rows(2, start_ms=START_MS - HOUR_MS)))
    result = processor.archive_month('BTCUSDT', JAN)
    assert result.status == 'rejected'
    assert any(i.rule == 'bougies hors du mois' for i in result.issues)


def test_unpublished_month_not_traced(tmp_path):
    processor, recorder = make(tmp_path, FakeClient(archive=None))
    assert processor.archive_month('BTCUSDT', JAN).status == 'unavailable'
    assert recorder.calls == []


@pytest.mark.parametrize(
    ('error', 'rule'),
    [(DownloadError('HTTP 418'), 'téléchargement'), (SourceFormatError('SHA-256 faux'), 'format source')],
)
def test_fetch_failure_traced_as_rejected(tmp_path, error, rule):
    processor, recorder = make(tmp_path, FakeClient(error=error))
    result = processor.archive_month('BTCUSDT', JAN)
    assert result.status == 'rejected' and result.rows == 0
    assert recorder.calls[0]['issues'][0]['rule'] == rule


# -- récentes ----------------------------------------------------------------------------------
def test_recent_keeps_closed_candles_only(tmp_path):
    table = FakeTable()
    processor, recorder = make(tmp_path, FakeClient(recent=raw_rows(3)), table)
    now = JAN + timedelta(hours=2, minutes=30)  # la 3e bougie (02:00-03:00) n'est pas clôturée
    result = processor.recent('BTCUSDT', JAN, now)
    assert result.accepted and result.rows == 2
    assert len(table.upserted[0]) == 2
    assert recorder.calls[0]['source'] == 'rest' and len(recorder.calls[0]['source_sha256']) == 64


def test_recent_nothing_new(tmp_path):
    table = FakeTable()
    processor, recorder = make(tmp_path, FakeClient(recent=[]), table)
    result = processor.recent('BTCUSDT', JAN, JAN + timedelta(minutes=10))
    assert result.accepted and result.rows == 0
    assert table.upserted == [] and recorder.calls[0]['data_sha256'] is None


def test_recent_rejected_not_upserted(tmp_path):
    rows = raw_rows(3)
    rows[1][2] = '1.0'  # high < low
    table = FakeTable()
    processor, _ = make(tmp_path, FakeClient(recent=rows), table)
    assert processor.recent('BTCUSDT', JAN, JAN + timedelta(hours=4)).status == 'rejected'
    assert table.upserted == []


# -- planification des lots --------------------------------------------------------------------
def test_months_between():
    months = pipeline.months_between(datetime(2024, 11, 15, tzinfo=UTC), datetime(2025, 2, 10, tzinfo=UTC))
    assert months == [datetime(2024, 12, 1, tzinfo=UTC), JAN]


def test_missing_months_skips_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'HISTORY_START', datetime(2024, 11, 1, tzinfo=UTC))
    parquet = ParquetStore(tmp_path)
    parquet.write_month(sources.normalize(raw_rows(2), 'BTCUSDT'), 'BTCUSDT', JAN)
    now = datetime(2025, 2, 20, tzinfo=UTC)
    assert pipeline.missing_months(parquet, 'BTCUSDT', now) == [
        datetime(2024, 11, 1, tzinfo=UTC),
        datetime(2024, 12, 1, tzinfo=UTC),
    ]


@pytest.mark.parametrize(
    ('latest', 'expected'),
    [
        (None, datetime(2025, 1, 6, 12, tzinfo=UTC)),  # table vide : début de la rétention
        (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 6, 12, tzinfo=UTC)),  # trop ancien
        (datetime(2025, 1, 10, 5, tzinfo=UTC), datetime(2025, 1, 10, 6, tzinfo=UTC)),  # reprise
    ],
)
def test_recent_start(latest, expected):
    now = datetime(2025, 1, 11, 12, 34, tzinfo=UTC)
    assert pipeline.recent_start(FakeTable(latest), 'BTCUSDT', now, timedelta(days=5)) == expected
