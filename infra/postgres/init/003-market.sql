-- Bougies récentes (base mlops, schéma market) : fenêtre glissante alimentée par l'API REST Binance.
-- L'historique complet est en Parquet (data/candles/...). Colonnes alignées sur marketdata.schema
-- (vérifié par tests/test_store.py). api_writer met à jour (upsert) et purge au-delà de la rétention ;
-- grafana_reader lit.
\set ON_ERROR_STOP on

CREATE SCHEMA market;
REVOKE ALL ON SCHEMA market FROM PUBLIC;
GRANT USAGE ON SCHEMA market TO api_writer, grafana_reader;

CREATE TABLE market.candles (
    symbol                 text             NOT NULL,
    open_time              timestamptz      NOT NULL,
    open                   double precision NOT NULL CHECK (open > 0),
    high                   double precision NOT NULL CHECK (high > 0),
    low                    double precision NOT NULL CHECK (low > 0),
    close                  double precision NOT NULL CHECK (close > 0),
    volume                 double precision NOT NULL CHECK (volume >= 0),
    close_time             timestamptz      NOT NULL,
    quote_volume           double precision NOT NULL CHECK (quote_volume >= 0),
    trades                 bigint           NOT NULL CHECK (trades >= 0),
    taker_buy_base_volume  double precision NOT NULL CHECK (taker_buy_base_volume >= 0),
    taker_buy_quote_volume double precision NOT NULL CHECK (taker_buy_quote_volume >= 0),
    ingested_at            timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, open_time),
    CHECK (close_time > open_time)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON market.candles TO api_writer;
GRANT SELECT ON market.candles TO grafana_reader;
