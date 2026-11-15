"""Helpers internes partagés par les transformations de features."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def closed_as_of(as_of: datetime | pd.Timestamp | None) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.now(tz='UTC')
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('as_of doit inclure un fuseau horaire')
    return cutoff.tz_convert('UTC')
