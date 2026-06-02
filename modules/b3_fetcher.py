"""
Busca lista de empresas da B3 via brapi.dev.
"""

import os
import requests
import pandas as pd
import streamlit as st
from typing import Optional


BRAPI_BASE = "https://brapi.dev/api"
_TOKEN = os.getenv("BRAPI_TOKEN", "")


def _headers() -> dict:
    return {"Authorization": f"Bearer {_TOKEN}"} if _TOKEN else {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_b3_stocks(search: str = "") -> pd.DataFrame:
    """
    Retorna DataFrame com todas as ações da B3 disponíveis no brapi.dev.
    Colunas: stock, name, close, change, volume, market_cap, sector, type
    """
    params: dict = {"limit": 500, "sortBy": "market_cap_basic", "sortOrder": "desc", "type": "stock"}
    if search:
        params["search"] = search
    if _TOKEN:
        params["token"] = _TOKEN

    try:
        resp = requests.get(f"{BRAPI_BASE}/quote/list", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return pd.DataFrame()

    stocks = data.get("stocks", [])
    if not stocks:
        return pd.DataFrame()

    df = pd.DataFrame(stocks)

    rename = {
        "stock": "ticker",
        "name": "nome",
        "close": "preco",
        "change": "variacao_pct",
        "change_abs": "variacao_abs",
        "volume": "volume",
        "market_cap_basic": "market_cap",
        "sector": "setor",
        "type": "tipo",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Garante colunas essenciais
    for col in ["ticker", "nome", "preco", "variacao_pct", "volume", "market_cap", "setor"]:
        if col not in df.columns:
            df[col] = None

    # Limpa nomes vazios
    df["nome"] = df["nome"].fillna(df["ticker"])
    df["setor"] = df["setor"].fillna("Outros")

    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_quote_batch(tickers: list[str]) -> dict:
    """Busca cotação em lote para até 20 tickers."""
    if not tickers:
        return {}
    ticker_str = ",".join(tickers[:20])
    params = {}
    if _TOKEN:
        params["token"] = _TOKEN
    try:
        resp = requests.get(f"{BRAPI_BASE}/quote/{ticker_str}", params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return {r["symbol"]: r for r in results if "symbol" in r}
    except Exception:
        return {}
