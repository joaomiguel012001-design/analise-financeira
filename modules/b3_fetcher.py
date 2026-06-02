"""
Busca lista de empresas da B3 via brapi.dev.
"""

import os
import re
import requests
import pandas as pd
import streamlit as st


BRAPI_BASE = "https://brapi.dev/api"
_TOKEN = os.getenv("BRAPI_TOKEN", "")

# Tickers válidos: 4 letras + 1-2 dígitos (ex: PETR3, VALE3, ITUB4, BBAS3, SANB11)
# Exclui fracionados (F no final), direitos (PETR3B, etc) e outros sufixos
_TICKER_RE = re.compile(r'^[A-Z]{4}\d{1,2}$')


def _is_valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_RE.match(ticker.upper()))


@st.cache_data(ttl=3600, show_spinner=False)
def get_b3_stocks(search: str = "") -> pd.DataFrame:
    """
    Retorna DataFrame apenas com ações ordinárias/preferenciais da B3.
    Exclui fracionados, BDRs, direitos, recibos e outros derivados.
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
    except Exception:
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

    for col in ["ticker", "nome", "preco", "variacao_pct", "volume", "market_cap", "setor"]:
        if col not in df.columns:
            df[col] = None

    # Filtra apenas tickers padrão (remove fracionados, BDRs, direitos, etc)
    df = df[df["ticker"].apply(_is_valid_ticker)].copy()

    # Remove duplicatas — mantém o de maior volume (ex: PETR3 e PETR4 ficam, mas não PETR3F)
    df = df.drop_duplicates(subset=["ticker"])

    # Remove empresas sem preço
    df = df[df["preco"].notna() & (df["preco"] > 0)]

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
