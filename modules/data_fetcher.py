"""
Módulo de busca de dados financeiros via brapi.dev (primário) com fallback para yfinance.
Suporta ações brasileiras (B3) e americanas (NYSE/NASDAQ).
"""

import os
import requests
import pandas as pd
import numpy as np
import re
from typing import Optional

BRAPI_BASE = "https://brapi.dev/api"
_TOKEN = os.getenv("BRAPI_TOKEN", "")


def _brapi_params(extra: dict = {}) -> dict:
    p = dict(extra)
    if _TOKEN:
        p["token"] = _TOKEN
    return p


def normalize_ticker(ticker: str) -> str:
    """Normaliza o ticker: adiciona .SA para ações brasileiras se necessário."""
    ticker = ticker.upper().strip()
    if re.match(r'^[A-Z]{4}\d{1,2}$', ticker) and not ticker.endswith('.SA'):
        return ticker + '.SA'
    return ticker


def _is_brazilian(ticker: str) -> bool:
    t = ticker.upper().replace(".SA", "")
    return bool(re.match(r'^[A-Z]{4}\d{1,2}$', t))


def get_company_data(ticker: str) -> dict:
    """
    Busca todos os dados financeiros de uma empresa via brapi.dev.
    """
    ticker_clean = ticker.upper().strip().replace(".SA", "")
    normalized = normalize_ticker(ticker)

    if _is_brazilian(ticker):
        return _get_brapi_data(ticker_clean, normalized)
    else:
        return _get_yfinance_data(ticker, normalized)


def _get_brapi_data(ticker: str, normalized: str) -> dict:
    """Busca dados via brapi.dev para ações brasileiras."""
    params = _brapi_params({
        "range": "5y",
        "interval": "1mo",
        "fundamental": "true",
        "dividends": "true",
    })

    try:
        resp = requests.get(f"{BRAPI_BASE}/quote/{ticker}", params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"Sem dados para {ticker}")
        r = results[0]
    except Exception as e:
        raise Exception(f"Too Many Requests. Rate limited. Try after a while." if "429" in str(e) else str(e))

    # Monta estrutura compatível com o restante do app
    info = _build_info_from_brapi(r)
    history = _build_history_from_brapi(r)
    income_stmt, balance_sheet, cash_flow = _build_financials_from_brapi(r)
    dividends = _build_dividends_from_brapi(r)

    return {
        "ticker": normalized,
        "ticker_original": ticker,
        "info": info,
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "quarterly_income": pd.DataFrame(),
        "history": history,
        "dividends": dividends,
    }


def _build_info_from_brapi(r: dict) -> dict:
    fd = r.get("financialData", {})
    ks = r.get("defaultKeyStatistics", {})
    sp = r.get("summaryProfile", {})

    price = r.get("regularMarketPrice") or r.get("currentPrice")
    shares = ks.get("sharesOutstanding", {})
    shares_val = shares.get("raw") if isinstance(shares, dict) else shares

    market_cap = r.get("marketCap")
    if not market_cap and price and shares_val:
        market_cap = price * shares_val

    def raw(v):
        return v.get("raw") if isinstance(v, dict) else v

    info = {
        "longName": r.get("longName") or r.get("shortName", ""),
        "shortName": r.get("shortName", ""),
        "sector": sp.get("sector", r.get("sector", "")),
        "industry": sp.get("industry", r.get("industry", "")),
        "longBusinessSummary": sp.get("longBusinessSummary", ""),
        "country": sp.get("country", "Brazil"),
        "currency": r.get("currency", "BRL"),
        "exchange": r.get("exchange", "SAO"),
        "currentPrice": price,
        "regularMarketPrice": price,
        "marketCap": raw(market_cap) if isinstance(market_cap, dict) else market_cap,
        "enterpriseValue": raw(ks.get("enterpriseValue")),
        "sharesOutstanding": raw(shares_val) if isinstance(shares_val, dict) else shares_val,
        "beta": raw(ks.get("beta")),
        # Valuation
        "trailingPE": raw(ks.get("trailingPE")) or raw(fd.get("trailingPE")),
        "priceToBook": raw(ks.get("priceToBook")),
        "enterpriseToEbitda": raw(ks.get("enterpriseToEbitda")),
        "enterpriseToRevenue": raw(ks.get("enterpriseToRevenue")),
        "pegRatio": raw(ks.get("pegRatio")),
        # Margens
        "grossMargins": raw(fd.get("grossMargins")),
        "operatingMargins": raw(fd.get("operatingMargins")),
        "profitMargins": raw(fd.get("profitMargins")) or raw(ks.get("profitMargins")),
        "ebitdaMargins": raw(ks.get("ebitdaMargins")),
        # Rentabilidade
        "returnOnEquity": raw(fd.get("returnOnEquity")),
        "returnOnAssets": raw(fd.get("returnOnAssets")),
        # Crescimento
        "revenueGrowth": raw(fd.get("revenueGrowth")),
        "earningsGrowth": raw(fd.get("earningsGrowth")),
        # Dívida
        "totalDebt": raw(fd.get("totalDebt")),
        "totalCash": raw(fd.get("totalCash")),
        "debtToEquity": raw(fd.get("debtToEquity")),
        # Dividendos
        "dividendYield": raw(ks.get("dividendYield")) or raw(r.get("dividendYield")),
        "payoutRatio": raw(ks.get("payoutRatio")),
        # FCF e resultados
        "freeCashflow": raw(fd.get("freeCashflow")),
        "totalRevenue": raw(fd.get("totalRevenue")),
        "ebitda": raw(fd.get("ebitda")),
        "netIncomeToCommon": raw(ks.get("netIncomeToCommon")),
    }
    return info


def _build_history_from_brapi(r: dict) -> pd.DataFrame:
    historical = r.get("historicalDataPrice", [])
    if not historical:
        return pd.DataFrame()
    rows = []
    for h in historical:
        if h.get("close"):
            rows.append({
                "Date": pd.to_datetime(h["date"], unit="s") if isinstance(h["date"], (int, float)) else pd.to_datetime(h["date"]),
                "Close": h.get("close"),
                "Open": h.get("open"),
                "High": h.get("high"),
                "Low": h.get("low"),
                "Volume": h.get("volume"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Date")
    return df


def _build_financials_from_brapi(r: dict) -> tuple:
    """Extrai demonstrativos financeiros do formato brapi."""
    income_stmt = pd.DataFrame()
    balance_sheet = pd.DataFrame()
    cash_flow = pd.DataFrame()

    # Income statement
    inc_hist = r.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
    if inc_hist:
        rows = {}
        for period in inc_hist:
            date = period.get("endDate", {}).get("fmt", str(period.get("endDate", "")))
            for key, val in period.items():
                if key == "endDate":
                    continue
                v = val.get("raw") if isinstance(val, dict) else val
                if v is not None:
                    rows.setdefault(key, {})[date] = v
        if rows:
            income_stmt = pd.DataFrame(rows).T

    # Balance sheet
    bal_hist = r.get("balanceSheetHistory", {}).get("balanceSheetStatements", [])
    if bal_hist:
        rows = {}
        for period in bal_hist:
            date = period.get("endDate", {}).get("fmt", str(period.get("endDate", "")))
            for key, val in period.items():
                if key == "endDate":
                    continue
                v = val.get("raw") if isinstance(val, dict) else val
                if v is not None:
                    rows.setdefault(key, {})[date] = v
        if rows:
            balance_sheet = pd.DataFrame(rows).T

    # Cash flow
    cf_hist = r.get("cashflowStatementHistory", {}).get("cashflowStatements", [])
    if cf_hist:
        rows = {}
        for period in cf_hist:
            date = period.get("endDate", {}).get("fmt", str(period.get("endDate", "")))
            for key, val in period.items():
                if key == "endDate":
                    continue
                v = val.get("raw") if isinstance(val, dict) else val
                if v is not None:
                    rows.setdefault(key, {})[date] = v
        if rows:
            cash_flow = pd.DataFrame(rows).T

    return income_stmt, balance_sheet, cash_flow


def _build_dividends_from_brapi(r: dict) -> pd.Series:
    divs = r.get("dividendsData", {}).get("cashDividends", [])
    if not divs:
        return pd.Series(dtype=float)
    data = {}
    for d in divs:
        try:
            date = pd.to_datetime(d.get("paymentDate") or d.get("approvedOn"))
            val = float(d.get("value", 0))
            data[date] = val
        except Exception:
            pass
    return pd.Series(data)


def _get_yfinance_data(ticker: str, normalized: str) -> dict:
    """Fallback para ações americanas via yfinance."""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    try:
        income_stmt = stock.financials
    except Exception:
        income_stmt = pd.DataFrame()
    try:
        balance_sheet = stock.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()
    try:
        cash_flow = stock.cashflow
    except Exception:
        cash_flow = pd.DataFrame()
    try:
        history = stock.history(period="5y")
    except Exception:
        history = pd.DataFrame()
    try:
        dividends = stock.dividends
    except Exception:
        dividends = pd.Series()
    return {
        "ticker": normalized,
        "ticker_original": ticker.upper(),
        "info": info,
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "quarterly_income": pd.DataFrame(),
        "history": history,
        "dividends": dividends,
    }


def extract_key_metrics(data: dict) -> dict:
    info = data["info"]
    income = data["income_stmt"]
    balance = data["balance_sheet"]
    cf = data["cash_flow"]

    metrics = {
        "nome": info.get("longName") or info.get("shortName", data["ticker_original"]),
        "setor": info.get("sector", "N/D"),
        "industria": info.get("industry", "N/D"),
        "descricao": info.get("longBusinessSummary", ""),
        "pais": info.get("country", "N/D"),
        "moeda": info.get("currency", "N/D"),
        "exchange": info.get("exchange", "N/D"),
        "preco_atual": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "acoes_em_circulacao": info.get("sharesOutstanding"),
        "beta": info.get("beta"),
        "p_l": info.get("trailingPE"),
        "p_vp": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_receita": info.get("enterpriseToRevenue"),
        "peg_ratio": info.get("pegRatio"),
        "margem_bruta": info.get("grossMargins"),
        "margem_operacional": info.get("operatingMargins"),
        "margem_liquida": info.get("profitMargins"),
        "margem_ebitda": info.get("ebitdaMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "roic": None,
        "crescimento_receita": info.get("revenueGrowth"),
        "crescimento_lucro": info.get("earningsGrowth"),
        "divida_total": info.get("totalDebt"),
        "caixa": info.get("totalCash"),
        "divida_liquida": None,
        "divida_ebitda": info.get("debtToEquity"),
        "dividend_yield": info.get("dividendYield"),
        "payout_ratio": info.get("payoutRatio"),
        "fcf": info.get("freeCashflow"),
        "receita_total": info.get("totalRevenue"),
        "ebitda": info.get("ebitda"),
        "lucro_liquido": info.get("netIncomeToCommon"),
    }

    if metrics["divida_total"] and metrics["caixa"]:
        metrics["divida_liquida"] = metrics["divida_total"] - metrics["caixa"]

    if not metrics["fcf"] and not cf.empty:
        try:
            fcf_keys = [k for k in cf.index if "Free Cash" in str(k) or "FreeCash" in str(k) or "freeCashflow" in str(k)]
            if fcf_keys:
                metrics["fcf"] = float(cf.loc[fcf_keys[0]].iloc[0])
            else:
                cfo_keys = [k for k in cf.index if "Operating" in str(k) and "Cash" in str(k)]
                capex_keys = [k for k in cf.index if "Capital" in str(k)]
                if cfo_keys and capex_keys:
                    cfo = float(cf.loc[cfo_keys[0]].iloc[0])
                    capex = float(cf.loc[capex_keys[0]].iloc[0])
                    metrics["fcf"] = cfo + capex
        except Exception:
            pass

    return metrics


def get_historical_fcf(data: dict) -> list:
    cf = data["cash_flow"]
    if cf.empty:
        return []
    fcf_series = []
    try:
        cfo_keys = [k for k in cf.index if "Operating" in str(k)]
        capex_keys = [k for k in cf.index if "Capital" in str(k)]
        free_cash_keys = [k for k in cf.index if "Free" in str(k)]

        if free_cash_keys:
            row = cf.loc[free_cash_keys[0]]
            fcf_series = [float(v) for v in row.values if pd.notna(v)]
        elif cfo_keys and capex_keys:
            cfo_row = cf.loc[cfo_keys[0]]
            capex_row = cf.loc[capex_keys[0]]
            for cfo_val, capex_val in zip(cfo_row.values, capex_row.values):
                if pd.notna(cfo_val) and pd.notna(capex_val):
                    fcf_series.append(float(cfo_val) + float(capex_val))
    except Exception:
        pass
    return fcf_series


def get_competitors(data: dict) -> list:
    sector_competitors = {
        "Real Estate": ["MULT3", "IGTI11", "BRPR3", "MALL11"],
        "Consumer Cyclical": ["MGLU3", "LREN3", "AMER3", "VVAR3"],
        "Financial Services": ["ITUB4", "BBDC4", "SANB11", "BBAS3"],
        "Energy": ["PETR4", "UGPA3", "CSAN3"],
        "Utilities": ["ELET3", "CPFE3", "ENGI11"],
        "Basic Materials": ["VALE3", "CSNA3", "GGBR4"],
        "Healthcare": ["RDOR3", "HAPV3", "FLRY3"],
        "Communication Services": ["VIVT3", "TIMS3", "OIBR3"],
        "Technology": ["AAPL", "MSFT", "GOOGL", "META"],
        "Consumer Defensive": ["WMT", "COST", "PG", "KO"],
        "Industrials": ["CAT", "DE", "GE", "HON"],
    }
    info = data["info"]
    sector = info.get("sector", "")
    ticker = data["ticker_original"].replace(".SA", "")
    competitors = sector_competitors.get(sector, [])
    competitors = [c for c in competitors if c != ticker]
    return competitors[:4]
