"""
Módulo de busca de dados financeiros via yfinance.
Suporta ações brasileiras (B3) e americanas (NYSE/NASDAQ).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def normalize_ticker(ticker: str) -> str:
    """Normaliza o ticker: adiciona .SA para ações brasileiras se necessário."""
    ticker = ticker.upper().strip()
    # Padrão brasileiro: 4 letras + 1 número (ex: ALOS3, PETR4, VALE3)
    import re
    if re.match(r'^[A-Z]{4}\d{1,2}$', ticker) and not ticker.endswith('.SA'):
        return ticker + '.SA'
    return ticker


def get_company_data(ticker: str) -> dict:
    """
    Busca todos os dados financeiros de uma empresa.
    Retorna um dicionário com info, demonstrativos e histórico.
    """
    normalized = normalize_ticker(ticker)
    stock = yf.Ticker(normalized)

    info = stock.info or {}

    # Demonstrativos financeiros
    try:
        income_stmt = stock.financials  # Demonstração de resultado (anual)
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
        quarterly_income = stock.quarterly_financials
    except Exception:
        quarterly_income = pd.DataFrame()

    # Histórico de preços (5 anos)
    try:
        history = stock.history(period="5y")
    except Exception:
        history = pd.DataFrame()

    # Dividendos
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
        "quarterly_income": quarterly_income,
        "history": history,
        "dividends": dividends,
    }


def extract_key_metrics(data: dict) -> dict:
    """Extrai métricas-chave do dicionário de dados."""
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
        # Valuation
        "p_l": info.get("trailingPE"),
        "p_vp": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_receita": info.get("enterpriseToRevenue"),
        "peg_ratio": info.get("pegRatio"),
        # Margens
        "margem_bruta": info.get("grossMargins"),
        "margem_operacional": info.get("operatingMargins"),
        "margem_liquida": info.get("profitMargins"),
        "margem_ebitda": info.get("ebitdaMargins"),
        # Rentabilidade
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "roic": None,  # Calculado separadamente
        # Crescimento
        "crescimento_receita": info.get("revenueGrowth"),
        "crescimento_lucro": info.get("earningsGrowth"),
        # Dívida
        "divida_total": info.get("totalDebt"),
        "caixa": info.get("totalCash"),
        "divida_liquida": None,
        "divida_ebitda": info.get("debtToEquity"),
        # Dividendos
        "dividend_yield": info.get("dividendYield"),
        "payout_ratio": info.get("payoutRatio"),
        # FCF
        "fcf": info.get("freeCashflow"),
        "receita_total": info.get("totalRevenue"),
        "ebitda": info.get("ebitda"),
        "lucro_liquido": info.get("netIncomeToCommon"),
    }

    # Calcula dívida líquida
    if metrics["divida_total"] and metrics["caixa"]:
        metrics["divida_liquida"] = metrics["divida_total"] - metrics["caixa"]

    # Tenta extrair FCF dos demonstrativos se não disponível via info
    if not metrics["fcf"] and not cf.empty:
        try:
            fcf_keys = [k for k in cf.index if "Free Cash" in str(k) or "FreeCash" in str(k)]
            if fcf_keys:
                metrics["fcf"] = float(cf.loc[fcf_keys[0]].iloc[0])
            else:
                # Calcula: CFO - Capex
                cfo_keys = [k for k in cf.index if "Operating" in str(k) and "Cash" in str(k)]
                capex_keys = [k for k in cf.index if "Capital" in str(k)]
                if cfo_keys and capex_keys:
                    cfo = float(cf.loc[cfo_keys[0]].iloc[0])
                    capex = float(cf.loc[capex_keys[0]].iloc[0])
                    metrics["fcf"] = cfo + capex  # capex é negativo
        except Exception:
            pass

    return metrics


def get_historical_fcf(data: dict) -> list[float]:
    """Extrai série histórica de FCF (últimos 4 anos)."""
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


def get_competitors(data: dict) -> list[str]:
    """Retorna lista de concorrentes baseado no setor."""
    sector_competitors = {
        # Brasil
        "Real Estate": ["MULT3.SA", "IGTI11.SA", "BRPR3.SA", "MALL11.SA"],
        "Consumer Cyclical": ["MGLU3.SA", "LREN3.SA", "AMER3.SA", "VVAR3.SA"],
        "Financial Services": ["ITUB4.SA", "BBDC4.SA", "SANB11.SA", "BBAS3.SA"],
        "Energy": ["PETR4.SA", "UGPA3.SA", "CSAN3.SA"],
        "Utilities": ["ELET3.SA", "CPFE3.SA", "ENGI11.SA"],
        "Basic Materials": ["VALE3.SA", "CSNA3.SA", "GGBR4.SA"],
        "Healthcare": ["RDOR3.SA", "HAPV3.SA", "FLRY3.SA"],
        "Communication Services": ["VIVT3.SA", "TIMS3.SA", "OIBR3.SA"],
        # USA
        "Technology": ["AAPL", "MSFT", "GOOGL", "META"],
        "Consumer Defensive": ["WMT", "COST", "PG", "KO"],
        "Industrials": ["CAT", "DE", "GE", "HON"],
    }

    info = data["info"]
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    ticker = data["ticker"]

    competitors = sector_competitors.get(sector, [])
    # Remove a própria empresa da lista
    competitors = [c for c in competitors if c != ticker]
    return competitors[:4]  # máximo 4 concorrentes
