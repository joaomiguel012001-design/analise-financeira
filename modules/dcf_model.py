"""
Modelo de Valuation por DCF (Discounted Cash Flow).

Metodologia:
- Projeção de FCF para 5 ou 10 anos com taxa de crescimento estimada
- Valor Terminal pelo modelo de Gordon (g constante)
- WACC calculado via CAPM para custo do equity
- Margem de segurança de 20% aplicada ao valor intrínseco
"""

import numpy as np
import pandas as pd
from typing import Optional


# Taxas de referência (atualizadas manualmente ou via API)
RISK_FREE_RATE_BR = 0.135   # SELIC ~13.5% a.a. (Brasil)
RISK_FREE_RATE_US = 0.045   # Treasury 10Y ~4.5% (EUA)
EQUITY_RISK_PREMIUM_BR = 0.065  # Prêmio de risco Brasil
EQUITY_RISK_PREMIUM_US = 0.055  # Prêmio de risco EUA
TERMINAL_GROWTH_RATE_BR = 0.04  # Crescimento perpétuo Brasil
TERMINAL_GROWTH_RATE_US = 0.025 # Crescimento perpétuo EUA
MARGEM_SEGURANCA = 0.20         # 20% de margem de segurança


def detect_market(ticker: str) -> str:
    """Detecta se é mercado brasileiro ou americano."""
    return "BR" if ticker.endswith(".SA") else "US"


def calculate_wacc(
    beta: float,
    market: str,
    debt_ratio: float = 0.30,
    tax_rate: float = 0.34,
    cost_of_debt: float = None,
) -> float:
    """
    Calcula o WACC (Custo Médio Ponderado de Capital).

    WACC = E/(D+E) * Ke + D/(D+E) * Kd * (1 - T)
    Ke = Rf + β * (Rm - Rf)
    """
    if market == "BR":
        rf = RISK_FREE_RATE_BR
        erp = EQUITY_RISK_PREMIUM_BR
    else:
        rf = RISK_FREE_RATE_US
        erp = EQUITY_RISK_PREMIUM_US

    # Custo do equity via CAPM
    beta = max(0.3, min(beta or 1.0, 3.0))  # limita entre 0.3 e 3.0
    cost_of_equity = rf + beta * erp

    # Custo da dívida
    if cost_of_debt is None:
        cost_of_debt = rf + 0.03  # spread padrão de 3%

    equity_ratio = 1 - debt_ratio

    wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - tax_rate))
    return wacc


def estimate_growth_rates(fcf_history: list[float], metrics: dict) -> tuple[float, float]:
    """
    Estima taxa de crescimento para fase 1 (5 anos) e fase 2 (próximos 5 anos).
    Usa CAGR histórico do FCF + ajustes por setor.
    """
    market = detect_market(metrics.get("ticker", ""))

    # CAGR histórico do FCF
    historical_cagr = 0.0
    if len(fcf_history) >= 2:
        valid_fcf = [f for f in fcf_history if f and f > 0]
        if len(valid_fcf) >= 2:
            years = len(valid_fcf) - 1
            historical_cagr = (valid_fcf[0] / valid_fcf[-1]) ** (1 / years) - 1

    # Crescimento de receita como referência adicional
    rev_growth = metrics.get("crescimento_receita") or 0.0

    # Combina estimativas (60% histórico FCF + 40% crescimento receita)
    if historical_cagr != 0:
        growth_phase1 = 0.6 * historical_cagr + 0.4 * rev_growth
    else:
        growth_phase1 = rev_growth if rev_growth else 0.05

    # Limita crescimento fase 1
    if market == "BR":
        growth_phase1 = max(-0.15, min(growth_phase1, 0.30))
    else:
        growth_phase1 = max(-0.10, min(growth_phase1, 0.25))

    # Fase 2: desacelera para metade do crescimento fase 1
    growth_phase2 = growth_phase1 * 0.5

    if market == "BR":
        terminal_g = TERMINAL_GROWTH_RATE_BR
        growth_phase2 = max(terminal_g, min(growth_phase2, growth_phase1))
    else:
        terminal_g = TERMINAL_GROWTH_RATE_US
        growth_phase2 = max(terminal_g, min(growth_phase2, growth_phase1))

    return growth_phase1, growth_phase2


def run_dcf(
    metrics: dict,
    fcf_history: list[float],
    years_phase1: int = 5,
    years_phase2: int = 5,
    custom_wacc: Optional[float] = None,
    custom_growth_phase1: Optional[float] = None,
    custom_growth_phase2: Optional[float] = None,
) -> dict:
    """
    Executa o modelo DCF completo.

    Retorna dicionário com:
    - valor_intrinseco: preço justo por ação
    - valor_intrinseco_com_margem: com margem de segurança
    - upside: % de upside/downside vs preço atual
    - detalhes: projeções ano a ano
    - wacc, growth_phase1, growth_phase2, terminal_value
    """
    market = detect_market(metrics.get("ticker", ""))
    preco_atual = metrics.get("preco_atual")
    acoes = metrics.get("acoes_em_circulacao")
    fcf_atual = metrics.get("fcf")

    # Validações básicas
    erros = []
    if not preco_atual:
        erros.append("Preço atual não disponível")
    if not acoes or acoes <= 0:
        erros.append("Número de ações não disponível")
    if not fcf_atual or fcf_atual <= 0:
        # Tenta usar média histórica positiva
        positive_fcfs = [f for f in (fcf_history or []) if f and f > 0]
        if positive_fcfs:
            fcf_atual = np.mean(positive_fcfs)
        else:
            erros.append("FCF atual não disponível ou negativo — DCF não aplicável")

    if erros:
        return {"erro": erros, "aplicavel": False}

    # Parâmetros
    beta = metrics.get("beta") or 1.0
    divida_total = metrics.get("divida_total") or 0
    market_cap = metrics.get("market_cap") or (preco_atual * acoes)
    ev = metrics.get("enterprise_value") or (market_cap + divida_total)
    debt_ratio = divida_total / ev if ev > 0 else 0.30

    wacc = custom_wacc or calculate_wacc(beta, market, debt_ratio)

    g1, g2 = estimate_growth_rates(fcf_history, metrics)
    growth_phase1 = custom_growth_phase1 if custom_growth_phase1 is not None else g1
    growth_phase2 = custom_growth_phase2 if custom_growth_phase2 is not None else g2

    terminal_g = TERMINAL_GROWTH_RATE_BR if market == "BR" else TERMINAL_GROWTH_RATE_US

    # Garante que WACC > terminal_g
    if wacc <= terminal_g:
        wacc = terminal_g + 0.03

    # Projeção de FCF
    projecoes = []
    fcf = fcf_atual

    # Fase 1
    for ano in range(1, years_phase1 + 1):
        fcf = fcf * (1 + growth_phase1)
        pv = fcf / ((1 + wacc) ** ano)
        projecoes.append({
            "ano": ano,
            "fase": "Crescimento",
            "fcf_projetado": fcf,
            "pv_fcf": pv,
            "taxa_crescimento": growth_phase1,
        })

    # Fase 2
    for ano in range(years_phase1 + 1, years_phase1 + years_phase2 + 1):
        fcf = fcf * (1 + growth_phase2)
        pv = fcf / ((1 + wacc) ** ano)
        projecoes.append({
            "ano": ano,
            "fase": "Desaceleração",
            "fcf_projetado": fcf,
            "pv_fcf": pv,
            "taxa_crescimento": growth_phase2,
        })

    # Valor Terminal (Gordon Growth Model)
    fcf_terminal = fcf * (1 + terminal_g)
    terminal_value = fcf_terminal / (wacc - terminal_g)
    pv_terminal = terminal_value / ((1 + wacc) ** (years_phase1 + years_phase2))

    # Soma dos PVs
    pv_fcfs = sum(p["pv_fcf"] for p in projecoes)
    enterprise_value_dcf = pv_fcfs + pv_terminal

    # Equity Value = EV - Dívida Líquida
    divida_liquida = metrics.get("divida_liquida") or (divida_total - (metrics.get("caixa") or 0))
    equity_value = enterprise_value_dcf - divida_liquida

    # Valor por ação
    valor_intrinseco = equity_value / acoes
    valor_com_margem = valor_intrinseco * (1 - MARGEM_SEGURANCA)

    # Upside
    upside = ((valor_intrinseco / preco_atual) - 1) * 100 if preco_atual else None
    upside_com_margem = ((valor_com_margem / preco_atual) - 1) * 100 if preco_atual else None

    # % que o valor terminal representa
    pct_terminal = (pv_terminal / enterprise_value_dcf) * 100 if enterprise_value_dcf else 0

    return {
        "aplicavel": True,
        "valor_intrinseco": valor_intrinseco,
        "valor_com_margem": valor_com_margem,
        "upside_pct": upside,
        "upside_com_margem_pct": upside_com_margem,
        "preco_atual": preco_atual,
        "enterprise_value_dcf": enterprise_value_dcf,
        "equity_value": equity_value,
        "pv_fcfs": pv_fcfs,
        "pv_terminal": pv_terminal,
        "pct_terminal": pct_terminal,
        "terminal_value": terminal_value,
        "wacc": wacc,
        "growth_phase1": growth_phase1,
        "growth_phase2": growth_phase2,
        "terminal_growth": terminal_g,
        "fcf_base": fcf_atual,
        "beta_usado": beta,
        "market": market,
        "projecoes": pd.DataFrame(projecoes),
    }


def sensitivity_analysis(
    metrics: dict,
    fcf_history: list[float],
    base_result: dict,
) -> pd.DataFrame:
    """
    Análise de sensibilidade do DCF.
    Varia WACC e taxa de crescimento fase 1.
    Retorna DataFrame com matriz de valores intrínsecos.
    """
    if not base_result.get("aplicavel"):
        return pd.DataFrame()

    base_wacc = base_result["wacc"]
    base_g1 = base_result["growth_phase1"]

    wacc_range = [base_wacc - 0.02, base_wacc - 0.01, base_wacc, base_wacc + 0.01, base_wacc + 0.02]
    growth_range = [base_g1 - 0.05, base_g1 - 0.025, base_g1, base_g1 + 0.025, base_g1 + 0.05]

    matrix = {}
    for g in growth_range:
        row = {}
        for w in wacc_range:
            if w <= 0.01:
                row[f"{w:.1%}"] = None
                continue
            result = run_dcf(metrics, fcf_history, custom_wacc=w, custom_growth_phase1=g)
            row[f"{w:.1%}"] = round(result.get("valor_intrinseco", 0), 2) if result.get("aplicavel") else None
        matrix[f"{g:.1%}"] = row

    df = pd.DataFrame(matrix).T
    df.index.name = "Crescimento Fase 1"
    df.columns.name = "WACC"
    return df
