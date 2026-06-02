"""
Módulo de análise de saúde financeira.
Calcula e classifica indicadores financeiros com scores de 0-100.
"""

import pandas as pd
import numpy as np
from typing import Optional


def score_indicator(value, thresholds: list, reverse: bool = False) -> int:
    """
    Retorna score 1-5 baseado em thresholds.
    thresholds = [ruim, fraco, ok, bom] → 5 faixas
    reverse=True: menor é melhor (ex: dívida)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 3  # neutro se sem dados

    if reverse:
        if value > thresholds[3]: return 1
        if value > thresholds[2]: return 2
        if value > thresholds[1]: return 3
        if value > thresholds[0]: return 4
        return 5
    else:
        if value >= thresholds[3]: return 5
        if value >= thresholds[2]: return 4
        if value >= thresholds[1]: return 3
        if value >= thresholds[0]: return 2
        return 1


def classify_score(score: float) -> tuple[str, str]:
    """Retorna (classificação, emoji_cor) baseado no score total."""
    if score >= 80:
        return "Excelente", "🟢"
    elif score >= 65:
        return "Bom", "🟡"
    elif score >= 45:
        return "Regular", "🟠"
    else:
        return "Crítico", "🔴"


def analyze_financial_health(metrics: dict) -> dict:
    """
    Análise completa de saúde financeira.
    Retorna scores por categoria e score geral.
    """
    is_br = (metrics.get("ticker", "").endswith(".SA"))

    indicadores = {}

    # ─── 1. LUCRATIVIDADE ──────────────────────────────────────────
    margem_liquida = metrics.get("margem_liquida")
    margem_bruta = metrics.get("margem_bruta")
    margem_op = metrics.get("margem_operacional")
    roe = metrics.get("roe")
    roa = metrics.get("roa")

    indicadores["margem_liquida"] = {
        "valor": margem_liquida,
        "score": score_indicator(margem_liquida, [-0.05, 0.03, 0.08, 0.15]),
        "label": "Margem Líquida",
        "formato": "pct",
        "categoria": "Lucratividade",
    }
    indicadores["margem_bruta"] = {
        "valor": margem_bruta,
        "score": score_indicator(margem_bruta, [0.10, 0.20, 0.35, 0.50]),
        "label": "Margem Bruta",
        "formato": "pct",
        "categoria": "Lucratividade",
    }
    indicadores["margem_operacional"] = {
        "valor": margem_op,
        "score": score_indicator(margem_op, [-0.05, 0.05, 0.12, 0.20]),
        "label": "Margem Operacional",
        "formato": "pct",
        "categoria": "Lucratividade",
    }
    indicadores["roe"] = {
        "valor": roe,
        "score": score_indicator(roe, [0.0, 0.08, 0.15, 0.25]),
        "label": "ROE",
        "formato": "pct",
        "categoria": "Lucratividade",
    }
    indicadores["roa"] = {
        "valor": roa,
        "score": score_indicator(roa, [-0.01, 0.03, 0.07, 0.12]),
        "label": "ROA",
        "formato": "pct",
        "categoria": "Lucratividade",
    }

    # ─── 2. ENDIVIDAMENTO ───────────────────────────────────────────
    divida_ebitda_raw = metrics.get("divida_ebitda")  # yfinance retorna D/E, não D/EBITDA
    divida_total = metrics.get("divida_total") or 0
    ebitda = metrics.get("ebitda") or 1
    divida_liquida = metrics.get("divida_liquida") or 0

    # Calcula D/EBITDA real
    divida_ebitda = abs(divida_liquida / ebitda) if ebitda and ebitda != 0 else None

    indicadores["divida_ebitda"] = {
        "valor": divida_ebitda,
        "score": score_indicator(divida_ebitda, [1.0, 2.0, 3.5, 5.0], reverse=True),
        "label": "Dívida Líq. / EBITDA",
        "formato": "mult",
        "categoria": "Endividamento",
    }

    # Cobertura de juros (se disponível via yfinance)
    interest_coverage = None
    if metrics.get("receita_total") and metrics.get("ebitda"):
        # Estimativa simplificada
        pass

    # D/E do yfinance (como percentual)
    de_ratio = (divida_ebitda_raw or 0) / 100 if divida_ebitda_raw else None
    indicadores["de_ratio"] = {
        "valor": de_ratio,
        "score": score_indicator(de_ratio, [0.3, 0.8, 1.5, 2.5], reverse=True),
        "label": "Dívida / Patrimônio",
        "formato": "mult",
        "categoria": "Endividamento",
    }

    # ─── 3. CRESCIMENTO ─────────────────────────────────────────────
    crescimento_receita = metrics.get("crescimento_receita")
    crescimento_lucro = metrics.get("crescimento_lucro")

    indicadores["crescimento_receita"] = {
        "valor": crescimento_receita,
        "score": score_indicator(crescimento_receita, [-0.05, 0.02, 0.08, 0.15]),
        "label": "Crescimento Receita (YoY)",
        "formato": "pct",
        "categoria": "Crescimento",
    }
    indicadores["crescimento_lucro"] = {
        "valor": crescimento_lucro,
        "score": score_indicator(crescimento_lucro, [-0.10, 0.0, 0.10, 0.20]),
        "label": "Crescimento Lucro (YoY)",
        "formato": "pct",
        "categoria": "Crescimento",
    }

    # ─── 4. VALUATION ───────────────────────────────────────────────
    pl = metrics.get("p_l")
    pvp = metrics.get("p_vp")
    ev_ebitda = metrics.get("ev_ebitda")
    div_yield = metrics.get("dividend_yield") or 0

    # P/L: contexto Brasil vs EUA
    pl_thresholds = [5, 10, 20, 35] if is_br else [8, 15, 25, 40]
    indicadores["p_l"] = {
        "valor": pl,
        "score": score_indicator(pl, pl_thresholds, reverse=True) if pl and pl > 0 else 3,
        "label": "P/L",
        "formato": "mult",
        "categoria": "Valuation",
    }
    indicadores["p_vp"] = {
        "valor": pvp,
        "score": score_indicator(pvp, [0.5, 1.0, 2.0, 4.0], reverse=True) if pvp else 3,
        "label": "P/VP",
        "formato": "mult",
        "categoria": "Valuation",
    }
    indicadores["ev_ebitda"] = {
        "valor": ev_ebitda,
        "score": score_indicator(ev_ebitda, [3, 6, 12, 20], reverse=True) if ev_ebitda else 3,
        "label": "EV/EBITDA",
        "formato": "mult",
        "categoria": "Valuation",
    }
    indicadores["dividend_yield"] = {
        "valor": div_yield,
        "score": score_indicator(div_yield, [0.0, 0.02, 0.05, 0.08]),
        "label": "Dividend Yield",
        "formato": "pct",
        "categoria": "Valuation",
    }

    # ─── 5. FCF ─────────────────────────────────────────────────────
    fcf = metrics.get("fcf") or 0
    receita = metrics.get("receita_total") or 1
    fcf_yield_receita = fcf / receita if receita and fcf else None

    indicadores["fcf_yield"] = {
        "valor": fcf_yield_receita,
        "score": score_indicator(fcf_yield_receita, [-0.05, 0.0, 0.05, 0.12]),
        "label": "FCF / Receita",
        "formato": "pct",
        "categoria": "Geração de Caixa",
    }

    # ─── SCORES POR CATEGORIA ───────────────────────────────────────
    categorias = {}
    for key, ind in indicadores.items():
        cat = ind["categoria"]
        if cat not in categorias:
            categorias[cat] = []
        categorias[cat].append(ind["score"])

    scores_categoria = {}
    for cat, scores in categorias.items():
        avg = np.mean(scores)
        scores_categoria[cat] = round((avg / 5) * 100, 1)

    score_geral = round(np.mean(list(scores_categoria.values())), 1)
    classificacao, cor = classify_score(score_geral)

    return {
        "indicadores": indicadores,
        "scores_categoria": scores_categoria,
        "score_geral": score_geral,
        "classificacao": classificacao,
        "cor": cor,
    }


def format_value(value, formato: str, moeda: str = "R$") -> str:
    """Formata valor para exibição."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    try:
        if formato == "pct":
            return f"{value * 100:.1f}%"
        elif formato == "mult":
            return f"{value:.1f}x"
        elif formato == "moeda_bilhoes":
            return f"{moeda} {value / 1e9:.2f}B"
        elif formato == "moeda_milhoes":
            return f"{moeda} {value / 1e6:.1f}M"
        else:
            return str(round(value, 2))
    except Exception:
        return "N/D"


def compare_competitors(main_metrics: dict, competitor_metrics: list[dict]) -> pd.DataFrame:
    """
    Compara métricas-chave entre empresa principal e concorrentes.
    """
    cols = [
        ("P/L", "p_l", "mult"),
        ("P/VP", "p_vp", "mult"),
        ("EV/EBITDA", "ev_ebitda", "mult"),
        ("Marg. Líq.", "margem_liquida", "pct"),
        ("ROE", "roe", "pct"),
        ("Div. Yield", "dividend_yield", "pct"),
        ("Cresc. Rec.", "crescimento_receita", "pct"),
    ]

    rows = []
    for m in [main_metrics] + competitor_metrics:
        row = {"Empresa": m.get("nome", m.get("ticker", "N/D"))}
        for label, key, fmt in cols:
            val = m.get(key)
            row[label] = format_value(val, fmt)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Empresa")
    return df
