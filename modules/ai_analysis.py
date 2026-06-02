"""
Módulo de análise qualitativa via Claude API.
Gera: modelo de negócios, pontos positivos/negativos,
análise macro do setor e comparação com concorrentes.
"""

import os
import json
from typing import Optional
import anthropic


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY não configurada no arquivo .env")
    return anthropic.Anthropic(api_key=api_key)


def _call_claude(prompt: str, system: str = None, max_tokens: int = 2000) -> str:
    """Chamada base ao Claude API."""
    client = get_client()
    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def analyze_business_model(metrics: dict, dcf_result: dict, health_result: dict) -> dict:
    """
    Gera análise do modelo de negócios, pontos positivos e negativos.
    Retorna JSON estruturado.
    """
    nome = metrics.get("nome", metrics.get("ticker_original", "N/D"))
    setor = metrics.get("setor", "N/D")
    industria = metrics.get("industria", "N/D")
    descricao = metrics.get("descricao", "")[:500]  # limita para não estourar tokens
    moeda = metrics.get("moeda", "BRL")
    pais = metrics.get("pais", "Brasil")

    # Resumo financeiro para contexto
    financials_summary = f"""
- Receita Total: {_fmt_bilhoes(metrics.get('receita_total'), moeda)}
- EBITDA: {_fmt_bilhoes(metrics.get('ebitda'), moeda)}
- Lucro Líquido: {_fmt_bilhoes(metrics.get('lucro_liquido'), moeda)}
- Margem Líquida: {_fmt_pct(metrics.get('margem_liquida'))}
- ROE: {_fmt_pct(metrics.get('roe'))}
- Dívida Líquida: {_fmt_bilhoes(metrics.get('divida_liquida'), moeda)}
- FCF: {_fmt_bilhoes(metrics.get('fcf'), moeda)}
- P/L: {_fmt_mult(metrics.get('p_l'))}
- EV/EBITDA: {_fmt_mult(metrics.get('ev_ebitda'))}
- Crescimento Receita: {_fmt_pct(metrics.get('crescimento_receita'))}
- Score Saúde Financeira: {health_result.get('score_geral', 'N/D')}/100 ({health_result.get('classificacao', 'N/D')})
"""

    dcf_summary = ""
    if dcf_result.get("aplicavel"):
        dcf_summary = f"""
- Valor Intrínseco DCF: {moeda} {dcf_result.get('valor_intrinseco', 0):.2f}
- Preço Atual: {moeda} {dcf_result.get('preco_atual', 0):.2f}
- Upside/Downside: {dcf_result.get('upside_pct', 0):.1f}%
- WACC utilizado: {dcf_result.get('wacc', 0):.1%}
- Taxa de crescimento projetada (fase 1): {dcf_result.get('growth_phase1', 0):.1%}
"""
    else:
        dcf_summary = "- DCF não aplicável (FCF negativo ou dados insuficientes)"

    system = """Você é um analista financeiro sênior especializado em mercados Brasil e EUA.
Responda SEMPRE em português do Brasil.
Forneça análises objetivas, baseadas em dados, com linguagem profissional mas acessível.
Responda no formato JSON exatamente como solicitado."""

    prompt = f"""Analise a empresa **{nome}** ({setor} / {industria}, {pais}).

Descrição: {descricao}

**Dados Financeiros:**
{financials_summary}

**Valuation DCF:**
{dcf_summary}

Responda em JSON com esta estrutura exata:
{{
  "modelo_de_negocios": {{
    "resumo": "Parágrafo de 3-4 linhas descrevendo como a empresa ganha dinheiro, seus principais segmentos, vantagens competitivas (moat) e posicionamento de mercado.",
    "fontes_de_receita": ["fonte 1", "fonte 2", "fonte 3"],
    "vantagens_competitivas": ["vantagem 1", "vantagem 2", "vantagem 3"]
  }},
  "pontos_positivos": [
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas com dados quando possível"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}}
  ],
  "pontos_negativos": [
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas com dados quando possível"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}},
    {{"titulo": "Título curto", "descricao": "Explicação em 1-2 linhas"}}
  ],
  "veredicto_investimento": {{
    "classificacao": "Compra Forte | Compra | Neutro | Venda | Venda Forte",
    "justificativa": "2-3 linhas explicando o veredicto baseado em DCF + fundamentos",
    "horizonte_recomendado": "Curto Prazo (< 1 ano) | Médio Prazo (1-3 anos) | Longo Prazo (3+ anos)"
  }}
}}"""

    try:
        response = _call_claude(prompt, system=system, max_tokens=2500)
        # Extrai JSON da resposta
        json_str = _extract_json(response)
        return json.loads(json_str)
    except Exception as e:
        return {"erro": str(e), "raw": response if 'response' in locals() else ""}


def analyze_macro_sector(metrics: dict) -> dict:
    """Análise macro do setor da empresa."""
    nome = metrics.get("nome", "N/D")
    setor = metrics.get("setor", "N/D")
    industria = metrics.get("industria", "N/D")
    pais = metrics.get("pais", "Brasil")

    system = """Você é um economista e estrategista de mercado sênior.
Responda em português do Brasil, com dados atualizados e perspectiva macro-econômica.
Use formato JSON como solicitado."""

    prompt = f"""Faça uma análise macro do setor de **{industria}** ({setor}) no mercado de **{pais}** para o contexto atual (2025-2026).

Esta análise é para avaliar o ambiente de negócios da empresa **{nome}**.

Responda em JSON com esta estrutura exata:
{{
  "panorama_setor": {{
    "descricao": "2-3 parágrafos sobre o estado atual do setor, tendências e perspectivas",
    "ciclo_atual": "Expansão | Estabilidade | Contração | Turnaround",
    "justificativa_ciclo": "1-2 linhas explicando o ciclo"
  }},
  "drivers_crescimento": [
    {{"driver": "Nome do driver", "impacto": "Alto | Médio | Baixo", "descricao": "1 linha"}},
    {{"driver": "Nome do driver", "impacto": "Alto | Médio | Baixo", "descricao": "1 linha"}},
    {{"driver": "Nome do driver", "impacto": "Alto | Médio | Baixo", "descricao": "1 linha"}}
  ],
  "riscos_macro": [
    {{"risco": "Nome do risco", "probabilidade": "Alta | Média | Baixa", "descricao": "1 linha"}},
    {{"risco": "Nome do risco", "probabilidade": "Alta | Média | Baixa", "descricao": "1 linha"}},
    {{"risco": "Nome do risco", "probabilidade": "Alta | Média | Baixa", "descricao": "1 linha"}}
  ],
  "tendencias_estruturais": [
    "Tendência 1 de longo prazo",
    "Tendência 2 de longo prazo",
    "Tendência 3 de longo prazo"
  ],
  "regulatorio": {{
    "ambiente": "Favorável | Neutro | Desfavorável",
    "descricao": "1-2 linhas sobre ambiente regulatório"
  }},
  "perspectiva_12_meses": "Otimista | Moderadamente Otimista | Neutro | Moderadamente Pessimista | Pessimista",
  "justificativa_perspectiva": "2-3 linhas com a justificativa da perspectiva"
}}"""

    try:
        response = _call_claude(prompt, system=system, max_tokens=2000)
        json_str = _extract_json(response)
        return json.loads(json_str)
    except Exception as e:
        return {"erro": str(e)}


def analyze_competitors_qualitative(
    main_metrics: dict,
    competitor_metrics_list: list[dict],
    comparison_df_str: str,
) -> dict:
    """Análise qualitativa comparando empresa com concorrentes."""
    nome = main_metrics.get("nome", "N/D")
    setor = main_metrics.get("setor", "N/D")

    concorrentes_nomes = [m.get("nome", m.get("ticker", "N/D")) for m in competitor_metrics_list]

    system = """Você é um analista de equity research sênior.
Responda em português do Brasil.
Use formato JSON como solicitado."""

    prompt = f"""Compare **{nome}** com seus concorrentes no setor de {setor}.

Concorrentes analisados: {', '.join(concorrentes_nomes) if concorrentes_nomes else 'Dados não disponíveis'}

Dados comparativos:
{comparison_df_str}

Responda em JSON com esta estrutura exata:
{{
  "posicao_competitiva": {{
    "classificacao": "Líder de Mercado | Challenger | Seguidor | Nicho",
    "descricao": "2-3 linhas sobre a posição competitiva da empresa"
  }},
  "vantagens_vs_concorrentes": [
    "Vantagem 1 frente aos concorrentes com dado específico",
    "Vantagem 2",
    "Vantagem 3"
  ],
  "desvantagens_vs_concorrentes": [
    "Desvantagem 1 frente aos concorrentes",
    "Desvantagem 2",
    "Desvantagem 3"
  ],
  "mais_barata_que": "Nome da empresa mais cara que {nome} e por quê (ou 'Nenhuma')",
  "melhor_alternativa": {{
    "empresa": "Nome da empresa ou '{nome}' se for a melhor opção",
    "motivo": "1-2 linhas explicando por que é a melhor opção de investimento no setor"
  }},
  "resumo_comparativo": "Parágrafo de 3-4 linhas com visão geral da comparação"
}}"""

    try:
        response = _call_claude(prompt, system=system, max_tokens=1500)
        json_str = _extract_json(response)
        result = json.loads(json_str)
        # Substitui placeholder {nome} que pode ter ficado no JSON
        result_str = json.dumps(result).replace("{nome}", nome)
        return json.loads(result_str)
    except Exception as e:
        return {"erro": str(e)}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extrai JSON de uma resposta que pode conter texto antes/depois."""
    # Tenta encontrar bloco ```json ... ```
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    # Tenta encontrar { ... } diretamente
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text


def _fmt_bilhoes(value, moeda="BRL") -> str:
    if not value:
        return "N/D"
    symbol = "R$" if moeda == "BRL" else "US$"
    if abs(value) >= 1e9:
        return f"{symbol} {value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{symbol} {value/1e6:.1f}M"
    return f"{symbol} {value:,.0f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "N/D"
    return f"{value * 100:.1f}%"


def _fmt_mult(value) -> str:
    if value is None:
        return "N/D"
    return f"{value:.1f}x"
