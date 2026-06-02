"""
Analisador Financeiro - Aplicativo Principal (Streamlit)

Funcionalidades:
- Screener com todas as empresas da B3 (via brapi.dev)
- Valuation por DCF com análise de sensibilidade
- Relatório de saúde financeira com score
- Análise de modelo de negócios (via Claude AI)
- Análise macro do setor (via Claude AI)
- Comparação com concorrentes
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from modules.data_fetcher import (
    get_company_data, extract_key_metrics,
    get_historical_fcf, get_competitors, normalize_ticker
)
from modules.dcf_model import run_dcf, sensitivity_analysis
from modules.health_analysis import analyze_financial_health, format_value, compare_competitors
from modules.ai_analysis import (
    analyze_business_model, analyze_macro_sector, analyze_competitors_qualitative
)
from modules.b3_fetcher import get_b3_stocks

load_dotenv()

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisador Financeiro B3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px;
    margin: 4px 0;
    border-left: 4px solid #4CAF50;
}
.score-badge {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
}
.positive { color: #4CAF50; }
.negative { color: #f44336; }
.neutral  { color: #FF9800; }
.section-title {
    font-size: 1.3rem;
    font-weight: bold;
    border-bottom: 2px solid #4CAF50;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}
.stock-row:hover { background: #2a2d3e; cursor: pointer; }
</style>
""", unsafe_allow_html=True)


# ─── Navegação entre telas ────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "screener"


def go_to_analysis(ticker: str):
    st.session_state["page"] = "analysis"
    st.session_state["last_ticker"] = ticker
    for key in ["data", "metrics", "dcf", "health", "fcf_hist",
                "ai_biz", "ai_macro", "ai_comp", "comp_metrics"]:
        st.session_state.pop(key, None)


def go_to_screener():
    st.session_state["page"] = "screener"


# ══════════════════════════════════════════════════════════════════════════════
# TELA 1: SCREENER B3
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "screener":

    with st.sidebar:
        st.title("📊 Analisador B3")
        st.markdown("---")
        st.markdown("**Filtros**")

        busca = st.text_input("Buscar empresa ou ticker", placeholder="Ex: Petrobras, PETR4")
        st.markdown("---")
        st.caption("Dados via brapi.dev · yfinance")
        st.caption("⚠️ Não é recomendação de investimento.")

    st.title("📈 Empresas da B3")

    with st.spinner("Carregando empresas da B3..."):
        df_all = get_b3_stocks(search=busca if busca else "")

    if df_all.empty:
        st.error("Não foi possível carregar a lista de empresas. Verifique sua conexão ou configure BRAPI_TOKEN no .env")
        st.info("Você também pode analisar uma empresa diretamente digitando o ticker abaixo.")
        ticker_direto = st.text_input("Ticker direto", placeholder="Ex: PETR4, VALE3")
        if st.button("Analisar", type="primary") and ticker_direto:
            go_to_analysis(ticker_direto.upper().strip())
            st.rerun()
        st.stop()

    # ─── Filtros em linha ─────────────────────────────────────────────────────
    setores = sorted(df_all["setor"].dropna().unique().tolist())
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        setor_sel = st.selectbox("Setor", ["Todos"] + setores)
    with col_f2:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["Market Cap", "Preço", "Variação %", "Volume"],
        )
    with col_f3:
        ordem = st.radio("Ordem", ["Decrescente", "Crescente"], horizontal=True)

    # Aplica filtros
    df = df_all.copy()
    if setor_sel != "Todos":
        df = df[df["setor"] == setor_sel]

    sort_map = {
        "Market Cap": "market_cap",
        "Preço": "preco",
        "Variação %": "variacao_pct",
        "Volume": "volume",
    }
    sort_col = sort_map[ordenar_por]
    ascending = ordem == "Crescente"
    df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    st.markdown(f"**{len(df)} empresas encontradas**")
    st.markdown("---")

    # ─── Tabela principal ─────────────────────────────────────────────────────
    df_display = df[["ticker", "nome", "setor", "preco", "variacao_pct", "volume", "market_cap"]].copy()
    df_display.columns = ["Ticker", "Nome", "Setor", "Preço (R$)", "Variação %", "Volume", "Market Cap (R$)"]

    df_display["Preço (R$)"] = pd.to_numeric(df_display["Preço (R$)"], errors="coerce")
    df_display["Variação %"] = pd.to_numeric(df_display["Variação %"], errors="coerce")
    df_display["Volume"] = pd.to_numeric(df_display["Volume"], errors="coerce")
    df_display["Market Cap (R$)"] = pd.to_numeric(df_display["Market Cap (R$)"], errors="coerce")

    # Tabela interativa nativa — scrollável, ordenável, sem paginação manual
    event = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=520,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Nome": st.column_config.TextColumn("Nome", width="medium"),
            "Setor": st.column_config.TextColumn("Setor", width="medium"),
            "Preço (R$)": st.column_config.NumberColumn("Preço (R$)", format="R$ %.2f", width="small"),
            "Variação %": st.column_config.NumberColumn("Variação %", format="%.2f%%", width="small"),
            "Volume": st.column_config.NumberColumn("Volume", format="%.0f", width="small"),
            "Market Cap (R$)": st.column_config.NumberColumn("Market Cap", format="R$ %.0f", width="small"),
        },
    )

    # Abre análise ao clicar numa linha
    selected = event.selection.rows if event and event.selection else []
    if selected:
        ticker_sel = df_display.iloc[selected[0]]["Ticker"]
        go_to_analysis(ticker_sel)
        st.rerun()

    st.caption("Clique em uma linha para abrir a análise completa.")


# ══════════════════════════════════════════════════════════════════════════════
# TELA 2: ANÁLISE INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ─── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Analisador Financeiro")

        if st.button("← Voltar ao Screener", use_container_width=True):
            go_to_screener()
            st.rerun()

        st.markdown("---")

        ticker_input = st.text_input(
            "Ticker da empresa",
            value=st.session_state.get("last_ticker", "ALOS3"),
            help="Ex: ALOS3 (Brasil) ou AAPL (EUA).",
            placeholder="Ex: ALOS3, PETR4, AAPL"
        ).upper().strip()

        st.markdown("**Módulos de análise:**")
        run_ai = st.checkbox("Análise por IA (Claude)", value=True,
                             help="Requer ANTHROPIC_API_KEY no arquivo .env")
        run_competitors = st.checkbox("Análise de concorrentes", value=True)

        st.markdown("---")
        st.markdown("**Parâmetros DCF (opcional):**")
        custom_wacc = st.slider("WACC (%)", 5.0, 25.0, 0.0, 0.5,
                                help="0 = automático") / 100
        custom_g1 = st.slider("Crescimento Fase 1 (%)", -10.0, 40.0, 0.0, 0.5,
                              help="0 = automático") / 100

        wacc_param = custom_wacc if custom_wacc > 0 else None
        g1_param = custom_g1 if custom_g1 != 0 else None

        st.markdown("---")
        analisar_btn = st.button("🔍 Analisar", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Powered by Claude AI + yfinance")
        st.caption("⚠️ Não é recomendação de investimento.")

    # ─── Main Content ─────────────────────────────────────────────────────────
    st.title("📈 Análise Financeira Completa")

    if analisar_btn:
        st.session_state["last_ticker"] = ticker_input
        for key in ["data", "metrics", "dcf", "health", "fcf_hist",
                    "ai_biz", "ai_macro", "ai_comp", "comp_metrics"]:
            st.session_state.pop(key, None)

    if "last_ticker" not in st.session_state:
        st.info("👈 Selecione uma empresa no screener ou digite o ticker.")
        st.stop()

    ticker = st.session_state["last_ticker"]

    # ─── Carregamento de dados ────────────────────────────────────────────────
    if "metrics" not in st.session_state:
        with st.spinner(f"Buscando dados de **{ticker}**..."):
            try:
                data = get_company_data(ticker)
                metrics = extract_key_metrics(data)
                fcf_hist = get_historical_fcf(data)
                st.session_state["data"] = data
                st.session_state["metrics"] = metrics
                st.session_state["fcf_hist"] = fcf_hist
            except Exception as e:
                st.error(f"Erro ao buscar dados: {e}")
                st.stop()

    metrics = st.session_state["metrics"]
    data = st.session_state["data"]
    fcf_hist = st.session_state["fcf_hist"]
    moeda_symbol = "R$" if metrics.get("moeda") == "BRL" else "US$"

    # ─── Header da empresa ────────────────────────────────────────────────────
    nome = metrics.get("nome", ticker)
    setor = metrics.get("setor", "N/D")
    industria = metrics.get("industria", "N/D")
    preco = metrics.get("preco_atual")

    col_h1, col_h2, col_h3, col_h4 = st.columns([3, 1, 1, 1])
    with col_h1:
        st.markdown(f"## {nome}")
        st.caption(f"**{normalize_ticker(ticker)}** | {setor} | {industria} | {metrics.get('pais', 'N/D')}")
    with col_h2:
        st.metric("Preço Atual", f"{moeda_symbol} {preco:.2f}" if preco else "N/D")
    with col_h3:
        mc = metrics.get("market_cap")
        st.metric("Market Cap", f"{moeda_symbol} {mc/1e9:.1f}B" if mc else "N/D")
    with col_h4:
        dy = metrics.get("dividend_yield")
        st.metric("Dividend Yield", f"{dy*100:.1f}%" if dy else "N/D")

    st.markdown("---")

    # ─── TABS ─────────────────────────────────────────────────────────────────
    tab_dcf, tab_saude, tab_negocio, tab_macro, tab_concorrentes = st.tabs([
        "💰 Valuation DCF",
        "🏥 Saúde Financeira",
        "🏢 Modelo de Negócios",
        "🌍 Análise Macro",
        "⚔️ Concorrentes",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: VALUATION DCF
    # ══════════════════════════════════════════════════════════════════════
    with tab_dcf:
        if "dcf" not in st.session_state:
            with st.spinner("Calculando DCF..."):
                dcf = run_dcf(metrics, fcf_hist,
                              custom_wacc=wacc_param,
                              custom_growth_phase1=g1_param)
                st.session_state["dcf"] = dcf
        dcf = st.session_state["dcf"]

        if not dcf.get("aplicavel"):
            st.warning("⚠️ DCF não aplicável: " + " | ".join(dcf.get("erro", [])))
        else:
            col1, col2, col3, col4 = st.columns(4)
            upside = dcf.get("upside_pct", 0)

            with col1:
                st.metric("Valor Intrínseco", f"{moeda_symbol} {dcf['valor_intrinseco']:.2f}")
            with col2:
                st.metric("c/ Margem Segurança (20%)", f"{moeda_symbol} {dcf['valor_com_margem']:.2f}")
            with col3:
                delta_str = f"{upside:.1f}%"
                st.metric("Upside / Downside", delta_str, delta=delta_str)
            with col4:
                st.metric("WACC utilizado", f"{dcf['wacc']:.1%}")

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Parâmetros do Modelo**")
                params_df = pd.DataFrame([
                    {"Parâmetro": "FCF Base", "Valor": format_value(dcf["fcf_base"], "moeda_bilhoes", moeda_symbol)},
                    {"Parâmetro": "Crescimento Fase 1 (5 anos)", "Valor": f"{dcf['growth_phase1']:.1%}"},
                    {"Parâmetro": "Crescimento Fase 2 (5 anos)", "Valor": f"{dcf['growth_phase2']:.1%}"},
                    {"Parâmetro": "Crescimento Terminal", "Valor": f"{dcf['terminal_growth']:.1%}"},
                    {"Parâmetro": "WACC", "Valor": f"{dcf['wacc']:.1%}"},
                    {"Parâmetro": "Beta Usado", "Valor": f"{dcf['beta_usado']:.2f}"},
                    {"Parâmetro": "% Valor Terminal", "Valor": f"{dcf['pct_terminal']:.1f}%"},
                ])
                st.dataframe(params_df, use_container_width=True, hide_index=True)

            with col_b:
                projecoes = dcf.get("projecoes")
                if projecoes is not None and not projecoes.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=projecoes["ano"],
                        y=projecoes["fcf_projetado"] / 1e9,
                        marker_color=["#4CAF50" if f == "Crescimento" else "#2196F3"
                                      for f in projecoes["fase"]],
                        name="FCF Projetado",
                    ))
                    fig.add_trace(go.Scatter(
                        x=projecoes["ano"],
                        y=projecoes["pv_fcf"] / 1e9,
                        mode="lines+markers",
                        name="PV do FCF",
                        line=dict(color="#FF9800", width=2),
                    ))
                    fig.update_layout(
                        title="Projeção de FCF (10 anos)",
                        xaxis_title="Ano",
                        yaxis_title=f"Valor ({moeda_symbol} bilhões)",
                        template="plotly_dark",
                        height=300,
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Análise de Sensibilidade — Valor Intrínseco**")
            with st.spinner("Calculando sensibilidade..."):
                sens_df = sensitivity_analysis(metrics, fcf_hist, dcf)

            if not sens_df.empty:
                fig_heat = px.imshow(
                    sens_df.astype(float),
                    text_auto=".2f",
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    title=f"Sensibilidade (linhas = crescimento fase 1 | colunas = WACC) | Preço atual: {moeda_symbol} {preco:.2f}",
                )
                fig_heat.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_heat, use_container_width=True)

            with st.expander("Ver projeções detalhadas ano a ano"):
                proj_display = projecoes.copy()
                proj_display["fcf_projetado"] = proj_display["fcf_projetado"].apply(
                    lambda x: f"{moeda_symbol} {x/1e9:.2f}B")
                proj_display["pv_fcf"] = proj_display["pv_fcf"].apply(
                    lambda x: f"{moeda_symbol} {x/1e9:.2f}B")
                proj_display["taxa_crescimento"] = proj_display["taxa_crescimento"].apply(
                    lambda x: f"{x:.1%}")
                proj_display.columns = ["Ano", "Fase", "FCF Projetado", "PV do FCF", "Taxa Crescimento"]
                st.dataframe(proj_display, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: SAÚDE FINANCEIRA
    # ══════════════════════════════════════════════════════════════════════
    with tab_saude:
        if "health" not in st.session_state:
            with st.spinner("Analisando saúde financeira..."):
                health = analyze_financial_health(metrics)
                st.session_state["health"] = health
        health = st.session_state["health"]

        col_sc1, col_sc2, col_sc3 = st.columns([1, 2, 2])
        with col_sc1:
            score = health["score_geral"]
            cor_map = {"Excelente": "#4CAF50", "Bom": "#8BC34A", "Regular": "#FF9800", "Crítico": "#f44336"}
            cor = cor_map.get(health["classificacao"], "#FF9800")
            st.markdown(f"""
<div style="text-align:center; padding:20px; background:#1e2130; border-radius:12px; border: 3px solid {cor}">
    <div style="font-size:3rem; font-weight:bold; color:{cor}">{score}</div>
    <div style="font-size:1.1rem; color:{cor}">{health['cor']} {health['classificacao']}</div>
    <div style="color:#888; font-size:0.9rem">Score Geral (0-100)</div>
</div>
""", unsafe_allow_html=True)

        with col_sc2:
            cats = health["scores_categoria"]
            fig_radar = go.Figure(go.Scatterpolar(
                r=list(cats.values()),
                theta=list(cats.keys()),
                fill='toself',
                fillcolor='rgba(76, 175, 80, 0.2)',
                line=dict(color='#4CAF50'),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                title="Score por Categoria",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_sc3:
            fig_bar = go.Figure(go.Bar(
                x=list(cats.values()),
                y=list(cats.keys()),
                orientation='h',
                marker_color=[
                    "#4CAF50" if v >= 70 else "#FF9800" if v >= 45 else "#f44336"
                    for v in cats.values()
                ],
                text=[f"{v:.0f}" for v in cats.values()],
                textposition='outside',
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                height=300,
                xaxis=dict(range=[0, 110]),
                title="Score por Categoria",
                margin=dict(l=20, r=30, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        indicadores = health["indicadores"]
        categorias_unicas = list(dict.fromkeys(v["categoria"] for v in indicadores.values()))

        for cat in categorias_unicas:
            st.markdown(f"**{cat}**")
            inds_cat = {k: v for k, v in indicadores.items() if v["categoria"] == cat}
            cols = st.columns(len(inds_cat))

            for col, (key, ind) in zip(cols, inds_cat.items()):
                score_ind = ind["score"]
                val_fmt = format_value(ind["valor"], ind["formato"])
                score_colors = {1: "#f44336", 2: "#FF5722", 3: "#FF9800", 4: "#8BC34A", 5: "#4CAF50"}
                stars = "⭐" * score_ind
                bar_color = score_colors.get(score_ind, "#FF9800")

                with col:
                    st.markdown(f"""
<div style="background:#1e2130; border-radius:8px; padding:12px; border-left:4px solid {bar_color}; text-align:center">
    <div style="color:#888; font-size:0.75rem">{ind['label']}</div>
    <div style="font-size:1.3rem; font-weight:bold; color:{bar_color}">{val_fmt}</div>
    <div style="font-size:0.8rem">{stars}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("")

        st.markdown("---")
        st.markdown("**Histórico de Preço (5 anos)**")
        hist = data.get("history")
        if hist is not None and not hist.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=hist.index, y=hist["Close"],
                fill='tozeroy',
                fillcolor='rgba(76, 175, 80, 0.1)',
                line=dict(color='#4CAF50', width=1.5),
                name="Preço",
            ))
            fig_price.update_layout(
                template="plotly_dark",
                height=300,
                xaxis_title="Data",
                yaxis_title=f"Preço ({moeda_symbol})",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_price, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: MODELO DE NEGÓCIOS (AI)
    # ══════════════════════════════════════════════════════════════════════
    with tab_negocio:
        if not run_ai:
            st.info("Ative a opção **Análise por IA** na barra lateral.")
        else:
            if "dcf" not in st.session_state:
                st.warning("Execute o DCF primeiro (aba Valuation DCF).")
            elif "health" not in st.session_state:
                st.warning("Execute a análise de saúde primeiro.")
            else:
                if "ai_biz" not in st.session_state:
                    with st.spinner("Claude está analisando o modelo de negócios..."):
                        try:
                            ai_biz = analyze_business_model(
                                metrics,
                                st.session_state["dcf"],
                                st.session_state["health"]
                            )
                            st.session_state["ai_biz"] = ai_biz
                        except Exception as e:
                            st.error(f"Erro na análise por IA: {e}")
                            st.stop()
                ai_biz = st.session_state["ai_biz"]

                if "erro" in ai_biz:
                    st.error(f"Erro: {ai_biz['erro']}")
                else:
                    st.markdown("### Modelo de Negócios")
                    modelo = ai_biz.get("modelo_de_negocios", {})
                    st.markdown(modelo.get("resumo", ""))

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown("**Fontes de Receita**")
                        for fonte in modelo.get("fontes_de_receita", []):
                            st.markdown(f"- {fonte}")
                    with col_m2:
                        st.markdown("**Vantagens Competitivas (Moat)**")
                        for v in modelo.get("vantagens_competitivas", []):
                            st.markdown(f"- {v}")

                    st.markdown("---")

                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown("### ✅ Pontos Positivos")
                        for p in ai_biz.get("pontos_positivos", []):
                            st.markdown(f"""
<div style="background:#1a2a1a; border-left:4px solid #4CAF50; padding:10px; border-radius:6px; margin:6px 0">
    <strong style="color:#4CAF50">{p.get('titulo', '')}</strong><br>
    <span style="color:#ccc; font-size:0.9rem">{p.get('descricao', '')}</span>
</div>
""", unsafe_allow_html=True)

                    with col_neg:
                        st.markdown("### ⚠️ Pontos Negativos / Riscos")
                        for n in ai_biz.get("pontos_negativos", []):
                            st.markdown(f"""
<div style="background:#2a1a1a; border-left:4px solid #f44336; padding:10px; border-radius:6px; margin:6px 0">
    <strong style="color:#f44336">{n.get('titulo', '')}</strong><br>
    <span style="color:#ccc; font-size:0.9rem">{n.get('descricao', '')}</span>
</div>
""", unsafe_allow_html=True)

                    st.markdown("---")
                    veredicto = ai_biz.get("veredicto_investimento", {})
                    class_map = {
                        "Compra Forte": "#4CAF50", "Compra": "#8BC34A",
                        "Neutro": "#FF9800", "Venda": "#FF5722", "Venda Forte": "#f44336",
                    }
                    verd_class = veredicto.get("classificacao", "Neutro")
                    verd_color = class_map.get(verd_class, "#FF9800")

                    st.markdown(f"""
<div style="background:#1e2130; border-radius:12px; padding:20px; border:2px solid {verd_color}; text-align:center; margin:10px 0">
    <div style="font-size:1.8rem; font-weight:bold; color:{verd_color}">{verd_class}</div>
    <div style="color:#888; margin:8px 0">Horizonte: {veredicto.get('horizonte_recomendado', 'N/D')}</div>
    <div style="color:#ccc">{veredicto.get('justificativa', '')}</div>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: ANÁLISE MACRO
    # ══════════════════════════════════════════════════════════════════════
    with tab_macro:
        if not run_ai:
            st.info("Ative a opção **Análise por IA** na barra lateral.")
        else:
            if "ai_macro" not in st.session_state:
                with st.spinner("Claude está analisando o setor..."):
                    try:
                        ai_macro = analyze_macro_sector(metrics)
                        st.session_state["ai_macro"] = ai_macro
                    except Exception as e:
                        st.error(f"Erro: {e}")
                        st.stop()
            ai_macro = st.session_state["ai_macro"]

            if "erro" in ai_macro:
                st.error(f"Erro: {ai_macro['erro']}")
            else:
                panorama = ai_macro.get("panorama_setor", {})
                st.markdown(f"### Panorama do Setor — {metrics.get('setor', 'N/D')}")

                ciclo = panorama.get("ciclo_atual", "N/D")
                ciclo_colors = {"Expansão": "#4CAF50", "Estabilidade": "#8BC34A",
                                "Contração": "#f44336", "Turnaround": "#FF9800"}
                ciclo_color = ciclo_colors.get(ciclo, "#FF9800")

                col_pan1, col_pan2 = st.columns([3, 1])
                with col_pan1:
                    st.markdown(panorama.get("descricao", ""))
                with col_pan2:
                    st.markdown(f"""
<div style="background:#1e2130; border-radius:10px; padding:16px; text-align:center; border:2px solid {ciclo_color}">
    <div style="color:#888; font-size:0.8rem">Ciclo Atual</div>
    <div style="color:{ciclo_color}; font-size:1.4rem; font-weight:bold">{ciclo}</div>
    <div style="color:#aaa; font-size:0.8rem; margin-top:6px">{panorama.get('justificativa_ciclo', '')}</div>
</div>
""", unsafe_allow_html=True)

                st.markdown("---")
                col_drv, col_rsk = st.columns(2)

                with col_drv:
                    st.markdown("### Drivers de Crescimento")
                    for d in ai_macro.get("drivers_crescimento", []):
                        impacto = d.get("impacto", "Médio")
                        imp_color = {"Alto": "#4CAF50", "Médio": "#FF9800", "Baixo": "#888"}.get(impacto, "#888")
                        st.markdown(f"""
<div style="background:#1e2130; border-left:4px solid {imp_color}; padding:10px; border-radius:6px; margin:6px 0">
    <strong>{d.get('driver', '')}</strong>
    <span style="color:{imp_color}; font-size:0.8rem; margin-left:8px">● {impacto}</span><br>
    <span style="color:#aaa; font-size:0.9rem">{d.get('descricao', '')}</span>
</div>
""", unsafe_allow_html=True)

                with col_rsk:
                    st.markdown("### Riscos Macro")
                    for r in ai_macro.get("riscos_macro", []):
                        prob = r.get("probabilidade", "Média")
                        prob_color = {"Alta": "#f44336", "Média": "#FF9800", "Baixa": "#4CAF50"}.get(prob, "#FF9800")
                        st.markdown(f"""
<div style="background:#1e2130; border-left:4px solid {prob_color}; padding:10px; border-radius:6px; margin:6px 0">
    <strong>{r.get('risco', '')}</strong>
    <span style="color:{prob_color}; font-size:0.8rem; margin-left:8px">● Prob. {prob}</span><br>
    <span style="color:#aaa; font-size:0.9rem">{r.get('descricao', '')}</span>
</div>
""", unsafe_allow_html=True)

                st.markdown("---")
                col_tend, col_reg = st.columns(2)

                with col_tend:
                    st.markdown("### Tendências Estruturais")
                    for t in ai_macro.get("tendencias_estruturais", []):
                        st.markdown(f"→ {t}")

                with col_reg:
                    reg = ai_macro.get("regulatorio", {})
                    amb = reg.get("ambiente", "Neutro")
                    amb_color = {"Favorável": "#4CAF50", "Neutro": "#FF9800", "Desfavorável": "#f44336"}.get(amb, "#FF9800")
                    perspectiva = ai_macro.get("perspectiva_12_meses", "Neutro")
                    st.markdown("### Ambiente Regulatório")
                    st.markdown(f"**{amb}** — {reg.get('descricao', '')}")
                    st.markdown("### Perspectiva 12 Meses")
                    st.markdown(f"""
<div style="color:{amb_color}; font-size:1.2rem; font-weight:bold">{perspectiva}</div>
<div style="color:#aaa">{ai_macro.get('justificativa_perspectiva', '')}</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5: CONCORRENTES
    # ══════════════════════════════════════════════════════════════════════
    with tab_concorrentes:
        if not run_competitors:
            st.info("Ative a opção **Análise de concorrentes** na barra lateral.")
        else:
            if "comp_metrics" not in st.session_state:
                with st.spinner("Buscando dados dos concorrentes..."):
                    competitors = get_competitors(data)
                    comp_metrics_list = []
                    for comp_ticker in competitors:
                        try:
                            comp_data = get_company_data(comp_ticker)
                            comp_m = extract_key_metrics(comp_data)
                            comp_metrics_list.append(comp_m)
                        except Exception:
                            pass
                    st.session_state["comp_metrics"] = comp_metrics_list
                    st.session_state["competitors"] = competitors

            comp_metrics_list = st.session_state["comp_metrics"]
            competitors = st.session_state.get("competitors", [])

            if not comp_metrics_list:
                st.warning("Não foram encontrados dados de concorrentes para este setor.")
            else:
                st.markdown("### Comparação de Múltiplos")
                comp_df = compare_competitors(metrics, comp_metrics_list)
                st.dataframe(comp_df, use_container_width=True)

                all_companies = [metrics] + comp_metrics_list
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    nomes = [m.get("nome", m.get("ticker", "N/D"))[:15] for m in all_companies]
                    pls = [m.get("p_l") for m in all_companies]
                    fig_pl = go.Figure(go.Bar(
                        x=nomes, y=pls,
                        marker_color=["#4CAF50"] + ["#2196F3"] * len(comp_metrics_list),
                        text=[f"{v:.1f}x" if v else "N/D" for v in pls],
                        textposition='outside',
                    ))
                    fig_pl.update_layout(title="P/L Comparativo", template="plotly_dark", height=300)
                    st.plotly_chart(fig_pl, use_container_width=True)

                with col_chart2:
                    roes = [m.get("roe") for m in all_companies]
                    fig_roe = go.Figure(go.Bar(
                        x=nomes, y=[(r * 100 if r else None) for r in roes],
                        marker_color=["#4CAF50"] + ["#2196F3"] * len(comp_metrics_list),
                        text=[f"{r*100:.1f}%" if r else "N/D" for r in roes],
                        textposition='outside',
                    ))
                    fig_roe.update_layout(title="ROE Comparativo (%)", template="plotly_dark", height=300)
                    st.plotly_chart(fig_roe, use_container_width=True)

                if run_ai:
                    if "ai_comp" not in st.session_state:
                        with st.spinner("Claude está comparando com concorrentes..."):
                            try:
                                ai_comp = analyze_competitors_qualitative(
                                    metrics,
                                    comp_metrics_list,
                                    comp_df.to_string(),
                                )
                                st.session_state["ai_comp"] = ai_comp
                            except Exception as e:
                                st.error(f"Erro: {e}")
                                ai_comp = {}
                                st.session_state["ai_comp"] = ai_comp

                    ai_comp = st.session_state.get("ai_comp", {})

                    if ai_comp and "erro" not in ai_comp:
                        st.markdown("---")
                        st.markdown("### Análise Competitiva (IA)")

                        pos = ai_comp.get("posicao_competitiva", {})
                        st.markdown(f"**Posição Competitiva: {pos.get('classificacao', 'N/D')}**")
                        st.markdown(pos.get("descricao", ""))

                        col_v, col_d = st.columns(2)
                        with col_v:
                            st.markdown("**Vantagens vs Concorrentes**")
                            for v in ai_comp.get("vantagens_vs_concorrentes", []):
                                st.markdown(f"✅ {v}")
                        with col_d:
                            st.markdown("**Desvantagens vs Concorrentes**")
                            for d in ai_comp.get("desvantagens_vs_concorrentes", []):
                                st.markdown(f"⚠️ {d}")

                        melhor = ai_comp.get("melhor_alternativa", {})
                        if melhor:
                            st.markdown("---")
                            st.markdown(f"**Melhor opção no setor: {melhor.get('empresa', 'N/D')}**")
                            st.markdown(melhor.get("motivo", ""))

                        resumo = ai_comp.get("resumo_comparativo", "")
                        if resumo:
                            st.markdown("---")
                            st.info(resumo)
