"""
================================================================================
IMPACTO DOS CICLOS ELEITORAIS NO MERCADO DE CAPITAIS BRASILEIRO (2002–2022)
Análise de Estudo de Evento — Versão Completa Acadêmica
================================================================================

OBJETIVO DA PESQUISA:
  Quantificar o "risco político" no mercado brasileiro via Mapa de Risco Setorial.
  
ESCOPO:
  - Período: 2002-2022 (6 Eleições Presidenciais)
  - Universo: ~700 tickers (incluindo IPOs e empresas deslistadas)
  - Classificação: 11 setores oficiais da B3
  
MÉTODOLOGIA TÉCNICA:
  - Modelo de Mercado (CAPM): R_i = α + β·R_m + ε
  - Fama-French 3 Fatores (BR): R_i-Rf = α + β·MktRf + s·SMB + h·HML + ε
  - Janela de Estimação: [-252, -30] dias úteis (MacKinlay, 1997)
  - Ponderação: Equal-Weighted (EW) e Volume-Weighted (VW)
  - Pré-processamento: Winsorização (1%/99%) e Filtro de Liquidez (80%)
  - Covariância: Erros padrão Newey-West (HAC) para autocorrelação
  - Análise DiD: Estatais vs. Privadas em setores sensíveis

BENCHMARKS:
  - Ibovespa (^BVSP) - benchmark de mercado
  - SELIC - custo de oportunidade
  - IPCA - retornos reais
  - Fatores NEFIN-USP para Fama-French

TESTES ESTATÍSTICOS:
  - Teste t simples (longitudinal)
  - Teste t corrigido (Silva et al., 2015)
  - Teste BMP (Boehmer, Musumeci & Poulsen, 1991)
  - OLS com erros HAC (Newey-West)

TESTES DE ROBUSTEZ:
  - Janelas alternativas: [-60,-1], [-10,+10], [-20,+20]
  - Cenário sem crises (exclui 2008/2020)
  - Teste placebo: 1000 pseudo-eventos
  - DiD: Estatais vs. Privadas
  - Sharpe Ratio comparativo

REFERÊNCIAS:
  - Silva et al. (2015) – Event Study / Retornos anormais IPO
  - Nordhaus (1975) – Political Business Cycles
  - Boehmer, Musumeci & Poulsen (1991) – Event-induced variance
  - MacKinlay (1997) – Event Studies in Economics and Finance
  - Fama & French (1993) – Common risk factors in stock returns
  - Newey & West (1987) – HAC covariance estimation

REQUISITOS:
    pip install pandas numpy openpyxl yfinance statsmodels scipy matplotlib seaborn

ENTRADA:
    resultados_analise_b3_com_tickers.xlsx → aba "LISTA FINAL (Cont+IPOs-Canc)"

SAÍDA (pasta ./output_v3/):
    analise_ciclos_eleitorais_v3.xlsx   – resultados consolidados
    resultados_consolidados_v3.csv      – CSV para replicação
    resultados_ff3_v3.csv               – resultados Fama-French 3 fatores
    tabela_sobrevivencia.csv            – N por setor/ano
    diagnostico_download.csv            – log de tickers obtidos/faltantes
    heatmap_*.png                       – mapas de calor dos CARs
    evolucao_temporal_cars.png          – evolução temporal
    volatilidade_comparativa.png        – eleitoral vs. não-eleitoral
    placebo_test.png / _results.csv     – teste placebo
    did_regulados_vs_nao.png / .csv     – Difference-in-Differences
    sharpe_eleitoral.csv                – Sharpe Ratio comparativo
    metodologia_limitacoes.md           – texto pronto para artigo

================================================================================
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ElectoralCycles")

ARQUIVO_ENTRADA = "resultados_analise_b3_com_tickers.xlsx"
SHEET_NAME = "LISTA FINAL (Cont+IPOs-Canc)"
OUTPUT_DIR = "./output_v3"

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# ---- Datas eleitorais (1º e 2º turnos) -----------------------------------
DATAS_PRIMEIRO_TURNO = {
    2002: pd.Timestamp("2002-10-06"),
    2006: pd.Timestamp("2006-10-01"),
    2010: pd.Timestamp("2010-10-03"),
    2014: pd.Timestamp("2014-10-05"),
    2018: pd.Timestamp("2018-10-07"),
    2022: pd.Timestamp("2022-10-02"),
}
DATAS_SEGUNDO_TURNO = {
    2002: pd.Timestamp("2002-10-27"),
    2006: pd.Timestamp("2006-10-29"),
    2010: pd.Timestamp("2010-10-31"),
    2014: pd.Timestamp("2014-10-26"),
    2018: pd.Timestamp("2018-10-28"),
    2022: pd.Timestamp("2022-10-30"),
}
ANOS_ELEITORAIS = sorted(DATAS_PRIMEIRO_TURNO.keys())

# ---- Setores B3 ----------------------------------------------------------
SETORES_B3 = [
    "Bens Industriais", "Comunicações", "Construção e Transporte",
    "Consumo Cíclico", "Consumo Não Cíclico", "Financeiro",
    "Materiais Básicos", "Outros", "Petróleo, Gás e Biocombustíveis",
    "Saúde", "Utilidade Pública",
]

# ---- Janela de estimação (MacKinlay, 1997) --------------------------------
ESTIMACAO_INICIO_DU = -252    # dias úteis antes do 1º turno
ESTIMACAO_FIM_DU = -30        # dias úteis antes do 1º turno

# ---- Janelas de evento (dias úteis relativos ao 1º turno) ----------------
JANELAS_EVENTO_1T = {
    "antecipacao_45":  (-45, -1),     # Efeito propaganda / pricing-in
    "antecipacao_60":  (-60, -1),     # Robustez: janela mais larga
    "reacao_curta_1t": (-5, +5),      # Reação imediata ao 1º turno
    "reacao_media_1t": (-10, +10),    # Robustez
    "reacao_ampla_1t": (-20, +20),    # Robustez: persistência
}

# ---- Janelas relativas ao 2º turno --------------------------------------
JANELAS_EVENTO_2T = {
    "reacao_curta_2t": (-5, +5),      # Reação imediata ao 2º turno
    "reacao_media_2t": (-10, +10),    # Robustez: digestão do resultado
}

JANELA_ENTRE_TURNOS = True    # flag para ativar

# ---- Janelas semestrais (ciclo interno) ----------------------------------
JANELAS_CICLO = True          # 1º sem (expectativa) vs 2º sem (disputa)

# ---- Janela estendida ----------------------------------------------------
JANELA_ESTENDIDA = True       # últimos 6 meses do ano eleitoral

JANELAS_ESPELHO = True

MIN_PREGOES_PCT = 0.80        # Mínimo 80% de pregões no ano para inclusão
MIN_EMPRESAS_SETOR = 5        # Corte: mínimo N empresas por setor/ano
ANOS_CRISE = [2008, 2014, 2020]    
N_PLACEBO_EVENTS = 1000             

SELIC_ANUAL = {
    2002: 0.1911, 2006: 0.1513, 2010: 0.0975,
    2014: 0.1115, 2018: 0.0640, 2022: 0.1275,
}
IPCA_ANUAL = {
    2002: 0.1253, 2006: 0.0314, 2010: 0.0591,
    2014: 0.0641, 2018: 0.0375, 2022: 0.0562,
}

# Selic diária (proxy CDI) para cálculo de Sharpe
SELIC_DIARIA = {k: (1 + v) ** (1/252) - 1 for k, v in SELIC_ANUAL.items()}

# ---- Mapeamento de tickers (correções históricas) ------------------------
TICKER_MAPPING = {
    "VVAR3": "BHIA3", "BTOW3": "AMER3", "LAME4": "LAME3", "PCAR4": "PCAR3",
    "KROT3": "COGN3", "ESTC3": "YDUQ3", "RAIL3": "RUMO3",
    "BVMF3": "B3SA3", "CTIP3": "B3SA3",
    "BRIN3": "BRML3", "BRML3": "ALOS3", "SMLE3": "SMFT3",
    "LINX3": "STNE3", "VIVT4": "VIVT3", "TIMP3": "TIMS3",
    "QGEP3": "BRAV3", "GNDI3": "HAPV3", "FIBR3": "SUZB3",
}

# ============================================================================
# ETAPA 1: INGESTÃO DE DADOS
# ============================================================================

def carregar_lista_empresas(caminho: str) -> pd.DataFrame:
    """
    Carrega lista de empresas, aplica mapeamento De-Para,
    e lê coluna ESTATAL direto da planilha.
    """
    log.info("=" * 70)
    log.info("ETAPA 1: INGESTÃO DE DADOS")
    log.info("=" * 70)

    df = pd.read_excel(caminho, sheet_name="Sheet1")
    df = df.dropna(subset=["TICKER", "SETOR_B3"])

    # Parse de datas
    df["DT_REG"] = pd.to_datetime(df["DT_REG"], errors="coerce")
    
    # Tratamento de DT_CANCEL se existir
    if "DT_CANCEL" in df.columns:
        df["DT_CANCEL"] = pd.to_datetime(df["DT_CANCEL"], errors="coerce")

    # Coluna ESTATAL: converte "Sim"/"Não" para booleano
    if "ESTATAL" in df.columns:
        df["ESTATAL"] = df["ESTATAL"].str.strip().str.upper().eq("SIM")
    else:
        df["ESTATAL"] = False

    # Ticker original → mapeado → yfinance
    df["TICKER_ORIGINAL"] = df["TICKER"].str.strip()
    df["TICKER_MAPEADO"] = df["TICKER_ORIGINAL"].map(TICKER_MAPPING).fillna(
        df["TICKER_ORIGINAL"]
    )
    df["TICKER_YF"] = df["TICKER_MAPEADO"] + ".SA"

    n_map = (df["TICKER_ORIGINAL"] != df["TICKER_MAPEADO"]).sum()
    n_est = df["ESTATAL"].sum()

    log.info("  → %d empresas | %d setores | %d remapeados | %d estatais",
             len(df), df["SETOR_B3"].nunique(), n_map, n_est)

    # Log das estatais para conferência
    for _, r in df[df["ESTATAL"]].iterrows():
        log.info("    [ESTATAL] %s — %s (%s)",
                 r["TICKER_ORIGINAL"], str(r.get("DENOM_SOCIAL", ""))[:45], r["SETOR_B3"])

    return df


# ============================================================================
# ETAPA 2: DOWNLOAD DE PREÇOS
# ============================================================================

# Forma segura: defina como variável de ambiente (recomendado)
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

# Fallback temporário só para teste local (remova em produção!)
if not ALPHA_VANTAGE_KEY:
    ALPHA_VANTAGE_KEY = 'SSU8PQU94ONCSLAF'  # Configure como env var para segurança!
    print("AVISO: Usando key hardcoded – configure como env var para segurança!")


def baixar_precos_yfinance(tickers: list, start="2001-01-01", end="2023-12-31"):
    """Baixa preços ajustados e volumes via yfinance (primário) + Alpha Vantage (fallback)."""
    import yfinance as yf
    from alpha_vantage.timeseries import TimeSeries

    log.info("Baixando preços de %d tickers (primário: yfinance, fallback: Alpha Vantage)...", len(tickers))
    all_close = {}
    all_volume = {}
    diagnostico_list = []

    blocos = [tickers[i:i+50] for i in range(0, len(tickers), 50)]

    # Cliente Alpha Vantage (inicializa só se key válida)
    ts_av = None
    if ALPHA_VANTAGE_KEY:
        try:
            ts_av = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        except Exception as e:
            log.warning("Falha ao inicializar Alpha Vantage: %s", e)

    for idx, bloco in enumerate(blocos):
        log.info("  Bloco %d/%d (%d tickers)", idx+1, len(blocos), len(bloco))
        try:
            data = yf.download(bloco, start=start, end=end, auto_adjust=True,
                               progress=False, threads=True)
            if data.empty:
                raise ValueError("Dados vazios no yfinance")

            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                volume = data["Volume"]
            else:
                close = data[["Close"]]
                close.columns = bloco
                volume = data[["Volume"]]
                volume.columns = bloco

            for col in close.columns:
                if close[col].notna().sum() > 0:
                    all_close[col] = close[col]
                    all_volume[col] = volume[col] if col in volume.columns else pd.Series(dtype=float)
                    diagnostico_list.append({
                        "ticker_yf": col, "status": "ok", "fonte": "yfinance", "motivo": ""
                    })
                else:
                    diagnostico_list.append({
                        "ticker_yf": col, "status": "falha", "fonte": "", "motivo": "no_data_yf"
                    })

            for t in bloco:
                if t not in close.columns:
                    diagnostico_list.append({
                        "ticker_yf": t, "status": "falha", "fonte": "", "motivo": "not_found_yf"
                    })

        except Exception as e:
            log.warning("  Erro yfinance bloco %d: %s → Marcando para fallback", idx+1, e)
            for t in bloco:
                diagnostico_list.append({
                    "ticker_yf": t, "status": "falha", "fonte": "", "motivo": f"erro_yf: {str(e)}"
                })

        time.sleep(0.3)

    # Fallback Alpha Vantage para falhas
    falhas = [d for d in diagnostico_list if d["status"] == "falha"]
    if falhas and ts_av:
        log.info("  Fallback Alpha Vantage para %d tickers falhos...", len(falhas))
        for diag in falhas:
            t = diag["ticker_yf"]
            try:
                data_av, _ = ts_av.get_daily(symbol=t, outputsize='full')
                data_av = data_av.rename(columns={
                    '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                    '4. close': 'Close', '5. volume': 'Volume'
                })
                if data_av['Close'].notna().sum() > 0:
                    all_close[t] = data_av['Close']
                    all_volume[t] = data_av['Volume']
                    diag["status"] = "ok"
                    diag["fonte"] = "alpha_vantage"
                    diag["motivo"] = ""
                
                time.sleep(12)  # Rate limit free (5 calls/min)

            except Exception as av_e:
                log.warning("  Falha Alpha Vantage %s: %s", t, av_e)
                diag["motivo"] += f"; erro_av: {str(av_e)}"

    # Finaliza
    df_precos = pd.DataFrame(all_close)
    df_precos.index = pd.to_datetime(df_precos.index)
    df_volumes = pd.DataFrame(all_volume)
    df_volumes.index = pd.to_datetime(df_volumes.index)

    df_diag = pd.DataFrame(diagnostico_list)
    n_ok = (df_diag["status"] == "ok").sum()
    n_falha = (df_diag["status"] == "falha").sum()

    log.info("  → OK: %d | Falha: %d (%.1f%%)", n_ok, n_falha,
             100 * n_falha / max(len(tickers), 1))

    return df_precos, df_volumes, df_diag


def baixar_ibovespa(start="2000-01-02", end="2023-12-31"):
    import yfinance as yf
    log.info("Baixando Ibovespa ...")
    ibov = yf.download("^BVSP", start=start, end=end, auto_adjust=True, progress=False)
    serie = ibov["Close"].squeeze()
    serie.index = pd.to_datetime(serie.index)
    serie.name = "IBOV"
    log.info("  → %d obs", len(serie))
    return serie


def baixar_fatores_nefin():
    """
    Baixa fatores Fama-French brasileiros do NEFIN-USP (Versão CSV único).
    Retorna DataFrame com colunas: Mkt_Rf, SMB, HML, Rf.
    Se falhar, retorna None (script continua sem FF3).
    """
    log.info("Baixando fatores Fama-French do NEFIN-USP ...")
    
    url_csv = "https://nefin.com.br/resources/risk_factors/nefin_factors.csv"
    
    try:
        df = pd.read_csv(url_csv)

        # 1. Tratamento de Data
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
            df = df.set_index("date")
        else:
            if {"year", "month", "day"}.issubset(df.columns):
                 df["date"] = pd.to_datetime(df[["year", "month", "day"]])
                 df = df.set_index("date")
            else:
                log.warning("  Formato de data desconhecido no CSV do NEFIN.")
                return None

        # 2. Renomear colunas para o padrão do script
        rename_map = {
            "Rm_minus_Rf": "Mkt_Rf", 
            "Risk_Free": "Rf"
        }
        df = df.rename(columns=rename_map)

        # 3. Filtrar apenas as colunas necessárias
        cols_necessarias = ["Mkt_Rf", "SMB", "HML", "Rf"]
        
        if not set(cols_necessarias).issubset(df.columns):
            log.warning(f"  Colunas faltando no NEFIN. Encontradas: {df.columns.tolist()}")
            return None

        df_factors = df[cols_necessarias]

        log.info("  → Fatores NEFIN: %d obs (de %s a %s)",
                 len(df_factors), 
                 df_factors.index.min().strftime('%Y-%m'), 
                 df_factors.index.max().strftime('%Y-%m'))
        
        return df_factors

    except Exception as e:
        log.warning("  Falha ao baixar NEFIN: %s. FF3 desabilitado.", e)
        return None


# ============================================================================
# ETAPA 2b: FILTROS E PRÉ-PROCESSAMENTO
# ============================================================================

def aplicar_winsorizacao(df_ret, lower=0.01, upper=0.99):
    """
    Aplica winsorização nos retornos (tratamento de outliers).
    Substitui valores abaixo do percentil 'lower' e acima de 'upper'.
    """
    log.info("Aplicando winsorização (%.0f%%/%.0f%%)...", lower*100, upper*100)
    df_wins = df_ret.copy()
    for col in df_wins.columns:
        q_low = df_wins[col].quantile(lower)
        q_high = df_wins[col].quantile(upper)
        df_wins[col] = df_wins[col].clip(lower=q_low, upper=q_high)
    return df_wins


def aplicar_filtro_existencia(df_precos, df_empresas):
    """
    Invalida (NaN) precos anteriores a data de registro (DT_REG)
    e posteriores a data de cancelamento (DT_CANCEL) se existir.
    """
    log.info("Aplicando filtro de existência (DT_REG e DT_CANCEL)...")

    df = df_precos.copy()

    # Remove fuso horario para evitar erro de comparacao
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Dicionário: Ticker -> (Data Inicio, Data Fim)
    lookup = {}
    
    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        reg = pd.to_datetime(row["DT_REG"], errors='coerce')
        cancel = pd.to_datetime(row.get("DT_CANCEL"), errors='coerce') if "DT_CANCEL" in row else pd.NaT

        if tk in lookup:
            old_reg, old_cancel = lookup[tk]
            if pd.notna(reg) and (pd.isna(old_reg) or reg < old_reg):
                lookup[tk] = (reg, cancel if pd.notna(cancel) else old_cancel)
        else:
            lookup[tk] = (reg, cancel)

    total_cortes = 0

    for col in df.columns:
        if col in lookup:
            dt_inicio, dt_fim = lookup[col]

            if pd.notna(dt_inicio):
                mask_pre = df.index < dt_inicio
                if mask_pre.any():
                    qtd = df.loc[mask_pre, col].notna().sum()
                    if qtd > 0:
                        total_cortes += qtd
                        df.loc[mask_pre, col] = np.nan
            
            if pd.notna(dt_fim):
                mask_pos = df.index > dt_fim
                if mask_pos.any():
                    qtd = df.loc[mask_pos, col].notna().sum()
                    if qtd > 0:
                        total_cortes += qtd
                        df.loc[mask_pos, col] = np.nan

    log.info("  → Filtro aplicado. Observações removidas: %d", total_cortes)
    
    return df


def aplicar_filtro_liquidez(df_ret, ano, min_pct=MIN_PREGOES_PCT):
    """
    Filtra tickers que negociaram em menos de min_pct dos pregões no ano.
    
    Parâmetros:
        df_ret    : DataFrame de retornos (index=datas, columns=tickers)
        ano       : int, ano para filtrar
        min_pct   : float, fração mínima de pregões (default 0.80)
    
    Retorna:
        lista de tickers que passaram no filtro para aquele ano
    """
    mask_ano = df_ret.index.year == ano
    ret_ano = df_ret.loc[mask_ano]
    
    if ret_ano.empty:
        return []
    
    n_pregoes = mask_ano.sum()
    min_obs = int(n_pregoes * min_pct)
    
    obs_por_ticker = ret_ano.notna().sum()
    
    aprovados = obs_por_ticker[obs_por_ticker >= min_obs].index.tolist()
    return aprovados


# ============================================================================
# ETAPA 2c: CONSTRUÇÃO DE ÍNDICES SETORIAIS
# ============================================================================

def construir_indices_setoriais(df_precos, df_volumes, df_empresas, aplicar_wins=True):
    """
    Constrói índices setoriais EW e VW com filtro de liquidez.
    
    Etapas internas:
      1. Retornos log de todos os ativos
      2. Winsorização opcional (1%/99%)
      3. Para cada (setor, ano): aplica filtro de liquidez
      4. EW = média simples dos retornos dos tickers líquidos
      5. VW = média ponderada por volume financeiro (proxy market cap)
      6. Registra composição real (N após filtro)
    
    Retorna:
        (df_ret_ew, df_ret_vw, df_composicao)
    """
    log.info("=" * 70)
    log.info("ETAPA 2: ÍNDICES SETORIAIS (EW + VW) COM FILTRO DE LIQUIDEZ")
    log.info("=" * 70)

    # Retornos log
    ret = np.log(df_precos / df_precos.shift(1))
    
    # Winsorização opcional
    if aplicar_wins:
        ret = aplicar_winsorizacao(ret)
    
    # Volume financeiro = preço × volume de ações
    vol_fin = df_precos * df_volumes

    # Mapeamento ticker → setor
    t2s = {}
    for _, row in df_empresas.iterrows():
        t2s[row["TICKER_YF"]] = row["SETOR_B3"]

    # Listas para acumular resultados
    series_ew = {}
    series_vw = {}
    composicao = []

    for setor in SETORES_B3:
        cols_setor = [c for c in ret.columns if t2s.get(c) == setor]
        if not cols_setor:
            log.warning("  Sem tickers para '%s'", setor)
            continue

        ew_parts = []
        vw_parts = []

        for ano in range(2001, 2024):
            mask_ano = ret.index.year == ano
            if mask_ano.sum() == 0:
                continue

            # Filtro de liquidez
            aprovados_global = aplicar_filtro_liquidez(ret, ano)
            cols_ano = [c for c in cols_setor if c in aprovados_global]
            
            n_antes = len(cols_setor)
            n_depois = len(cols_ano)

            composicao.append({
                "setor": setor,
                "ano": ano,
                "n_tickers_setor": n_antes,
                "n_com_dados_liquidos": n_depois,
                "filtro_removeu": n_antes - n_depois,
            })

            if n_depois < MIN_EMPRESAS_SETOR:
                continue

            ret_ano = ret.loc[mask_ano, cols_ano]
            
            # EW: média simples
            ew_parts.append(ret_ano.mean(axis=1))

            # VW: ponderado por volume financeiro médio 20d
            vf_ano = vol_fin.loc[mask_ano, cols_ano].rolling(20, min_periods=5).mean()
            vf_sum = vf_ano.sum(axis=1)
            weights = vf_ano.div(vf_sum.replace(0, np.nan), axis=0)
            vw_parts.append((ret_ano * weights).sum(axis=1))

        # Concatena os pedaços anuais
        if ew_parts:
            series_ew[setor] = pd.concat(ew_parts).sort_index()
        if vw_parts:
            series_vw[setor] = pd.concat(vw_parts).sort_index()

        # Log resumo
        comp_setor = [c for c in composicao if c["setor"] == setor]
        anos_validos = sum(1 for c in comp_setor if c["n_com_dados_liquidos"] >= MIN_EMPRESAS_SETOR)
        log.info("  %s: %d tickers totais | %d/%d anos com ≥%d empresas líquidas",
                 setor, len(cols_setor), anos_validos, len(comp_setor), MIN_EMPRESAS_SETOR)

    df_ret_ew = pd.DataFrame(series_ew)
    df_ret_vw = pd.DataFrame(series_vw)
    df_comp = pd.DataFrame(composicao)

    log.info("  → EW: %d setores × %d datas", df_ret_ew.shape[1], len(df_ret_ew))
    log.info("  → VW: %d setores × %d datas", df_ret_vw.shape[1], len(df_ret_vw))

    return df_ret_ew, df_ret_vw, df_comp


def gerar_tabela_sobrevivencia(df_empresas, df_precos):
    """
    Tabela de cobertura: N esperado vs. N com dados por setor/ano.
    Considera DT_REG e DT_CANCEL para cálculo correto.
    """
    log.info("Gerando tabela de sobrevivência ...")
    
    t2s = {row["TICKER_YF"]: row["SETOR_B3"] for _, row in df_empresas.iterrows()}
    registros = []

    for ano in range(2001, 2024):
        mask_ano = df_precos.index.year == ano
        if mask_ano.sum() == 0:
            continue
        
        precos_ano = df_precos.loc[mask_ano]

        for setor in SETORES_B3:
            cols = [c for c in precos_ano.columns if t2s.get(c) == setor]
            
            # N com dados: tickers que têm ≥1 preço válido no ano
            n_dados = precos_ano[cols].notna().any().sum() if cols else 0

            # N esperado: empresas registradas até esse ano e não canceladas
            if "DT_CANCEL" in df_empresas.columns:
                n_esp = (
                    (df_empresas["SETOR_B3"] == setor) &
                    (df_empresas["DT_REG"].dt.year <= ano) &
                    ((df_empresas["DT_CANCEL"].isna()) | (df_empresas["DT_CANCEL"].dt.year >= ano))
                ).sum()
            else:
                n_esp = (
                    (df_empresas["SETOR_B3"] == setor) &
                    (df_empresas["DT_REG"].dt.year <= ano)
                ).sum()

            cobertura = round(100 * n_dados / max(n_esp, 1), 1)

            registros.append({
                "ano": ano,
                "setor": setor,
                "n_esperado": n_esp,
                "n_com_dados": n_dados,
                "cobertura_pct": cobertura,
            })

    df_sobrev = pd.DataFrame(registros)
    
    # Log resumo por ano eleitoral
    for ano in ANOS_ELEITORAIS:
        sub = df_sobrev[df_sobrev["ano"] == ano]
        if sub.empty:
            continue
        cob_media = sub["cobertura_pct"].mean()
        log.info("  %d: cobertura média %.1f%% (N esperado=%d, N com dados=%d)",
                 ano, cob_media, sub["n_esperado"].sum(), sub["n_com_dados"].sum())

    return df_sobrev


# ============================================================================
# ETAPA 3: JANELAS DE EVENTO
# ============================================================================

def offset_du(bdates, data_ref, offset):
    """
    Retorna a data de dia útil com offset a partir de data_ref.
    
    Parâmetros:
        bdates   : Series de datas úteis (Index do DataFrame)
        data_ref : Timestamp de referência
        offset   : int, número de dias úteis de deslocamento
    
    Retorna:
        Timestamp da data resultante
    """
    bdays = bdates.sort_values()
    pos = min(bdays.searchsorted(data_ref), len(bdays)-1)
    target = max(0, min(pos + offset, len(bdays)-1))
    return bdays[target]


def janela_estimacao(bdates, dt_evento):
    """
    Define a janela de estimação: [-252, -30] dias úteis antes do evento.
    Seguindo MacKinlay (1997) para evitar contaminação pré-evento.
    """
    return (offset_du(bdates, dt_evento, ESTIMACAO_INICIO_DU),
            offset_du(bdates, dt_evento, ESTIMACAO_FIM_DU))


def janelas_evento(bdates, ano):
    """
    Define todas as janelas de evento para um ano eleitoral.
    
    Retorna dicionário com:
      - Janelas de 1º turno (antecipação, reação)
      - Janelas de 2º turno (se houver)
      - Janelas semestrais (ciclo interno)
      - Janela estendida (últimos 6 meses)
    """
    dt1 = DATAS_PRIMEIRO_TURNO[ano]
    dt2 = DATAS_SEGUNDO_TURNO.get(ano)
    
    j = {}
    
    # Janelas de 1º turno
    for nome, (di, df_) in JANELAS_EVENTO_1T.items():
        j[nome] = (offset_du(bdates, dt1, di), offset_du(bdates, dt1, df_))
    
    # Janelas de 2º turno
    if dt2:
        for nome, (di, df_) in JANELAS_EVENTO_2T.items():
            j[nome] = (offset_du(bdates, dt2, di), offset_du(bdates, dt2, df_))
    
    # Janela entre turnos
    if JANELA_ENTRE_TURNOS and dt2:
        j["entre_turnos"] = (offset_du(bdates, dt1, 1), offset_du(bdates, dt2, -1))
    
    # Janelas semestrais (ciclo interno)
    if JANELAS_CICLO:
        j["ciclo_1sem"] = (pd.Timestamp(f"{ano}-01-02"), pd.Timestamp(f"{ano}-06-30"))
        j["ciclo_2sem"] = (pd.Timestamp(f"{ano}-07-01"), pd.Timestamp(f"{ano}-12-31"))
    
    # Janela estendida
    if JANELA_ESTENDIDA:
        j["estendida"] = (pd.Timestamp(f"{ano}-07-01"), pd.Timestamp(f"{ano}-12-31"))
    
    return j


# ============================================================================
# ETAPA 4: MODELOS DE ESTIMAÇÃO E INFERÊNCIA
# ============================================================================

def estimar_ols_simples(ret_y, ret_x):
    """
    Estima OLS manual: y = α + β·x
    Retorna dicionário com parâmetros ou None se dados insuficientes.
    """
    df = pd.DataFrame({"y": ret_y, "x": ret_x}).dropna()
    if len(df) < 30:
        return None
    
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    Y = df["y"].values
    
    try:
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef
        ss_res = (resid**2).sum()
        ss_tot = ((Y - Y.mean())**2).sum()
        
        return {
            "alpha": coef[0],
            "beta": coef[1],
            "sigma_resid": np.sqrt(ss_res / (len(df)-2)),
            "r_squared": 1 - ss_res/ss_tot if ss_tot > 0 else 0,
            "n_obs_est": len(df)
        }
    except:
        return None


def estimar_ols_hac(ret_y, ret_x, maxlags=1):
    """
    Estima OLS com erros padrão HAC (Newey-West) via statsmodels.
    Controla heterocedasticidade e autocorrelação.
    
    Parâmetros:
        ret_y    : Series de retornos do ativo
        ret_x    : Series de retornos do benchmark
        maxlags  : int, número de lags para HAC (default 1)
    
    Retorna:
        Dicionário com parâmetros e p-values ou None
    """
    import statsmodels.api as sm
    
    df = pd.DataFrame({"y": ret_y, "x": ret_x}).dropna()
    if len(df) < 30:
        return None
    
    X = sm.add_constant(df["x"])
    
    try:
        model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        return {
            "alpha": model.params.iloc[0],
            "beta": model.params.iloc[1],
            "alpha_pv": model.pvalues.iloc[0],
            "beta_pv": model.pvalues.iloc[1],
            "sigma_resid": np.sqrt(model.mse_resid),
            "r_squared": model.rsquared,
            "n_obs_est": int(model.nobs)
        }
    except:
        return None


def estimar_ff3(ret_y, df_factors_window):
    """
    Estima Fama-French 3 fatores: R_i - Rf = α + β·MktRf + s·SMB + h·HML + ε
t
    Parâmetros:
        ret_y             : Series de retornos do ativo
        df_factors_window : DataFrame com colunas Mkt_Rf, SMB, HML, Rf
    
    Retorna:
        Dicionário com parâmetros ou None
    """
    import statsmodels.api as sm
    
    common = ret_y.index.intersection(df_factors_window.index)
    if len(common) < 30:
        return None
    
    y = ret_y.loc[common]
    fac = df_factors_window.loc[common]
    
    rf = fac["Rf"] if "Rf" in fac.columns else 0
    y_excess = y - rf
    
    X_cols = [c for c in ["Mkt_Rf", "SMB", "HML"] if c in fac.columns]
    if len(X_cols) < 2:
        return None
    
    X = sm.add_constant(fac[X_cols])
    
    try:
        model = sm.OLS(y_excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        return {
            "alpha_ff3": model.params.iloc[0],
            "beta_mkt": model.params.get("Mkt_Rf", np.nan),
            "beta_smb": model.params.get("SMB", np.nan),
            "beta_hml": model.params.get("HML", np.nan),
            "alpha_pv_ff3": model.pvalues.iloc[0],
            "sigma_resid_ff3": np.sqrt(model.mse_resid),
            "r_squared_ff3": model.rsquared,
            "n_obs_est_ff3": int(model.nobs)
        }
    except:
        return None


def calcular_ar(ret_y, ret_x, alpha, beta):
    """
    Calcula Retornos Anormais (AR) usando o Modelo de Mercado.
    
    Fórmula: AR = R_i - (α + β·R_m)
    """
    df = pd.DataFrame({"y": ret_y, "x": ret_x}).dropna()
    return df["y"] - (alpha + beta * df["x"])


def calcular_ar_ff3(ret_y, df_factors_window, params_ff3):
    """
    Calcula Retornos Anormais usando Fama-French 3 fatores.
    
    Fórmula: AR = R_i - Rf - (α + β_mkt·MktRf + β_smb·SMB + β_hml·HML)
    """
    common = ret_y.index.intersection(df_factors_window.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    
    y = ret_y.loc[common]
    fac = df_factors_window.loc[common]
    
    rf = fac["Rf"] if "Rf" in fac.columns else 0
    expected = (params_ff3["alpha_ff3"]
                + params_ff3["beta_mkt"] * fac.get("Mkt_Rf", 0)
                + params_ff3["beta_smb"] * fac.get("SMB", 0)
                + params_ff3["beta_hml"] * fac.get("HML", 0))
    
    return (y - rf) - expected


def tstat_car_silva(car_values, ar_series_list, n_dias):
    """
    Teste t para CAR com desvio padrão corrigido (Silva et al., 2015).
    
    Fórmula: csd_t = sqrt(t·var_media + 2·(t-1)·cov_media)
    
    Corrige para autocorrelação nos retornos anormais.
    """
    n = len(car_values)
    if n < 2:
        return {"t_silva": np.nan, "p_silva": np.nan}
    
    car_mean = np.mean(car_values)
    variances, covariances = [], []
    
    for ar_s in ar_series_list:
        arr = np.array(ar_s.dropna()) if hasattr(ar_s, 'dropna') else np.array(ar_s)
        if len(arr) < 2:
            continue
        variances.append(np.var(arr, ddof=1))
        if len(arr) > 2:
            covariances.append(np.cov(arr[:-1], arr[1:])[0, 1])
    
    if not variances:
        return {"t_silva": np.nan, "p_silva": np.nan}
    
    var_m = np.mean(variances)
    cov_m = np.mean(covariances) if covariances else 0
    t = max(n_dias, 1)
    
    csd_sq = t * var_m + 2 * max(t-1, 0) * cov_m
    if csd_sq <= 0:
        return {"t_silva": np.nan, "p_silva": np.nan}
    
    t_stat = car_mean * np.sqrt(n) / np.sqrt(csd_sq)
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), max(n-1, 1)))
    
    return {"t_silva": t_stat, "p_silva": p_val}


def tstat_car_bmp(car_values, sigma_resids, n_dias):
    """
    Teste BMP (Boehmer, Musumeci & Poulsen, 1991).
    
    Fórmula: SCAR = CAR / (σ·√T)
    
    Controla variância induzida pelo evento.
    """
    n = len(car_values)
    if n < 2:
        return {"t_bmp": np.nan, "p_bmp": np.nan}
    
    denom = sigma_resids * np.sqrt(max(n_dias, 1))
    scar = np.where(denom > 0, car_values / denom, np.nan)
    scar = scar[~np.isnan(scar)]
    
    if len(scar) < 2:
        return {"t_bmp": np.nan, "p_bmp": np.nan}
    
    s_std = np.std(scar, ddof=1)
    if s_std == 0:
        return {"t_bmp": np.nan, "p_bmp": np.nan}
    
    t_bmp = np.mean(scar) * np.sqrt(len(scar)) / s_std
    p_bmp = 2 * (1 - stats.t.cdf(abs(t_bmp), len(scar)-1))
    
    return {"t_bmp": t_bmp, "p_bmp": p_bmp}


# ============================================================================
# ETAPA 5: EXECUÇÃO EVENT STUDY (MODELO DE MERCADO)
# ============================================================================

def executar_analise(df_ret_setores, ret_ibov, cenario="completo", ponderacao="EW"):
    """
    Executa Event Study usando Modelo de Mercado com OLS simples + HAC.
    
    Parâmetros:
        df_ret_setores : DataFrame de retornos setoriais
        ret_ibov       : Series de retornos do Ibovespa
        cenario        : str, "completo" ou "sem_crises"
        ponderacao     : str, "EW" (Equal-Weighted) ou "VW" (Volume-Weighted)
    
    Retorna:
        DataFrame com resultados do Event Study
    """
    log.info("  EVENT STUDY [%s / %s] ...", cenario, ponderacao)
    
    bdates = df_ret_setores.dropna(how="all").index
    resultados = []

    for ano in ANOS_ELEITORAIS:
        dt1 = DATAS_PRIMEIRO_TURNO[ano]
        est_ini, est_fim = janela_estimacao(bdates, dt1)
        jans = janelas_evento(bdates, ano)

        for setor in df_ret_setores.columns:
            m_est = (df_ret_setores.index >= est_ini) & (df_ret_setores.index <= est_fim)
            rs_est = df_ret_setores.loc[m_est, setor].dropna()
            rm_est = ret_ibov.loc[m_est].dropna()

            # OLS simples
            p = estimar_ols_simples(rs_est, rm_est)
            if p is None:
                continue
            
            # OLS HAC
            p_hac = estimar_ols_hac(rs_est, rm_est)

            for nome_j, (ji, jf) in jans.items():
                m_j = (df_ret_setores.index >= ji) & (df_ret_setores.index <= jf)
                rs_j = df_ret_setores.loc[m_j, setor].dropna()
                rm_j = ret_ibov.loc[m_j].dropna()
                
                if len(rs_j) < 3:
                    continue

                ar = calcular_ar(rs_j, rm_j, p["alpha"], p["beta"])
                car = ar.sum()
                nd = len(ar)
                ar_std = ar.std()
                
                t_s = (ar.mean() * np.sqrt(nd) / ar_std) if ar_std > 0 else np.nan
                p_s = 2*(1-stats.t.cdf(abs(t_s), max(nd-1,1))) if not np.isnan(t_s) else np.nan

                resultados.append({
                    "cenario": cenario,
                    "ponderacao": ponderacao,
                    "ano": ano,
                    "setor": setor,
                    "janela": nome_j,
                    "car": car,
                    "ar_medio": ar.mean(),
                    "ar_std": ar_std,
                    "n_dias": nd,
                    "alpha": p["alpha"],
                    "beta": p["beta"],
                    "r_squared": p["r_squared"],
                    "sigma_resid": p["sigma_resid"],
                    "n_obs_est": p["n_obs_est"],
                    "alpha_hac_pv": p_hac["alpha_pv"] if p_hac else np.nan,
                    "beta_hac_pv": p_hac["beta_pv"] if p_hac else np.nan,
                    "est_inicio": est_ini,
                    "est_fim": est_fim,
                    "janela_inicio": ji,
                    "janela_fim": jf,
                    "t_simples": t_s,
                    "p_simples": p_s,
                    "t_silva": np.nan,
                    "p_silva": np.nan,
                    "t_bmp": np.nan,
                    "p_bmp": np.nan,
                })

    df_res = pd.DataFrame(resultados)

    # Testes cross-sectional por (ano, janela)
    if not df_res.empty:
        for (ano, jan), grp in df_res.groupby(["ano", "janela"]):
            cars = grp["car"].values
            sigmas = grp["sigma_resid"].values
            nd = int(grp["n_dias"].median())
            
            ar_lists = []
            for _, row in grp.iterrows():
                m = (df_ret_setores.index >= row["janela_inicio"]) & \
                    (df_ret_setores.index <= row["janela_fim"])
                rs = df_ret_setores.loc[m, row["setor"]].dropna()
                rm = ret_ibov.loc[m].dropna()
                ar_lists.append(calcular_ar(rs, rm, row["alpha"], row["beta"]))

            silva = tstat_car_silva(cars, ar_lists, nd)
            bmp = tstat_car_bmp(cars, sigmas, nd)
            
            mask = (df_res["ano"]==ano) & (df_res["janela"]==jan) & \
                   (df_res["cenario"]==cenario) & (df_res["ponderacao"]==ponderacao)
            df_res.loc[mask, "t_silva"] = silva["t_silva"]
            df_res.loc[mask, "p_silva"] = silva["p_silva"]
            df_res.loc[mask, "t_bmp"] = bmp["t_bmp"]
            df_res.loc[mask, "p_bmp"] = bmp["p_bmp"]

    # Adiciona flags de significância
    for col, pv in [("sig5_simples","p_simples"), ("sig5_silva","p_silva"), ("sig5_bmp","p_bmp")]:
        df_res[col] = df_res[pv] < 0.05
    
    for col, pv in [("sig10_simples","p_simples"), ("sig10_silva","p_silva"), ("sig10_bmp","p_bmp")]:
        df_res[col] = df_res[pv] < 0.10
    
    for col, pv in [("sig1_simples","p_simples"), ("sig1_silva","p_silva"), ("sig1_bmp","p_bmp")]:
        df_res[col] = df_res[pv] < 0.01

    return df_res


# ============================================================================
# ETAPA 5b: EXECUÇÃO EVENT STUDY (FAMA-FRENCH 3)
# ============================================================================

def executar_analise_ff3(df_ret_setores, df_factors, cenario="completo", ponderacao="EW"):
    """
    Executa Event Study com Fama-French 3 fatores (NEFIN-USP).
    
    Parâmetros:
        df_ret_setores : DataFrame de retornos setoriais
        df_factors     : DataFrame com fatores FF3 do NEFIN
        cenario        : str, identificador do cenário
        ponderacao     : str, "EW" ou "VW"
    
    Retorna:
        DataFrame com resultados FF3
    """
    if df_factors is None:
        log.info("  FF3 desabilitado (sem fatores NEFIN)")
        return pd.DataFrame()

    log.info("  EVENT STUDY FF3 [%s / %s] ...", cenario, ponderacao)
    
    bdates = df_ret_setores.dropna(how="all").index
    resultados = []

    for ano in ANOS_ELEITORAIS:
        dt1 = DATAS_PRIMEIRO_TURNO[ano]
        est_ini, est_fim = janela_estimacao(bdates, dt1)
        jans = janelas_evento(bdates, ano)

        m_est = (df_factors.index >= est_ini) & (df_factors.index <= est_fim)
        fac_est = df_factors.loc[m_est]

        for setor in df_ret_setores.columns:
            rs_est = df_ret_setores.loc[
                (df_ret_setores.index >= est_ini) & (df_ret_setores.index <= est_fim), setor
            ].dropna()
            
            pff = estimar_ff3(rs_est, fac_est)
            if pff is None:
                continue

            for nome_j, (ji, jf) in jans.items():
                m_j = (df_ret_setores.index >= ji) & (df_ret_setores.index <= jf)
                rs_j = df_ret_setores.loc[m_j, setor].dropna()
                fac_j = df_factors.loc[(df_factors.index >= ji) & (df_factors.index <= jf)]
                
                if len(rs_j) < 3:
                    continue

                ar = calcular_ar_ff3(rs_j, fac_j, pff)
                if len(ar) < 3:
                    continue
                
                car = ar.sum()
                nd = len(ar)
                ar_std = ar.std()
                
                t_s = (ar.mean()*np.sqrt(nd)/ar_std) if ar_std > 0 else np.nan
                p_s = 2*(1-stats.t.cdf(abs(t_s), max(nd-1,1))) if not np.isnan(t_s) else np.nan

                resultados.append({
                    "cenario": cenario,
                    "ponderacao": ponderacao,
                    "modelo": "FF3",
                    "ano": ano,
                    "setor": setor,
                    "janela": nome_j,
                    "car_ff3": car,
                    "ar_medio_ff3": ar.mean(),
                    "n_dias": nd,
                    "alpha_ff3": pff["alpha_ff3"],
                    "beta_mkt": pff["beta_mkt"],
                    "beta_smb": pff["beta_smb"],
                    "beta_hml": pff["beta_hml"],
                    "r_squared_ff3": pff["r_squared_ff3"],
                    "t_simples_ff3": t_s,
                    "p_simples_ff3": p_s,
                    "sig5_ff3": p_s < 0.05 if not np.isnan(p_s) else False,
                    "sig10_ff3": p_s < 0.10 if not np.isnan(p_s) else False,
                    "sig1_ff3": p_s < 0.01 if not np.isnan(p_s) else False,
                })

    return pd.DataFrame(resultados)


# ============================================================================
# ETAPA 6: TESTE PLACEBO
# ============================================================================

def executar_teste_placebo(df_ret, ret_ibov, n_placebos=N_PLACEBO_EVENTS, seed=42):
    """
    Executa teste placebo com pseudo-eventos em anos não-eleitorais.
    
    Parâmetros:
        df_ret      : DataFrame de retornos setoriais
        ret_ibov    : Series de retornos do Ibovespa
        n_placebos  : int, número de pseudo-eventos
        seed        : int, seed para reprodutibilidade
    
    Retorna:
        DataFrame com resultados placebo
    """
    log.info("=" * 70)
    log.info("ETAPA 6: TESTE PLACEBO (%d pseudo-eventos)", n_placebos)
    log.info("=" * 70)
    
    np.random.seed(seed)
    bdates = df_ret.dropna(how="all").index
    
    # Anos não-eleitorais no período
    anos_ne = [a for a in range(2003, 2022) if a not in ANOS_ELEITORAIS]
    
    resultados = []
    
    for i in range(n_placebos):
        ano = np.random.choice(anos_ne)
        dt = pd.Timestamp(f"{ano}-10-05")  # Data fictícia de "eleição"
        
        ei, ef = janela_estimacao(bdates, dt)
        
        windows = [
            ("placebo_antecip", offset_du(bdates, dt, -45), offset_du(bdates, dt, -1)),
            ("placebo_reacao", offset_du(bdates, dt, -5), offset_du(bdates, dt, +5))
        ]
        
        for setor in df_ret.columns:
            m = (df_ret.index >= ei) & (df_ret.index <= ef)
            p = estimar_ols_simples(df_ret.loc[m, setor].dropna(), ret_ibov.loc[m].dropna())
            
            if p is None:
                continue
            
            for nome, ji, jf in windows:
                mj = (df_ret.index >= ji) & (df_ret.index <= jf)
                rs = df_ret.loc[mj, setor].dropna()
                rm = ret_ibov.loc[mj].dropna()
                
                if len(rs) < 3:
                    continue
                
                ar = calcular_ar(rs, rm, p["alpha"], p["beta"])
                resultados.append({
                    "placebo_id": i,
                    "ano_placebo": ano,
                    "setor": setor,
                    "janela": nome,
                    "car": ar.sum(),
                    "n_dias": len(ar)
                })
    
    df_p = pd.DataFrame(resultados)
    log.info("  → %d obs placebo", len(df_p))
    
    return df_p


# ============================================================================
# ETAPA 6b: DIFFERENCE-IN-DIFFERENCES (Estatais vs. Privadas)
# ============================================================================

def executar_did_empresas(df_precos, df_empresas, ret_ibov):
    """
    Executa DiD no nível de empresa individual: Estatais vs. Privadas.
    
    Calcula CAR por empresa para cada (ano, janela) e testa:
    Δ = CAR_estatais − CAR_privadas (Welch t-test)
    
    Retorna:
        (df_did_individual, df_did_agregado)
    """
    log.info("=" * 70)
    log.info("ETAPA 6b: DiD ESTATAIS vs. PRIVADAS (nível empresa)")
    log.info("=" * 70)

    # Retornos de cada ativo individual
    ret_ativos = np.log(df_precos / df_precos.shift(1))

    # Mapeamento ticker → (setor, estatal)
    t2info = {}
    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        t2info[tk] = {
            "setor": row["SETOR_B3"],
            "estatal": row["ESTATAL"],
            "denom": row.get("DENOM_SOCIAL", "")
        }

    bdates = ret_ativos.dropna(how="all").index

    # Janelas principais para DiD
    janelas_did = ["antecipacao_45", "reacao_curta_1t", "reacao_curta_2t"]

    resultados_ind = []
    
    n_estatais = sum(1 for v in t2info.values() if v["estatal"])
    n_privadas = sum(1 for v in t2info.values() if not v["estatal"])
    
    log.info("  Empresas com dados: %d tickers (%d estatais, %d privadas)",
             len([c for c in ret_ativos.columns if c in t2info]),
             n_estatais, n_privadas)

    for ano in ANOS_ELEITORAIS:
        dt1 = DATAS_PRIMEIRO_TURNO[ano]
        est_ini, est_fim = janela_estimacao(bdates, dt1)
        jans = janelas_evento(bdates, ano)

        # Filtra só as janelas de interesse
        jans_did = {k: v for k, v in jans.items() if k in janelas_did}

        tickers_processados = 0
        
        for ticker in ret_ativos.columns:
            if ticker not in t2info:
                continue

            info = t2info[ticker]

            # Retornos na janela de estimação
            m_est = (ret_ativos.index >= est_ini) & (ret_ativos.index <= est_fim)
            ret_tk_est = ret_ativos.loc[m_est, ticker].dropna()
            ret_m_est = ret_ibov.loc[m_est].dropna()

            if len(ret_tk_est) < 30:
                continue

            params = estimar_ols_simples(ret_tk_est, ret_m_est)
            if params is None:
                continue

            for nome_j, (ji, jf) in jans_did.items():
                m_j = (ret_ativos.index >= ji) & (ret_ativos.index <= jf)
                ret_tk_j = ret_ativos.loc[m_j, ticker].dropna()
                ret_m_j = ret_ibov.loc[m_j].dropna()
                
                if len(ret_tk_j) < 3:
                    continue

                ar = calcular_ar(ret_tk_j, ret_m_j, params["alpha"], params["beta"])
                car = ar.sum()
                nd = len(ar)

                resultados_ind.append({
                    "ano": ano,
                    "janela": nome_j,
                    "ticker": ticker,
                    "setor": info["setor"],
                    "estatal": info["estatal"],
                    "denom_social": str(info["denom"])[:50],
                    "car": car,
                    "n_dias": nd,
                    "alpha": params["alpha"],
                    "beta": params["beta"],
                })
            
            tickers_processados += 1

        log.info("  Ano %d: %d empresas processadas", ano, tickers_processados)

    df_ind = pd.DataFrame(resultados_ind)
    log.info("  → %d observações individuais", len(df_ind))

    # Agrega: DiD por (ano, janela)
    resultados_agg = []
    
    if not df_ind.empty:
        for (ano, janela), grp in df_ind.groupby(["ano", "janela"]):
            est = grp[grp["estatal"]]["car"].values
            priv = grp[~grp["estatal"]]["car"].values

            if len(est) < 2 or len(priv) < 2:
                continue

            delta = np.mean(est) - np.mean(priv)
            t_did, p_did = stats.ttest_ind(est, priv, equal_var=False)

            resultados_agg.append({
                "ano": ano,
                "janela": janela,
                "car_medio_estatais": np.mean(est),
                "car_medio_privadas": np.mean(priv),
                "delta_did": delta,
                "t_did": t_did,
                "p_did": p_did,
                "sig5_did": p_did < 0.05 if not np.isnan(p_did) else False,
                "sig10_did": p_did < 0.10 if not np.isnan(p_did) else False,
                "n_estatais": len(est),
                "n_privadas": len(priv),
                "car_std_estatais": np.std(est, ddof=1),
                "car_std_privadas": np.std(priv, ddof=1),
            })

    df_agg = pd.DataFrame(resultados_agg)
    
    if not df_agg.empty:
        n_sig = df_agg["sig5_did"].sum()
        log.info("  → DiD: %d comparações, %d significativas a 5%%", len(df_agg), n_sig)
    
    return df_ind, df_agg


# ============================================================================
# ETAPA 7: VOLATILIDADE, BENCHMARKING E SHARPE
# ============================================================================

def calcular_volatilidade_comparativa(df_ret):
    """
    Calcula volatilidade anualizada comparando anos eleitorais vs. não-eleitorais.
    
    Retorna:
        DataFrame com volatilidade por setor/ano/tipo
    """
    log.info("Calculando volatilidade comparativa ...")
    
    registros = []
    
    for setor in df_ret.columns:
        for ano in range(2002, 2023):
            m = df_ret.index.year == ano
            r = df_ret.loc[m, setor].dropna()
            
            if len(r) < 20:
                continue
            
            registros.append({
                "setor": setor,
                "ano": ano,
                "tipo_ano": "Eleitoral" if ano in ANOS_ELEITORAIS else "Não-Eleitoral",
                "vol_anualizada": r.std() * np.sqrt(252)
            })
    
    return pd.DataFrame(registros)


def calcular_sharpe_comparativo(df_ret):
    """
    Calcula Sharpe Ratio comparando anos eleitorais vs. não-eleitorais.
    
    Fórmula: Sharpe = (R_setor − Rf) / σ_setor, anualizado
    
    Retorna:
        DataFrame com Sharpe por setor/ano/tipo
    """
    log.info("Calculando Sharpe Ratio comparativo ...")
    
    registros = []
    
    for setor in df_ret.columns:
        for ano in range(2002, 2023):
            m = df_ret.index.year == ano
            r = df_ret.loc[m, setor].dropna()
            
            if len(r) < 20:
                continue
            
            rf_diario = SELIC_DIARIA.get(ano, 0.0003)
            excess = r - rf_diario
            
            sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else np.nan

            # Teste formal: H0: excess_return_medio = 0
            t_exc, p_exc = stats.ttest_1samp(excess.dropna(), 0)

            registros.append({
                "setor": setor,
                "ano": ano,
                "tipo_ano": "Eleitoral" if ano in ANOS_ELEITORAIS else "Não-Eleitoral",
                "sharpe_anualizado": sharpe,
                "excesso_medio_diario": excess.mean(),
                "t_excesso": t_exc,
                "p_excesso": p_exc,
            })
    
    return pd.DataFrame(registros)


def adicionar_benchmarking(df_res):
    """
    Adiciona colunas de benchmarking com SELIC e IPCA.
    
    Nota: Comparativo ilustrativo, NÃO é teste formal.
    """
    df = df_res.copy()
    df["selic_proporcional"] = df["ano"].map(SELIC_ANUAL) * df["n_dias"] / 252
    df["ipca_proporcional"] = df["ano"].map(IPCA_ANUAL) * df["n_dias"] / 252
    df["car_exc_selic"] = df["car"] - df["selic_proporcional"]
    df["car_exc_ipca"] = df["car"] - df["ipca_proporcional"]
    return df


# ============================================================================
# ETAPA 8: VISUALIZAÇÕES
# ============================================================================

def gerar_visualizacoes(df_res, df_vol, df_placebo, df_did, df_sharpe, output_dir):
    """
    Gera todas as visualizações acadêmicas do estudo.
    
    Inclui:
      - Heatmaps de CAR por setor e ano
      - Evolução temporal dos CARs
      - Volatilidade comparativa
      - Teste placebo
      - DiD (Estatais vs. Privadas)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": (16, 9),
        "font.size": 11,
        "figure.dpi": 150
    })

    df = df_res[(df_res["cenario"]=="completo") & (df_res["ponderacao"]=="EW")]

    # ---- HEATMAPS --------------------------------------------------------
    hm_map = {
        "antecipacao_45": "Antecipação (−45 a −1 d.u.)",
        "reacao_curta_1t": "Reação 1º Turno [−5,+5]",
        "reacao_curta_2t": "Reação 2º Turno [−5,+5]",
        "reacao_media_1t": "Reação [−10,+10] (Robustez)",
        "reacao_ampla_1t": "Reação [−20,+20] (Robustez)",
        "antecipacao_60": "Antecipação (−60 d.u.) (Robustez)",
        "estendida": "Últimos 6 Meses",
    }
    
    for jkey, titulo in hm_map.items():
        dj = df[df["janela"]==jkey]
        if dj.empty:
            continue
        
        pivot = dj.pivot_table(index="setor", columns="ano", values="car", aggfunc="mean")
        pivot_sig = dj.pivot_table(index="setor", columns="ano", values="sig5_simples", aggfunc="any")
        
        if pivot.empty:
            continue
        
        annot = pivot.copy().astype(object)
        for r in annot.index:
            for c in annot.columns:
                v = pivot.loc[r,c] if r in pivot.index and c in pivot.columns else np.nan
                s = pivot_sig.loc[r,c] if r in pivot_sig.index and c in pivot_sig.columns else False
                annot.loc[r,c] = f"{v:.3f}{'*' if s else ''}" if pd.notna(v) else ""
        
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=0,
                    linewidths=.5, linecolor="gray",
                    cbar_kws={"label":"CAR (EW)"}, ax=ax)
        ax.set_title(f"CAR por Setor e Ano – {titulo}\n(Índice Equal-Weighted · * p<0.05)",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Ano Eleitoral")
        ax.set_ylabel("Setor B3")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{jkey}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ heatmap_%s.png", jkey)

    # ---- EVOLUÇÃO TEMPORAL -----------------------------------------------
    df_ant = df[df["janela"]=="antecipacao_45"]
    if not df_ant.empty:
        mag = df_ant.groupby("setor")["car"].apply(lambda x: abs(x).mean())
        top5 = mag.nlargest(5).index.tolist()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        for idx, (j, nm) in enumerate([("antecipacao_45","Antecipação −45"), 
                                        ("reacao_curta_1t","Reação [−5,+5]")]):
            ax = axes[idx]
            for s in top5:
                d = df[(df["setor"]==s) & (df["janela"]==j)].sort_values("ano")
                if not d.empty:
                    ax.plot(d["ano"], d["car"], marker="o", lw=2, label=s, ms=7)
            ax.axhline(0, color="k", ls="--", alpha=.3)
            ax.set_xlabel("Ano")
            ax.set_ylabel("CAR (EW)")
            ax.set_title(f"Top 5 – {nm}", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=.3)
        
        plt.suptitle("Evolução Temporal dos CARs (Equal-Weighted)", 
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "evolucao_temporal_cars.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ evolucao_temporal_cars.png")

    # ---- VOLATILIDADE COMPARATIVA ----------------------------------------
    if not df_vol.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        va = df_vol.groupby(["setor", "tipo_ano"])["vol_anualizada"].mean().reset_index()
        pv = va.pivot(index="setor", columns="tipo_ano", values="vol_anualizada")
        pv.plot(kind="barh", ax=ax, alpha=.85, edgecolor="k")
        ax.set_title("Volatilidade Anualizada: Eleitoral vs. Não-Eleitoral (EW)", 
                     fontweight="bold")
        ax.set_xlabel("σ·√252")
        ax.set_ylabel("")
        ax.legend(title="Tipo de Ano")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volatilidade_comparativa.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ volatilidade_comparativa.png")

    # ---- PLACEBO vs REAL -------------------------------------------------
    if not df_placebo.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (jr, jp, nm) in enumerate([("antecipacao_45","placebo_antecip","Antecipação"),
                                              ("reacao_curta_1t","placebo_reacao","Reação [−5,+5]")]):
            ax = axes[idx]
            cp = df_placebo[df_placebo["janela"]==jp]["car"].dropna()
            cr = df[df["janela"]==jr]["car"].dropna()
            
            if len(cp) > 0:
                ax.hist(cp, bins=30, alpha=.5, label="Placebo", color="gray", 
                        density=True, edgecolor="k")
            if len(cr) > 0:
                ax.hist(cr, bins=15, alpha=.7, label="Eleições", color="steelblue", 
                        density=True, edgecolor="k")
            ax.axvline(0, color="r", ls="--", alpha=.5)
            ax.set_title(f"CARs – {nm}", fontweight="bold")
            ax.set_xlabel("CAR")
            ax.legend()
        
        plt.suptitle("Teste Placebo: Eleições Reais vs. Pseudo-Eventos", 
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "placebo_test.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ placebo_test.png")

    # ---- DiD CHART (Estatais vs. Privadas) -------------------------------
    if not df_did.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for jan in ["antecipacao_45", "reacao_curta_1t", "reacao_curta_2t"]:
            d = df_did[df_did["janela"]==jan].sort_values("ano")
            if not d.empty:
                ax.plot(d["ano"], d["delta_did"], marker="s", lw=2, ms=8, label=jan)
                for _, r in d.iterrows():
                    if r["sig5_did"]:
                        ax.annotate("*", (r["ano"], r["delta_did"]), fontsize=14,
                                    ha="center", va="bottom", color="red")
        
        ax.axhline(0, color="k", ls="--", alpha=.3)
        ax.set_title("Difference-in-Differences: Estatais − Privadas (* p<0.05)\n"
                     "(Nível de Empresa Individual)", fontweight="bold")
        ax.set_xlabel("Ano Eleitoral")
        ax.set_ylabel("Δ CAR (Estatais − Privadas)")
        ax.legend()
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "did_estatais_vs_privadas.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ did_estatais_vs_privadas.png")

    # ---- SHARPE COMPARATIVO ----------------------------------------------
    if not df_sharpe.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        sharpe_pivot = df_sharpe.groupby(["setor", "tipo_ano"])["sharpe_anualizado"].mean().reset_index()
        sharpe_pv = sharpe_pivot.pivot(index="setor", columns="tipo_ano", values="sharpe_anualizado")
        sharpe_pv.plot(kind="barh", ax=ax, alpha=.85, edgecolor="k")
        
        ax.axvline(0, color="k", ls="--", alpha=.5)
        ax.set_title("Sharpe Ratio: Eleitoral vs. Não-Eleitoral", fontweight="bold")
        ax.set_xlabel("Sharpe Anualizado")
        ax.set_ylabel("")
        ax.legend(title="Tipo de Ano")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sharpe_comparativo.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ sharpe_comparativo.png")


# ============================================================================
# ETAPA 9: EXPORTAÇÃO E METODOLOGIA
# ============================================================================

def gerar_texto_metodologia(output_dir):
    """
    Gera arquivo markdown com metodologia completa e limitações.
    """
    texto = r"""# Metodologia e Limitações — Análise de Ciclos Eleitorais na B3

## 1. Desenho do Estudo

Estudo de Evento (MacKinlay, 1997) aplicado aos ciclos eleitorais presidenciais brasileiros: 2002, 2006, 2010, 2014, 2018, 2022.

### 1.1 Objetivo da Pesquisa

Quantificar o "risco político" no mercado brasileiro via Mapa de Risco Setorial, analisando o impacto dos ciclos eleitorais nos retornos setoriais da B3.

### 1.2 Universo e Período

- **Período**: 2002–2022 (6 Eleições Presidenciais)
- **Universo**: ~700 tickers (incluindo IPOs e empresas deslistadas)
- **Classificação**: 11 setores oficiais da B3

### 1.3 Índices Setoriais

Dois esquemas de ponderação:

- **Equal-Weighted (EW)**: retorno médio simples das empresas ativas.
  - *Nota*: Amplifica small caps. Reportado como análise principal.
  
- **Volume-Weighted (VW)**: retorno ponderado por volume financeiro médio 20 dias.
  - Proxy de market cap. Reportado como robustez.

### 1.4 Pré-processamento

- **Winsorização**: tratamento de outliers nos percentis 1% e 99%
- **Filtro de Liquidez**: mínimo 80% de pregões no ano para inclusão
- **Filtro de Existência**: considera DT_REG e DT_CANCEL

### 1.5 Janela de Estimação

**[-252, -30] dias úteis antes do 1º turno** (~222 d.u., ~10,5 meses).
Janela móvel que evita contaminação por choques pré-evento (MacKinlay, 1997).

### 1.6 Modelos de Retorno Normal

#### (A) Modelo de Mercado (CAPM)

$$R_{i,t} = \\alpha + \\beta \\cdot R_{m,t} + \\varepsilon$$

- Benchmark: Ibovespa (^BVSP)
- Estimação via OLS com erros padrão HAC-Newey-West (1 lag)

#### (B) Fama-French 3 Fatores (BR)

$$R_i - R_f = \\alpha + \\beta \\cdot MktRf + s \\cdot SMB + h \\cdot HML + \\varepsilon$$

- Fatores: NEFIN-USP (nefin.com.br)
- Controla por tamanho (SMB) e valor (HML)
- Estimação via OLS-HAC

### 1.7 Janelas de Evento

| Janela | Definição | Motivação |
|--------|-----------|-----------|
| Antecipação 45 d.u. | [−45, −1] 1ºT | Efeito Propaganda / Pricing-in |
| Antecipação 60 d.u. | [−60, −1] 1ºT | Robustez |
| Reação Curta 1ºT | [−5, +5] | Reação imediata |
| Reação Média | [−10, +10] | Robustez |
| Reação Ampla | [−20, +20] | Robustez |
| Reação 2ºT | [−5, +5] 2ºT | Resolução de incerteza |
| Ciclo Interno | 1ºSem vs 2ºSem | Expectativa vs. Disputa |
| Estendida | Jul–Dez | Efeito agregado |

### 1.8 Inferência Estatística

1. **Teste t simples** (longitudinal): $t = \\bar{AR} \\cdot \\sqrt{n} / \\sigma(AR)$

2. **Teste t Silva et al. (2015)**: $csd_t = \\sqrt{t \\cdot var + 2(t-1) \\cdot cov}$ — corrige autocorrelação

3. **Teste BMP (1991)**: $SCAR = CAR / (\\sigma \\cdot \\sqrt{T})$ — controla event-induced variance

4. **OLS-HAC**: Newey-West com 1 lag na estimação dos parâmetros

### 1.9 Testes de Robustez

- Janelas alternativas: [−60,−1], [−10,+10], [−20,+20]
- Cenário sem crises (exclui 2008/2020)
- Teste placebo: 1000 pseudo-eventos em anos não-eleitorais
- DiD: Estatais vs. Privadas (teste Welch)
- Sharpe Ratio + teste formal de excesso de retorno vs. Selic
- Ponderação alternativa: Volume-Weighted (VW)
- Modelo alternativo: Fama-French 3 fatores (NEFIN)

## 2. Benchmarks Utilizados

- **Ibovespa (^BVSP)**: benchmark de mercado
- **SELIC**: custo de oportunidade
- **IPCA**: ajuste para retornos reais
- **Fatores NEFIN-USP**: SMB, HML, Mkt-Rf, Rf

## 3. Limitações

1. **Viés de sobrevivência**: ~45% dos tickers não possuem dados no yfinance.
   Small caps e deslistadas sub-representadas, especialmente pré-2006.
   Mitigação: mapeamento De-Para + filtro de liquidez.

2. **Ponderação**: EW amplifica small caps. VW reportado como robustez.

3. **Clusterização de volatilidade**: Mitigada por HAC + BMP, mas não eliminada.

4. **Endogeneidade**: Resultados são evidência consistente com risco político —
   NÃO causalidade. Placebo e DiD oferecem validação parcial.

5. **Fonte de dados (yfinance)**: API gratuita, sujeita a falhas e lacunas.
   Solução definitiva: COTAHIST/B3 (séries históricas brutas).

6. **Benchmarking Selic/IPCA**: Ilustrativo. O teste formal de excesso usa
   Sharpe Ratio + t-test do excesso de retorno diário.

## 4. Referências

- Boehmer, E., Musumeci, J., & Poulsen, A. (1991). Event-study methodology under conditions of event-induced variance. *Journal of Financial Economics*, 30(2), 253-272.

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13-39.

- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

- Nordhaus, W. D. (1975). The political business cycle. *Review of Economic Studies*, 42(2), 169-190.

- Silva, A. C., et al. (2015). Retornos anormais em IPOs na BMF&Bovespa. *Revista Brasileira de Finanças*.

---

**Nota**: Este estudo foi desenvolvido para fins acadêmicos. Os resultados devem ser interpretados como evidência consistente com risco político, não como prova de causalidade.
"""
    
    with open(os.path.join(output_dir, "metodologia_limitacoes.md"), "w", encoding="utf-8") as f:
        f.write(texto)
    
    log.info("  ✓ metodologia_limitacoes.md")


def exportar_resultados(df_res, df_ff3, df_vol, df_sobrev, df_comp, df_diag,
                        df_placebo, df_did_ind, df_did_agg, df_sharpe, output_dir):
    """
    Exporta todos os resultados para CSV e Excel.
    """
    log.info("=" * 70)
    log.info("ETAPA 9: EXPORTAÇÃO")
    log.info("=" * 70)

    # CSVs individuais
    df_res.to_csv(os.path.join(output_dir, "resultados_consolidados_v3.csv"), 
                  index=False, encoding="utf-8-sig")
    log.info("  ✓ resultados_consolidados_v3.csv")
    
    df_sobrev.to_csv(os.path.join(output_dir, "tabela_sobrevivencia.csv"), 
                     index=False, encoding="utf-8-sig")
    log.info("  ✓ tabela_sobrevivencia.csv")
    
    df_diag.to_csv(os.path.join(output_dir, "diagnostico_download.csv"), 
                   index=False, encoding="utf-8-sig")
    log.info("  ✓ diagnostico_download.csv")
    
    if not df_placebo.empty:
        df_placebo.to_csv(os.path.join(output_dir, "placebo_test_results.csv"), 
                          index=False, encoding="utf-8-sig")
        log.info("  ✓ placebo_test_results.csv")
    
    if not df_did_ind.empty:
        df_did_ind.to_csv(os.path.join(output_dir, "did_individual_empresas.csv"), 
                          index=False, encoding="utf-8-sig")
        log.info("  ✓ did_individual_empresas.csv")
    
    if not df_did_agg.empty:
        df_did_agg.to_csv(os.path.join(output_dir, "did_estatais_vs_privadas.csv"), 
                          index=False, encoding="utf-8-sig")
        log.info("  ✓ did_estatais_vs_privadas.csv")
    
    if not df_ff3.empty:
        df_ff3.to_csv(os.path.join(output_dir, "resultados_ff3_v3.csv"), 
                      index=False, encoding="utf-8-sig")
        log.info("  ✓ resultados_ff3_v3.csv")
    
    if not df_sharpe.empty:
        df_sharpe.to_csv(os.path.join(output_dir, "sharpe_eleitoral.csv"), 
                         index=False, encoding="utf-8-sig")
        log.info("  ✓ sharpe_eleitoral.csv")
    
    if not df_vol.empty:
        df_vol.to_csv(os.path.join(output_dir, "volatilidade_comparativa.csv"), 
                      index=False, encoding="utf-8-sig")
        log.info("  ✓ volatilidade_comparativa.csv")

    # Excel consolidado com múltiplas abas
    xlsx = os.path.join(output_dir, "analise_ciclos_eleitorais_v3.xlsx")
    
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        # Resultados completos
        df_res.to_excel(writer, sheet_name="Resultados", index=False)

        # Pivot tables para principais janelas
        df_ew = df_res[(df_res["cenario"]=="completo") & (df_res["ponderacao"]=="EW")]
        
        for jan in ["antecipacao_45", "reacao_curta_1t", "reacao_curta_2t", "estendida"]:
            dj = df_ew[df_ew["janela"]==jan]
            if dj.empty:
                continue
            
            pc = dj.pivot_table(index="setor", columns="ano", values="car")
            pp = dj.pivot_table(index="setor", columns="ano", values="p_simples")
            
            merged = pc.copy().astype(object)
            for r in merged.index:
                for c in merged.columns:
                    v = pc.loc[r,c] if r in pc.index and c in pc.columns else np.nan
                    p = pp.loc[r,c] if r in pp.index and c in pp.columns else np.nan
                    if pd.notna(v):
                        star = "***" if pd.notna(p) and p < .01 else \
                               ("**" if pd.notna(p) and p < .05 else \
                                ("*" if pd.notna(p) and p < .10 else ""))
                        merged.loc[r,c] = f"{v:.4f}{star}"
            
            merged.to_excel(writer, sheet_name=f"CAR_{jan[:15]}")

        # Outras abas
        if not df_ff3.empty:
            df_ff3.to_excel(writer, sheet_name="FF3", index=False)
        
        if not df_vol.empty:
            df_vol.to_excel(writer, sheet_name="Volatilidade", index=False)
        
        if not df_did_agg.empty:
            df_did_agg.to_excel(writer, sheet_name="DiD Agregado", index=False)
        
        if not df_did_ind.empty:
            df_did_ind.head(100000).to_excel(writer, sheet_name="DiD Individual", index=False)
        
        if not df_sharpe.empty:
            df_sharpe.to_excel(writer, sheet_name="Sharpe", index=False)
        
        df_sobrev.to_excel(writer, sheet_name="Sobrevivência", index=False)
        df_comp.to_excel(writer, sheet_name="Composição", index=False)

        # Sumário executivo
        sumario = []
        for jan in sorted(df_ew["janela"].unique()):
            dj = df_ew[df_ew["janela"]==jan]
            if dj.empty:
                continue
            sumario.append({
                "Janela": jan,
                "CAR Médio": f"{dj['car'].mean():.4f}",
                "CAR Mediano": f"{dj['car'].median():.4f}",
                "%Sig1 simples": f"{dj['sig1_simples'].mean()*100:.0f}%",
                "%Sig5 simples": f"{dj['sig5_simples'].mean()*100:.0f}%",
                "%Sig10 simples": f"{dj['sig10_simples'].mean()*100:.0f}%",
                "%Sig5 Silva": f"{dj['sig5_silva'].mean()*100:.0f}%",
                "%Sig5 BMP": f"{dj['sig5_bmp'].mean()*100:.0f}%",
                "Setor |CAR| max": dj.loc[dj["car"].abs().idxmax(),"setor"] if len(dj) > 0 else "",
                "N": len(dj),
            })
        
        pd.DataFrame(sumario).to_excel(writer, sheet_name="Sumário", index=False)

    log.info("  ✓ %s", xlsx)
    gerar_texto_metodologia(output_dir)


# ============================================================================
# MAIN - EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal que executa toda a análise de ciclos eleitorais.
    """
    t0 = time.time()
    
    print("\n" + "█"*72)
    print("█  CICLOS ELEITORAIS NA B3 (2002–2022) — VERSÃO COMPLETA ACADÊMICA   █")
    print("█  CAPM + FF3 · HAC · BMP · Placebo · DiD · Sharpe · Winsorização    █")
    print("█"*72 + "\n")

    if not os.path.exists(ARQUIVO_ENTRADA):
        log.error("Arquivo não encontrado: %s", ARQUIVO_ENTRADA)
        sys.exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Ingestão -----------------------------------------------------
    df_emp = carregar_lista_empresas(ARQUIVO_ENTRADA)

    # ---- 2. Download preços + volumes ------------------------------------
    tickers_yf = df_emp["TICKER_YF"].unique().tolist()

    cache_p = os.path.join(OUTPUT_DIR, "_cache_precos.pkl")
    cache_v = os.path.join(OUTPUT_DIR, "_cache_volumes.pkl")
    cache_i = os.path.join(OUTPUT_DIR, "_cache_ibov.pkl")

    if all(os.path.exists(f) for f in [cache_p, cache_v, cache_i]):
        log.info("Carregando do cache ...")
        df_precos = pd.read_pickle(cache_p)
        df_volumes = pd.read_pickle(cache_v)
        ibov = pd.read_pickle(cache_i)
        df_diag = pd.DataFrame({
            "ticker_yf": tickers_yf,
            "status": ["ok" if t in df_precos.columns else "falha" for t in tickers_yf],
        })
    else:
        df_precos, df_volumes, df_diag = baixar_precos_yfinance(tickers_yf)
        ibov = baixar_ibovespa()
        df_precos.to_pickle(cache_p)
        df_volumes.to_pickle(cache_v)
        ibov.to_pickle(cache_i)

    # Enriquece diagnóstico
    ds = df_emp[["TICKER_YF","SETOR_B3","TICKER_ORIGINAL"]].drop_duplicates("TICKER_YF")
    df_diag = df_diag.merge(ds, left_on="ticker_yf", right_on="TICKER_YF", how="left")

    log.info("\nCobertura por setor:")
    for s in SETORES_B3:
        d = df_diag[df_diag["SETOR_B3"]==s]
        log.info("  %s: %d/%d (%.0f%%)", s, (d["status"]=="ok").sum(), len(d),
                 100*(d["status"]=="ok").sum()/max(len(d),1))

    # ---- 3. Filtro existência --------------------------------------------
    df_precos = aplicar_filtro_existencia(df_precos, df_emp)

    # ---- 4. Índices setoriais (EW + VW) ----------------------------------
    df_ret_ew, df_ret_vw, df_comp = construir_indices_setoriais(df_precos, df_volumes, df_emp)
    ret_ibov = np.log(ibov / ibov.shift(1)).dropna()

    # Alinhar índices
    idx_comum = df_ret_ew.index.intersection(ret_ibov.index)
    df_ret_ew = df_ret_ew.loc[idx_comum]
    df_ret_vw = df_ret_vw.loc[idx_comum]
    ret_ibov = ret_ibov.loc[idx_comum]

    # ---- Sobrevivência ---------------------------------------------------
    df_sobrev = gerar_tabela_sobrevivencia(df_emp, df_precos)

    # ---- Download fatores NEFIN ------------------------------------------
    cache_nefin = os.path.join(OUTPUT_DIR, "_cache_nefin.pkl")
    if os.path.exists(cache_nefin):
        df_factors = pd.read_pickle(cache_nefin)
        log.info("Fatores NEFIN carregados do cache")
    else:
        df_factors = baixar_fatores_nefin()
        if df_factors is not None:
            df_factors.to_pickle(cache_nefin)

    # ---- 5. Event Study: Modelo de Mercado (EW + VW) ---------------------
    log.info("\n" + "="*70)
    log.info("ETAPA 5: EVENT STUDY")
    log.info("="*70)

    df_res_ew = executar_analise(df_ret_ew, ret_ibov, "completo", "EW")
    df_res_vw = executar_analise(df_ret_vw, ret_ibov, "completo", "VW")

    # Cenário sem crises (EW apenas)
    df_ret_sc = df_ret_ew.copy()
    mask_c = df_ret_sc.index.year.isin(ANOS_CRISE)
    df_ret_sc.loc[mask_c] = np.nan
    ri_sc = ret_ibov.copy()
    ri_sc.loc[mask_c] = np.nan
    df_res_sc = executar_analise(df_ret_sc, ri_sc, "sem_crises", "EW")

    df_resultados = pd.concat([df_res_ew, df_res_vw, df_res_sc], ignore_index=True)
    df_resultados = adicionar_benchmarking(df_resultados)

    # ---- 5b. Event Study: Fama-French 3 (EW) -----------------------------
    df_ff3 = executar_analise_ff3(df_ret_ew, df_factors, "completo", "EW")

    # ---- 6. Placebo ------------------------------------------------------
    df_placebo = executar_teste_placebo(df_ret_ew, ret_ibov)

    # ---- 6b. DiD (Estatais vs. Privadas, nível empresa) ------------------
    df_did_ind, df_did_agg = executar_did_empresas(df_precos, df_emp, ret_ibov)

    # ---- 7. Volatilidade + Sharpe ----------------------------------------
    df_vol = calcular_volatilidade_comparativa(df_ret_ew)
    df_sharpe = calcular_sharpe_comparativo(df_ret_ew)

    # ---- 8. Visualizações ------------------------------------------------
    log.info("\n" + "="*70)
    log.info("ETAPA 8: VISUALIZAÇÕES")
    log.info("="*70)
    gerar_visualizacoes(df_resultados, df_vol, df_placebo, df_did_agg, df_sharpe, OUTPUT_DIR)

    # ---- 9. Exportação ---------------------------------------------------
    exportar_resultados(df_resultados, df_ff3, df_vol, df_sobrev, df_comp,
                        df_diag, df_placebo, df_did_ind, df_did_agg, df_sharpe, OUTPUT_DIR)

    # ---- Resumo final ----------------------------------------------------
    elapsed = time.time() - t0
    dfc = df_resultados[(df_resultados["cenario"]=="completo") & 
                        (df_resultados["ponderacao"]=="EW")]

    print("\n" + "="*72)
    print("RESUMO FINAL")
    print("="*72)
    print(f"Tempo de execução: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Obs EW completo: {len(dfc)} | VW: {len(df_res_vw)} | FF3: {len(df_ff3)} | Placebo: {len(df_placebo)}")
    print(f"Setores: {dfc['setor'].nunique()} | Anos: {sorted(dfc['ano'].unique())}")

    print("\nCAR Médio por Janela (EW completo):")
    print("-" * 80)
    print(f"{'Janela':<25} {'CAR':>10} {'sig1%':>8} {'sig5%':>8} {'sig10%':>8} {'Silva':>8} {'BMP':>8}")
    print("-" * 80)
    
    for j in sorted(dfc["janela"].unique()):
        d = dfc[dfc["janela"]==j]
        print(f"{j:<25} {d['car'].mean():>+10.4f} "
              f"{d['sig1_simples'].mean()*100:>7.0f}% "
              f"{d['sig5_simples'].mean()*100:>7.0f}% "
              f"{d['sig10_simples'].mean()*100:>7.0f}% "
              f"{d['sig5_silva'].mean()*100:>7.0f}% "
              f"{d['sig5_bmp'].mean()*100:>7.0f}%")

    nok = (df_diag["status"]=="ok").sum()
    print(f"\nCobertura de dados: {nok}/{len(df_diag)} tickers ({100*nok/len(df_diag):.0f}%)")
    
    if not df_ff3.empty:
        print(f"FF3: {len(df_ff3)} observações | sig5: {df_ff3['sig5_ff3'].mean()*100:.0f}%")
    
    if not df_did_agg.empty:
        did_sig = df_did_agg["sig5_did"].sum()
        print(f"\nDiD (Estatais vs Privadas): {len(df_did_agg)} comparações, "
              f"{did_sig} significativas a 5%")
    
    if not df_did_ind.empty:
        n_est = df_did_ind[df_did_ind["estatal"]]["ticker"].nunique()
        n_priv = df_did_ind[~df_did_ind["estatal"]]["ticker"].nunique()
        print(f"  → {n_est} empresas estatais, {n_priv} privadas analisadas")

    print(f"\nArquivos salvos em: {os.path.abspath(OUTPUT_DIR)}/")
    print("="*72)
    print("✓ ANÁLISE COMPLETA FINALIZADA!")
    print("="*72)


if __name__ == "__main__":
    main()
