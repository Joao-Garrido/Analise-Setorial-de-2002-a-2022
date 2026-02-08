"""
================================================================================
IMPACTO DOS CICLOS ELEITORAIS NO MERCADO DE CAPITAIS BRASILEIRO (2002–2022)
Análise de Estudo de Evento com Modelo de Mercado
================================================================================

Referências metodológicas:
  - Silva, W. A. M. et al. (2015). "Evidências de retornos anormais nos
    processos de IPO na BMF&Bovespa no período de 2004 a 2013".
  - Nordhaus, W. D. (1975). "The political business cycle".
    Review of Economic Studies, 42(2), 169-190.
  - Boehmer, E., Musumeci, J. & Poulsen, A. (1991). "Event-study methodology
    under conditions of event-induced variance". Journal of Financial Economics.
  - MacKinlay, A. C. (1997). "Event Studies in Economics and Finance".
    Journal of Economic Literature.

Requisitos:
    pip install pandas numpy openpyxl yfinance statsmodels scipy matplotlib seaborn tqdm

Entrada:
    resultados_analise_b3_com_tickers.xlsx
        → aba "LISTA FINAL (Cont+IPOs-Canc)"

Saída (pasta ./output_v2/):
    analise_ciclos_eleitorais_v2.xlsx   – resultados consolidados
    resultados_consolidados_v2.csv      – CSV para replicação
    tabela_sobrevivencia.csv            – N por setor/ano
    diagnostico_download.csv            – log de tickers obtidos/faltantes
    heatmap_*.png                       – mapas de calor dos CARs
    evolucao_temporal_cars.png          – evolução temporal
    volatilidade_comparativa.png        – eleitoral vs. não-eleitoral
    placebo_test_results.csv            – resultados do teste placebo
    metodologia_limitacoes.md           – texto pronto para artigo

DISCLAIMER:
    A amostra final inclui apenas empresas com dados de preço disponíveis
    no Yahoo Finance, representando o universo investível de empresas
    líquidas. Isso introduz viés de sobrevivência ("survivorship bias"),
    especialmente em anos anteriores a 2006, onde empresas deslistadas,
    falidas ou de baixa capitalização são sub-representadas. Resultados
    devem ser interpretados como evidência consistente com risco político
    — não como impacto causal dos ciclos eleitorais.

    Os p-values reportados utilizam o teste t cross-sectional corrigido
    para autocorrelação (Silva et al., 2015), mas podem ainda superestimar
    a significância em presença de clusterização de volatilidade.
================================================================================
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ===========================================================================
# LOGGING
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ElectoralCycles")

# ===========================================================================
# CONFIGURAÇÕES GLOBAIS
# ===========================================================================

ARQUIVO_ENTRADA = "resultados_analise_b3_com_tickers.xlsx"
SHEET_NAME = "LISTA FINAL (Cont+IPOs-Canc)"
OUTPUT_DIR = "./output_v2"

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

# ---- Setores B3 (conforme dados) -----------------------------------------
SETORES_B3 = [
    "Bens Industriais",
    "Comunicações",
    "Construção e Transporte",
    "Consumo Cíclico",
    "Consumo Não Cíclico",
    "Financeiro",
    "Materiais Básicos",
    "Outros",
    "Petróleo, Gás e Biocombustíveis",
    "Saúde",
    "Utilidade Pública",
]

# ---- Parâmetros de janela -------------------------------------------------
# Janela de estimação: [-252, -30] dias úteis antes do evento (MacKinlay 1997)
ESTIMACAO_INICIO_DU = -252  # dias úteis antes do 1º turno
ESTIMACAO_FIM_DU = -30      # dias úteis antes do 1º turno

# Janelas de evento (dias úteis relativos ao 1º turno)
JANELAS_EVENTO = {
    "antecipacao_45":  (-45, -1),    # 45 d.u. antes do 1º turno
    "antecipacao_60":  (-60, -1),    # Robustez: 60 d.u.
    "reacao_curta":    (-5, +5),     # [-5, +5] em torno do 1º turno
    "reacao_media":    (-10, +10),   # Robustez: [-10, +10]
    "reacao_ampla":    (-20, +20),   # Robustez: [-20, +20]
}
# Janelas adicionais relativas ao 2º turno (calculadas separadamente)
JANELAS_2TURNO = {
    "reacao_2turno":   (-5, +5),
}

# ---- Parâmetros de filtragem e robustez -----------------------------------
MIN_PREGOES_PCT = 0.80        # Mínimo 80% de pregões no ano para inclusão
MIN_EMPRESAS_SETOR = 5        # Corte: mínimo N empresas por setor/ano
ANOS_CRISE = [2008, 2020]     # Para cenário sem crises
N_PLACEBO_EVENTS = 100        # Número de datas aleatórias para teste placebo

# ---- Selic e IPCA anuais (fonte: BCB) – para benchmarking ilustrativo -----
SELIC_ANUAL = {
    2002: 0.1911, 2006: 0.1513, 2010: 0.0975,
    2014: 0.1115, 2018: 0.0640, 2022: 0.1275,
}
IPCA_ANUAL = {
    2002: 0.1253, 2006: 0.0314, 2010: 0.0591,
    2014: 0.0641, 2018: 0.0375, 2022: 0.0562,
}

# ===========================================================================
# DICIONÁRIO DE MAPEAMENTO DE TICKERS (De-Para)
# ===========================================================================
# Tickers que mudaram de código ao longo do tempo. O yfinance
# pode não encontrar o código antigo, mas encontra o novo.
# Formato: {ticker_antigo: ticker_novo}

TICKER_MAPPING = {
    # Varejo / E-commerce
    "VVAR3":  "BHIA3",     # Via Varejo → Casas Bahia
    "BTOW3":  "AMER3",     # B2W Digital → Americanas
    "LAME4":  "LAME3",     # Lojas Americanas (PN→ON antes de crise)
    "PCAR4":  "PCAR3",     # GPA (classe)
    # Educação
    "KROT3":  "COGN3",     # Kroton → Cogna
    "ESTC3":  "YDUQ3",     # Estácio → Yduqs
    # Logística / Infraestrutura
    "RAIL3":  "RUMO3",     # Rumo (reorganização)
    # Bolsa
    "BVMF3":  "B3SA3",     # BM&FBovespa → B3
    "CTIP3":  "B3SA3",     # Cetip → B3 (fusão)
    # Shopping / Imobiliário
    "BRIN3":  "BRML3",     # BR Malls (antes da fusão com Aliansce)
    "BRML3":  "ALOS3",     # BR Malls → Allos
    # Tecnologia
    "SMLE3":  "SMFT3",     # Smartfit
    "LINX3":  "STNE3",     # Linx → StoneCo (proxy)
    "SQIA3":  "SQIA3",     # Sinqia (mantém, depois virou Evertec)
    # Energia
    "ENEV3":  "ENEV3",     # mantém
    # Saneamento
    "SBSP3":  "SBSP3",     # Sabesp (mantém)
    # Telecom
    "OIBR3":  "OIBR3",     # Oi (mantém, em RJ)
    "VIVT4":  "VIVT3",     # Vivt (PN→ON)
    "TIMP3":  "TIMS3",     # TIM Participações → TIM
    # Petróleo
    "QGEP3":  "BRAV3",     # Queiroz Galvão E&P → 3R → Brava
    "RPMG3":  "RPMG3",     # Refinaria (mantém)
    # Saúde
    "GNDI3":  "HAPV3",     # NotreDame → Hapvida (fusão)
    "QUAL3":  "QUAL3",     # Qualicorp (mantém)
    # Construção
    "PDGR3":  "PDGR3",     # PDG (mantém, RJ)
    "GFSA3":  "GFSA3",     # Gafisa (mantém)
    # Mineração / Siderurgia
    "MMXM3":  "MMXM3",     # MMX (mantém)
    "USIM5":  "USIM5",     # Usiminas (mantém)
    # Aviação
    "GOLL4":  "GOLL4",     # Gol (mantém)
    # Financeiro
    "BPAN4":  "BPAN4",     # Banco Pan (mantém)
    "SANB11": "SANB11",    # Santander (mantém)
    # Alimentos
    "MRFG3":  "MRFG3",     # Marfrig (mantém)
    "MDIA3":  "MDIA3",     # M. Dias Branco (mantém)
    # Papel e Celulose
    "FIBR3":  "SUZB3",     # Fibria → Suzano (fusão)
    # Seguros
    "PSSA3":  "PSSA3",     # Porto Seguro (mantém)
    "SULA11": "SULA11",    # SulAmérica (mantém, depois fusão com Rede D'Or)
}


# ===========================================================================
# ETAPA 1 – INGESTÃO E TRATAMENTO DE DADOS
# ===========================================================================

def carregar_lista_empresas(caminho: str) -> pd.DataFrame:
    """
    Carrega a lista de empresas, aplica mapeamento de tickers
    e prepara para download.
    """
    log.info("=" * 70)
    log.info("ETAPA 1: INGESTÃO DE DADOS")
    log.info("=" * 70)
    log.info("Carregando %s ...", caminho)

    df = pd.read_excel(caminho, sheet_name=SHEET_NAME)
    df = df.dropna(subset=["TICKER", "SETOR_B3"])
    df["DT_REG"] = pd.to_datetime(df["DT_REG"], errors="coerce")
    df["DT_CANCEL"] = pd.to_datetime(df["DT_CANCEL"], errors="coerce")

    # Ticker original para referência
    df["TICKER_ORIGINAL"] = df["TICKER"].str.strip()

    # Aplica mapeamento De-Para
    df["TICKER_MAPEADO"] = df["TICKER_ORIGINAL"].map(TICKER_MAPPING).fillna(
        df["TICKER_ORIGINAL"]
    )

    # Sufixo .SA para yfinance
    df["TICKER_YF"] = df["TICKER_MAPEADO"] + ".SA"

    n_mapeados = (df["TICKER_ORIGINAL"] != df["TICKER_MAPEADO"]).sum()
    log.info("  → %d empresas carregadas, %d setores", len(df), df["SETOR_B3"].nunique())
    log.info("  → %d tickers remapeados via De-Para", n_mapeados)

    return df


def baixar_precos_yfinance(tickers: list, start: str = "2000-01-01",
                           end: str = "2023-12-31") -> tuple:
    """
    Baixa preços ajustados via yfinance.
    Retorna (df_precos, diagnostico_download).
    """
    import yfinance as yf

    log.info("Baixando preços de %d tickers únicos via yfinance ...", len(tickers))
    log.info("  Período: %s a %s", start, end)

    all_data = {}
    tickers_ok = []
    tickers_falha = []

    blocos = [tickers[i:i + 50] for i in range(0, len(tickers), 50)]
    for idx, bloco in enumerate(blocos):
        log.info("  Bloco %d/%d (%d tickers) ...", idx + 1, len(blocos), len(bloco))
        try:
            data = yf.download(
                bloco, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )
            if data.empty:
                tickers_falha.extend(bloco)
                continue

            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data[["Close"]]
                close.columns = bloco

            for col in close.columns:
                if close[col].notna().sum() > 0:
                    all_data[col] = close[col]
                    tickers_ok.append(col)
                else:
                    tickers_falha.append(col)

            # Tickers no bloco que não apareceram
            for t in bloco:
                if t not in close.columns and t not in tickers_ok:
                    tickers_falha.append(t)

        except Exception as e:
            log.warning("  Erro no bloco %d: %s", idx + 1, e)
            tickers_falha.extend(bloco)
        time.sleep(0.3)

    # Deduplica
    tickers_falha = list(set(tickers_falha) - set(tickers_ok))

    df_precos = pd.DataFrame(all_data)
    df_precos.index = pd.to_datetime(df_precos.index)

    log.info("  → Sucesso: %d tickers | Falha: %d tickers (%.1f%% de perda)",
             len(tickers_ok), len(tickers_falha),
             100 * len(tickers_falha) / max(len(tickers), 1))

    # Diagnóstico
    diagnostico = pd.DataFrame({
        "ticker_yf": tickers_ok + tickers_falha,
        "status": ["ok"] * len(tickers_ok) + ["falha"] * len(tickers_falha),
    })

    return df_precos, diagnostico


def baixar_ibovespa(start: str = "2000-01-01", end: str = "2023-12-31") -> pd.Series:
    """Baixa o Ibovespa (^BVSP)."""
    import yfinance as yf
    log.info("Baixando Ibovespa (^BVSP) ...")
    ibov = yf.download("^BVSP", start=start, end=end, auto_adjust=True, progress=False)
    if ibov.empty:
        raise RuntimeError("Não foi possível baixar o Ibovespa.")
    serie = ibov["Close"].squeeze()
    serie.index = pd.to_datetime(serie.index)
    serie.name = "IBOV"
    log.info("  → %d observações", len(serie))
    return serie


def aplicar_filtro_existencia(df_precos: pd.DataFrame,
                              df_empresas: pd.DataFrame) -> pd.DataFrame:
    """
    Invalida preços fora do período de vida da empresa
    (antes de DT_REG ou após DT_CANCEL).
    """
    log.info("Aplicando filtro de existência ...")
    df = df_precos.copy()

    # Mapa: TICKER_YF → (DT_REG, DT_CANCEL)
    lookup = {}
    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        if tk in lookup:
            # Se ticker aparece mais de uma vez, usa a mais antiga DT_REG
            old_reg, old_cancel = lookup[tk]
            new_reg = row["DT_REG"]
            new_cancel = row["DT_CANCEL"]
            reg = min(r for r in [old_reg, new_reg] if pd.notna(r)) if any(pd.notna(r) for r in [old_reg, new_reg]) else pd.NaT
            cancel = max(c for c in [old_cancel, new_cancel] if pd.notna(c)) if any(pd.notna(c) for c in [old_cancel, new_cancel]) else pd.NaT
            lookup[tk] = (reg, cancel)
        else:
            lookup[tk] = (row["DT_REG"], row["DT_CANCEL"])

    for col in df.columns:
        if col in lookup:
            dt_reg, dt_cancel = lookup[col]
            if pd.notna(dt_reg):
                df.loc[df.index < dt_reg, col] = np.nan
            if pd.notna(dt_cancel):
                df.loc[df.index > dt_cancel, col] = np.nan

    return df


def aplicar_filtro_liquidez(df_precos: pd.DataFrame, ano: int,
                            min_pct: float = MIN_PREGOES_PCT) -> list:
    """
    Retorna lista de tickers com >= min_pct dos pregões no ano.
    Filtro de liquidez: empresas com dados em >= 80% dos dias de negociação.
    """
    mask_ano = df_precos.index.year == ano
    precos_ano = df_precos.loc[mask_ano]
    n_pregoes = precos_ano.notna().sum()
    n_total = mask_ano.sum()
    pct = n_pregoes / n_total
    return pct[pct >= min_pct].index.tolist()


# ===========================================================================
# ETAPA 2 – ÍNDICES SETORIAIS EQUAL-WEIGHTED
# ===========================================================================

def construir_indices_setoriais(df_precos: pd.DataFrame,
                                df_empresas: pd.DataFrame) -> tuple:
    """
    Constrói índices setoriais equal-weighted com filtro de liquidez.
    Retorna (df_precos_setores, df_ret_setores, tabela_composicao).
    """
    log.info("=" * 70)
    log.info("ETAPA 2: CONSTRUÇÃO DE ÍNDICES SETORIAIS EQUAL-WEIGHTED")
    log.info("=" * 70)

    ret = np.log(df_precos / df_precos.shift(1))

    # Mapeamento ticker → setor (usa TICKER_YF)
    t2s = {}
    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        t2s[tk] = row["SETOR_B3"]

    indices_ret = {}
    composicao_registros = []

    for setor in SETORES_B3:
        cols_setor = [c for c in ret.columns if t2s.get(c) == setor]
        if len(cols_setor) == 0:
            log.warning("  Nenhum ticker disponível para '%s'", setor)
            continue

        ret_setor = ret[cols_setor].mean(axis=1)
        indices_ret[setor] = ret_setor

        # Registra composição por ano
        for ano in range(2001, 2024):
            mask_ano = ret.index.year == ano
            if mask_ano.sum() == 0:
                continue
            n_ativos = ret.loc[mask_ano, cols_setor].notna().any().sum()
            n_pregoes_medio = ret.loc[mask_ano, cols_setor].notna().sum(axis=1).mean()
            composicao_registros.append({
                "setor": setor,
                "ano": ano,
                "n_tickers_setor_total": len(cols_setor),
                "n_tickers_com_dados": n_ativos,
                "media_ativos_por_dia": round(n_pregoes_medio, 1),
            })

        log.info("  %s: %d tickers com dados", setor, len(cols_setor))

    df_ret_setores = pd.DataFrame(indices_ret)

    # Preços em nível (base 100)
    df_precos_setores = pd.DataFrame(index=df_ret_setores.index)
    for col in df_ret_setores.columns:
        cum = df_ret_setores[col].fillna(0).cumsum()
        df_precos_setores[col] = 100 * np.exp(cum)

    df_composicao = pd.DataFrame(composicao_registros)
    return df_precos_setores, df_ret_setores, df_composicao


def gerar_tabela_sobrevivencia(df_empresas: pd.DataFrame,
                               df_precos: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela N de empresas com dados de preço disponíveis por setor e ano.
    Essencial para transparência sobre viés de sobrevivência.
    """
    log.info("Gerando tabela de sobrevivência (N por setor/ano) ...")

    t2s = {}
    for _, row in df_empresas.iterrows():
        t2s[row["TICKER_YF"]] = row["SETOR_B3"]

    registros = []
    for ano in range(2001, 2024):
        mask = df_precos.index.year == ano
        if mask.sum() == 0:
            continue
        precos_ano = df_precos.loc[mask]
        for setor in SETORES_B3:
            cols = [c for c in precos_ano.columns if t2s.get(c) == setor]
            n_com_dados = precos_ano[cols].notna().any().sum() if cols else 0
            # Empresas que deveriam existir (por DT_REG/DT_CANCEL)
            mask_emp = (
                (df_empresas["SETOR_B3"] == setor) &
                (df_empresas["DT_REG"].dt.year <= ano) &
                ((df_empresas["DT_CANCEL"].isna()) | (df_empresas["DT_CANCEL"].dt.year >= ano))
            )
            n_esperado = mask_emp.sum()
            registros.append({
                "ano": ano,
                "setor": setor,
                "n_empresas_esperado": n_esperado,
                "n_empresas_com_dados": n_com_dados,
                "cobertura_pct": round(100 * n_com_dados / max(n_esperado, 1), 1),
            })

    return pd.DataFrame(registros)


# ===========================================================================
# ETAPA 3 – DEFINIÇÃO DE JANELAS (COM JANELA MÓVEL DE ESTIMAÇÃO)
# ===========================================================================

def obter_dias_uteis_offset(bdates: pd.DatetimeIndex, data_ref: pd.Timestamp,
                            offset: int) -> pd.Timestamp:
    """
    Retorna a data correspondente a `offset` dias úteis a partir de data_ref.
    """
    bdays = bdates.sort_values()
    pos = bdays.searchsorted(data_ref)
    pos = min(pos, len(bdays) - 1)
    target = pos + offset
    target = max(0, min(target, len(bdays) - 1))
    return bdays[target]


def definir_janela_estimacao(bdates: pd.DatetimeIndex,
                             dt_evento: pd.Timestamp) -> tuple:
    """
    Janela de estimação móvel: [evento - 252 d.u., evento - 30 d.u.]
    Conforme MacKinlay (1997) e padrão de event study.
    """
    est_ini = obter_dias_uteis_offset(bdates, dt_evento, ESTIMACAO_INICIO_DU)
    est_fim = obter_dias_uteis_offset(bdates, dt_evento, ESTIMACAO_FIM_DU)
    return (est_ini, est_fim)


def definir_janelas_evento(bdates: pd.DatetimeIndex, ano: int) -> dict:
    """
    Define todas as janelas de evento para um dado ano eleitoral.
    Retorna dict: {nome_janela: (data_inicio, data_fim)}.
    """
    dt_1t = DATAS_PRIMEIRO_TURNO[ano]
    dt_2t = DATAS_SEGUNDO_TURNO[ano]

    janelas = {}

    # Janelas relativas ao 1º turno
    for nome, (di, df_off) in JANELAS_EVENTO.items():
        j_ini = obter_dias_uteis_offset(bdates, dt_1t, di)
        j_fim = obter_dias_uteis_offset(bdates, dt_1t, df_off)
        janelas[nome] = (j_ini, j_fim)

    # Janelas relativas ao 2º turno
    for nome, (di, df_off) in JANELAS_2TURNO.items():
        j_ini = obter_dias_uteis_offset(bdates, dt_2t, di)
        j_fim = obter_dias_uteis_offset(bdates, dt_2t, df_off)
        janelas[nome] = (j_ini, j_fim)

    # Ciclo interno: 1º sem vs 2º sem
    janelas["ciclo_1sem"] = (
        pd.Timestamp(f"{ano}-01-02"),
        pd.Timestamp(f"{ano}-06-30"),
    )
    janelas["ciclo_2sem"] = (
        pd.Timestamp(f"{ano}-07-01"),
        pd.Timestamp(f"{ano}-12-31"),
    )

    # Estendida: últimos 6 meses
    janelas["estendida"] = (
        pd.Timestamp(f"{ano}-07-01"),
        pd.Timestamp(f"{ano}-12-31"),
    )

    return janelas


# ===========================================================================
# ETAPA 4 – MODELO DE MERCADO E INFERÊNCIA ESTATÍSTICA
# ===========================================================================

def estimar_ols(ret_ativo: pd.Series, ret_mercado: pd.Series) -> dict:
    """
    Estima R_i,t = α + β·R_m,t + ε_t via OLS.
    Retorna dict com alpha, beta, sigma_resid, r_squared, n_obs
    ou None se dados insuficientes.
    """
    df = pd.DataFrame({"y": ret_ativo, "x": ret_mercado}).dropna()
    if len(df) < 30:
        return None

    X = np.column_stack([np.ones(len(df)), df["x"].values])
    Y = df["y"].values

    try:
        coef, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
        Y_hat = X @ coef
        resid = Y - Y_hat
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        sigma = np.sqrt(ss_res / (len(df) - 2)) if len(df) > 2 else np.nan
        return {
            "alpha": coef[0],
            "beta": coef[1],
            "sigma_resid": sigma,
            "r_squared": r2,
            "n_obs_estimacao": len(df),
        }
    except Exception:
        return None


def calcular_ar(ret_ativo: pd.Series, ret_mercado: pd.Series,
                alpha: float, beta: float) -> pd.Series:
    """AR_t = R_{i,t} − (α + β · R_{m,t})"""
    df = pd.DataFrame({"y": ret_ativo, "x": ret_mercado}).dropna()
    ar = df["y"] - (alpha + beta * df["x"])
    return ar


def calcular_car(ar_series: pd.Series) -> float:
    """CAR = Σ_t AR_t"""
    return ar_series.sum()


def tstat_car_silva(car_values: np.ndarray, ar_series_list: list,
                    n_dias: int) -> dict:
    """
    Teste t para CAR com desvio padrão corrigido por autocorrelação
    (Silva et al., 2015, Equações 6-8).

    csd_t = sqrt(t · var_média + 2·(t−1) · cov_média)

    onde var_média e cov_média são calculados a partir das séries
    longitudinais de AR de cada setor/ativo.

    Parâmetros:
        car_values: array de CARs (um por setor ou ativo)
        ar_series_list: lista de pd.Series de ARs diários
        n_dias: número de dias na janela de evento

    Retorna: {t_stat, p_value, csd_t, n}
    """
    n = len(car_values)
    if n < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "csd_t": np.nan, "n": n}

    car_mean = np.mean(car_values)

    # Variâncias longitudinais e covariâncias de primeira ordem
    variances = []
    covariances = []
    for ar_s in ar_series_list:
        arr = ar_s.values if hasattr(ar_s, "values") else np.array(ar_s)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            continue
        variances.append(np.var(arr, ddof=1))
        if len(arr) > 2:
            cov_matrix = np.cov(arr[:-1], arr[1:])
            covariances.append(cov_matrix[0, 1])

    if len(variances) == 0:
        return {"t_stat": np.nan, "p_value": np.nan, "csd_t": np.nan, "n": n}

    var_mean = np.mean(variances)
    cov_mean = np.mean(covariances) if covariances else 0.0

    t = max(n_dias, 1)
    csd_sq = t * var_mean + 2 * max(t - 1, 0) * cov_mean
    if csd_sq <= 0:
        return {"t_stat": np.nan, "p_value": np.nan, "csd_t": np.nan, "n": n}

    csd_t = np.sqrt(csd_sq)
    t_stat = car_mean * np.sqrt(n) / csd_t
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), max(n - 1, 1)))

    return {"t_stat": t_stat, "p_value": p_value, "csd_t": csd_t, "n": n}


def tstat_car_bmp(car_values: np.ndarray, sigma_resids: np.ndarray,
                  n_dias: int) -> dict:
    """
    Teste de Boehmer-Musumeci-Poulsen (BMP, 1991).
    Standardized Cross-Sectional test.

    SCAR_i = CAR_i / (sigma_i * sqrt(T_evento))
    t_BMP = mean(SCAR) / (std(SCAR) / sqrt(N))
    """
    n = len(car_values)
    if n < 2:
        return {"t_stat_bmp": np.nan, "p_value_bmp": np.nan, "n": n}

    # Standardize
    scar = np.zeros(n)
    for i in range(n):
        denom = sigma_resids[i] * np.sqrt(max(n_dias, 1))
        scar[i] = car_values[i] / denom if denom > 0 else np.nan

    scar = scar[~np.isnan(scar)]
    if len(scar) < 2:
        return {"t_stat_bmp": np.nan, "p_value_bmp": np.nan, "n": len(scar)}

    scar_mean = np.mean(scar)
    scar_std = np.std(scar, ddof=1)
    if scar_std == 0:
        return {"t_stat_bmp": np.nan, "p_value_bmp": np.nan, "n": len(scar)}

    t_bmp = scar_mean * np.sqrt(len(scar)) / scar_std
    p_bmp = 2 * (1 - stats.t.cdf(abs(t_bmp), len(scar) - 1))

    return {"t_stat_bmp": t_bmp, "p_value_bmp": p_bmp, "n": len(scar)}


# ===========================================================================
# ETAPA 5 – EXECUÇÃO DA ANÁLISE PRINCIPAL
# ===========================================================================

def executar_analise(df_ret_setores: pd.DataFrame,
                     ret_ibov: pd.Series,
                     cenario: str = "completo") -> pd.DataFrame:
    """
    Executa Event Study completo para todos setores × anos × janelas.
    Usa janela de estimação móvel [-252, -30] d.u. antes do 1º turno.
    """
    log.info("=" * 70)
    log.info("ETAPA 5: EVENT STUDY – Cenário '%s'", cenario)
    log.info("=" * 70)

    bdates = df_ret_setores.dropna(how="all").index
    resultados = []

    for ano in ANOS_ELEITORAIS:
        log.info("  Ano %d ...", ano)
        dt_1t = DATAS_PRIMEIRO_TURNO[ano]

        # Janela de estimação móvel
        est_ini, est_fim = definir_janela_estimacao(bdates, dt_1t)
        log.info("    Estimação: %s a %s", est_ini.date(), est_fim.date())

        # Janelas de evento
        janelas = definir_janelas_evento(bdates, ano)

        for setor in df_ret_setores.columns:
            # Retornos na janela de estimação
            mask_est = (df_ret_setores.index >= est_ini) & (df_ret_setores.index <= est_fim)
            ret_s_est = df_ret_setores.loc[mask_est, setor].dropna()
            ret_m_est = ret_ibov.loc[mask_est].dropna()

            params = estimar_ols(ret_s_est, ret_m_est)
            if params is None:
                continue

            alpha = params["alpha"]
            beta = params["beta"]

            for nome_janela, (j_ini, j_fim) in janelas.items():
                mask_j = (df_ret_setores.index >= j_ini) & (df_ret_setores.index <= j_fim)
                ret_s_j = df_ret_setores.loc[mask_j, setor].dropna()
                ret_m_j = ret_ibov.loc[mask_j].dropna()

                if len(ret_s_j) < 3:
                    continue

                ar = calcular_ar(ret_s_j, ret_m_j, alpha, beta)
                car = calcular_car(ar)
                n_dias = len(ar)

                # t-stat simples (longitudinal, 1 série)
                ar_std = ar.std()
                t_simple = (ar.mean() * np.sqrt(n_dias)) / ar_std if ar_std > 0 else np.nan
                p_simple = 2 * (1 - stats.t.cdf(abs(t_simple), max(n_dias - 1, 1))) if not np.isnan(t_simple) else np.nan

                resultados.append({
                    "cenario": cenario,
                    "ano": ano,
                    "setor": setor,
                    "janela": nome_janela,
                    "car": car,
                    "ar_medio": ar.mean(),
                    "ar_std": ar_std,
                    "n_dias": n_dias,
                    "alpha": alpha,
                    "beta": beta,
                    "r_squared": params["r_squared"],
                    "sigma_resid": params["sigma_resid"],
                    "n_obs_estimacao": params["n_obs_estimacao"],
                    "est_inicio": est_ini,
                    "est_fim": est_fim,
                    "janela_inicio": j_ini,
                    "janela_fim": j_fim,
                    "t_stat_simples": t_simple,
                    "p_value_simples": p_simple,
                    # Cross-sectional tests serão preenchidos depois
                    "t_stat_silva": np.nan,
                    "p_value_silva": np.nan,
                    "t_stat_bmp": np.nan,
                    "p_value_bmp": np.nan,
                })

    df_res = pd.DataFrame(resultados)

    # ------------------------------------------------------------------
    # Cross-sectional tests (Silva et al. e BMP) por ano × janela
    # ------------------------------------------------------------------
    if not df_res.empty:
        log.info("  Calculando testes cross-sectional (Silva et al. / BMP) ...")
        for (ano, janela), grp in df_res.groupby(["ano", "janela"]):
            cars = grp["car"].values
            sigmas = grp["sigma_resid"].values
            n_dias = int(grp["n_dias"].median())

            # Reconstruir séries de AR para cada setor (para Silva)
            ar_lists = []
            for _, row in grp.iterrows():
                # Recalcula AR para obter a série
                s = row["setor"]
                j_ini_dt = row["janela_inicio"]
                j_fim_dt = row["janela_fim"]
                mask_j = (df_ret_setores.index >= j_ini_dt) & (df_ret_setores.index <= j_fim_dt)
                ret_s = df_ret_setores.loc[mask_j, s].dropna()
                ret_m = ret_ibov.loc[mask_j].dropna()
                ar_s = calcular_ar(ret_s, ret_m, row["alpha"], row["beta"])
                ar_lists.append(ar_s)

            # Silva et al.
            res_silva = tstat_car_silva(cars, ar_lists, n_dias)
            # BMP
            res_bmp = tstat_car_bmp(cars, sigmas, n_dias)

            # Preenche no DataFrame
            mask_grp = (df_res["ano"] == ano) & (df_res["janela"] == janela) & (df_res["cenario"] == cenario)
            df_res.loc[mask_grp, "t_stat_silva"] = res_silva["t_stat"]
            df_res.loc[mask_grp, "p_value_silva"] = res_silva["p_value"]
            df_res.loc[mask_grp, "t_stat_bmp"] = res_bmp.get("t_stat_bmp", np.nan)
            df_res.loc[mask_grp, "p_value_bmp"] = res_bmp.get("p_value_bmp", np.nan)

    # Significância
    df_res["sig_5pct_simples"] = df_res["p_value_simples"] < 0.05
    df_res["sig_10pct_simples"] = df_res["p_value_simples"] < 0.10
    df_res["sig_5pct_silva"] = df_res["p_value_silva"] < 0.05
    df_res["sig_5pct_bmp"] = df_res["p_value_bmp"] < 0.05

    log.info("  → %d observações", len(df_res))
    return df_res


# ===========================================================================
# ETAPA 6 – TESTE PLACEBO (PSEUDO-EVENTS)
# ===========================================================================

def executar_teste_placebo(df_ret_setores: pd.DataFrame,
                           ret_ibov: pd.Series,
                           n_placebos: int = N_PLACEBO_EVENTS,
                           seed: int = 42) -> pd.DataFrame:
    """
    Gera N datas aleatórias (não-eleitorais) e calcula CARs.
    Compara distribuição de CARs placebo vs. eleitorais para
    validar que os resultados não são artefatos.
    """
    log.info("=" * 70)
    log.info("ETAPA 6: TESTE PLACEBO (%d pseudo-eventos)", n_placebos)
    log.info("=" * 70)

    np.random.seed(seed)
    bdates = df_ret_setores.dropna(how="all").index

    # Gera datas aleatórias entre 2003 e 2021 (fora de anos eleitorais)
    anos_nao_eleitorais = [a for a in range(2003, 2022) if a not in ANOS_ELEITORAIS]
    datas_placebo = []
    for _ in range(n_placebos):
        ano = np.random.choice(anos_nao_eleitorais)
        # Simula uma "eleição" no primeiro domingo de outubro
        dt_fake = pd.Timestamp(f"{ano}-10-05")
        datas_placebo.append((ano, dt_fake))

    resultados = []
    for i, (ano, dt_evento) in enumerate(datas_placebo):
        est_ini = obter_dias_uteis_offset(bdates, dt_evento, ESTIMACAO_INICIO_DU)
        est_fim = obter_dias_uteis_offset(bdates, dt_evento, ESTIMACAO_FIM_DU)

        # Só usa janela de antecipação [-45, -1] e reação [-5, +5]
        ant_ini = obter_dias_uteis_offset(bdates, dt_evento, -45)
        ant_fim = obter_dias_uteis_offset(bdates, dt_evento, -1)
        rea_ini = obter_dias_uteis_offset(bdates, dt_evento, -5)
        rea_fim = obter_dias_uteis_offset(bdates, dt_evento, +5)

        for setor in df_ret_setores.columns:
            mask_est = (df_ret_setores.index >= est_ini) & (df_ret_setores.index <= est_fim)
            ret_s_est = df_ret_setores.loc[mask_est, setor].dropna()
            ret_m_est = ret_ibov.loc[mask_est].dropna()
            params = estimar_ols(ret_s_est, ret_m_est)
            if params is None:
                continue

            for nome, (ji, jf) in [("placebo_antecipacao", (ant_ini, ant_fim)),
                                     ("placebo_reacao", (rea_ini, rea_fim))]:
                mask_j = (df_ret_setores.index >= ji) & (df_ret_setores.index <= jf)
                ret_s = df_ret_setores.loc[mask_j, setor].dropna()
                ret_m = ret_ibov.loc[mask_j].dropna()
                if len(ret_s) < 3:
                    continue
                ar = calcular_ar(ret_s, ret_m, params["alpha"], params["beta"])
                car = calcular_car(ar)
                resultados.append({
                    "placebo_id": i,
                    "ano_placebo": ano,
                    "setor": setor,
                    "janela": nome,
                    "car": car,
                    "n_dias": len(ar),
                })

    df_placebo = pd.DataFrame(resultados)
    log.info("  → %d observações placebo geradas", len(df_placebo))
    return df_placebo


# ===========================================================================
# ETAPA 7 – VOLATILIDADE E BENCHMARKING
# ===========================================================================

def calcular_volatilidade_comparativa(df_ret_setores: pd.DataFrame) -> pd.DataFrame:
    """Volatilidade anualizada: anos eleitorais vs. não-eleitorais."""
    log.info("Calculando volatilidade comparativa ...")
    registros = []
    for setor in df_ret_setores.columns:
        for ano in range(2002, 2023):
            mask = df_ret_setores.index.year == ano
            ret = df_ret_setores.loc[mask, setor].dropna()
            if len(ret) < 20:
                continue
            vol = ret.std() * np.sqrt(252)
            tipo = "Eleitoral" if ano in ANOS_ELEITORAIS else "Não-Eleitoral"
            registros.append({
                "setor": setor, "ano": ano, "tipo_ano": tipo,
                "volatilidade_anualizada": vol,
            })
    return pd.DataFrame(registros)


def adicionar_benchmarking(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona Selic e IPCA proporcionais para benchmarking ilustrativo.
    NOTA: Isto NÃO é um teste formal; serve apenas como referência
    econômica para contextualizar a magnitude dos CARs.
    """
    df = df_res.copy()
    df["selic_anual"] = df["ano"].map(SELIC_ANUAL)
    df["ipca_anual"] = df["ano"].map(IPCA_ANUAL)
    df["selic_proporcional"] = df["selic_anual"] * df["n_dias"] / 252
    df["ipca_proporcional"] = df["ipca_anual"] * df["n_dias"] / 252
    df["car_excesso_selic"] = df["car"] - df["selic_proporcional"]
    df["car_excesso_ipca"] = df["car"] - df["ipca_proporcional"]
    return df


# ===========================================================================
# ETAPA 8 – VISUALIZAÇÕES
# ===========================================================================

def gerar_visualizacoes(df_resultados: pd.DataFrame,
                        df_volatilidade: pd.DataFrame,
                        df_placebo: pd.DataFrame,
                        output_dir: str):
    """Gera todos os gráficos de nível acadêmico."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": (16, 9),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    df = df_resultados[df_resultados["cenario"] == "completo"]

    # ---- HEATMAPS --------------------------------------------------------
    janelas_hm = {
        "antecipacao_45": "Antecipação Eleitoral (−45 a −1 d.u. do 1º Turno)",
        "reacao_curta": "Reação Imediata – 1º Turno [−5, +5] d.u.",
        "reacao_2turno": "Reação Imediata – 2º Turno [−5, +5] d.u.",
        "reacao_media": "Reação Média [−10, +10] d.u. (Robustez)",
        "reacao_ampla": "Reação Ampla [−20, +20] d.u. (Robustez)",
        "antecipacao_60": "Antecipação Estendida (−60 a −1 d.u.) (Robustez)",
        "estendida": "Últimos 6 Meses do Ano Eleitoral",
    }

    for janela_key, titulo in janelas_hm.items():
        df_j = df[df["janela"] == janela_key]
        if df_j.empty:
            continue

        pivot = df_j.pivot_table(index="setor", columns="ano", values="car", aggfunc="mean")
        if pivot.empty:
            continue

        # Marca células significativas
        pivot_sig = df_j.pivot_table(index="setor", columns="ano",
                                     values="sig_5pct_simples", aggfunc="any")

        fig, ax = plt.subplots(figsize=(15, 8))

        # Anotações com asteriscos para significância
        annot_matrix = pivot.copy().astype(str)
        for r in annot_matrix.index:
            for c in annot_matrix.columns:
                val = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
                sig = pivot_sig.loc[r, c] if r in pivot_sig.index and c in pivot_sig.columns else False
                if pd.notna(val):
                    star = "*" if sig else ""
                    annot_matrix.loc[r, c] = f"{val:.3f}{star}"
                else:
                    annot_matrix.loc[r, c] = ""

        sns.heatmap(
            pivot, annot=annot_matrix, fmt="", cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor="gray",
            cbar_kws={"label": "CAR (Equal-Weighted)"},
            ax=ax,
        )
        ax.set_title(
            f"CAR por Setor e Ano – {titulo}\n"
            f"(Índice Setorial Equal-Weighted · * p < 0.05)",
            fontweight="bold", fontsize=12,
        )
        ax.set_xlabel("Ano Eleitoral")
        ax.set_ylabel("Setor B3")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{janela_key}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ heatmap_%s.png", janela_key)

    # ---- EVOLUÇÃO TEMPORAL -----------------------------------------------
    df_ant = df[df["janela"] == "antecipacao_45"]
    if not df_ant.empty:
        magnitude = df_ant.groupby("setor")["car"].apply(lambda x: abs(x).mean())
        top5 = magnitude.nlargest(5).index.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for idx, (janela, nome) in enumerate([
            ("antecipacao_45", "Antecipação (−45 d.u.)"),
            ("reacao_curta", "Reação 1º Turno [−5,+5]"),
        ]):
            ax = axes[idx]
            for setor in top5:
                dados = df[(df["setor"] == setor) & (df["janela"] == janela)].sort_values("ano")
                if not dados.empty:
                    ax.plot(dados["ano"], dados["car"], marker="o", linewidth=2,
                            label=setor, markersize=7)
            ax.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax.set_xlabel("Ano Eleitoral")
            ax.set_ylabel("CAR (Equal-Weighted)")
            ax.set_title(f"Top 5 Setores – {nome}", fontweight="bold")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            "Evolução Temporal dos CARs Setoriais (Equal-Weighted)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "evolucao_temporal_cars.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ evolucao_temporal_cars.png")

    # ---- VOLATILIDADE COMPARATIVA ----------------------------------------
    if not df_volatilidade.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        vol_agg = df_volatilidade.groupby(
            ["setor", "tipo_ano"])["volatilidade_anualizada"].mean().reset_index()
        pivot_vol = vol_agg.pivot(index="setor", columns="tipo_ano",
                                  values="volatilidade_anualizada")
        pivot_vol.plot(kind="barh", ax=ax, alpha=0.85, edgecolor="black")
        ax.set_title(
            "Volatilidade Anualizada Média: Eleitoral vs. Não-Eleitoral\n"
            "(Índices Setoriais Equal-Weighted · 2002–2022)",
            fontweight="bold",
        )
        ax.set_xlabel("Volatilidade Anualizada (σ · √252)")
        ax.set_ylabel("")
        ax.legend(title="Tipo de Ano")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volatilidade_comparativa.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ volatilidade_comparativa.png")

    # ---- PLACEBO vs. REAL ------------------------------------------------
    if not df_placebo.empty and not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (janela_real, janela_placebo, titulo) in enumerate([
            ("antecipacao_45", "placebo_antecipacao", "Antecipação"),
            ("reacao_curta", "placebo_reacao", "Reação [−5,+5]"),
        ]):
            ax = axes[idx]
            cars_real = df[df["janela"] == janela_real]["car"].dropna()
            cars_placebo = df_placebo[df_placebo["janela"] == janela_placebo]["car"].dropna()

            if len(cars_placebo) > 0:
                ax.hist(cars_placebo, bins=30, alpha=0.5, label="Placebo",
                        color="gray", density=True, edgecolor="black")
            if len(cars_real) > 0:
                ax.hist(cars_real, bins=15, alpha=0.7, label="Eleições Reais",
                        color="steelblue", density=True, edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"Distribuição de CARs – {titulo}", fontweight="bold")
            ax.set_xlabel("CAR")
            ax.set_ylabel("Densidade")
            ax.legend()
        plt.suptitle(
            "Teste Placebo: CARs Eleitorais vs. Pseudo-Eventos Aleatórios",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "placebo_test.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ placebo_test.png")


# ===========================================================================
# ETAPA 9 – EXPORTAÇÃO
# ===========================================================================

def gerar_texto_metodologia(output_dir: str):
    """Gera texto de metodologia e limitações pronto para artigo."""
    texto = """# Metodologia e Limitações

## 1. Desenho do Estudo

Este trabalho emprega a metodologia de **Estudo de Evento** (Event Study), conforme
formalizada por MacKinlay (1997) e adaptada por Silva et al. (2015) para o mercado
brasileiro. O evento de interesse é o ciclo eleitoral presidencial brasileiro, analisado
em seis eleições: 2002, 2006, 2010, 2014, 2018 e 2022.

### 1.1 Construção dos Índices Setoriais

Para cada um dos 11 setores da classificação oficial da B3, construímos um **índice
equal-weighted** (pesos iguais) com base nos retornos logarítmicos diários de todas
as empresas ativas em cada dia de negociação. A ponderação igual visa isolar o
comportamento médio do setor, diluindo riscos idiossincráticos de empresas individuais.

**Nota:** A ponderação equal-weighted tende a amplificar a contribuição de empresas
de menor capitalização. Resultados com ponderação por valor de mercado podem diferir.

### 1.2 Janela de Estimação

A janela de estimação compreende os **[-252, -30] dias úteis** antes da data do
1º turno de cada eleição, totalizando aproximadamente 222 dias úteis (~10,5 meses).
Esta janela móvel evita contaminação por choques que ocorram imediatamente antes
do evento e é consistente com a recomendação de MacKinlay (1997).

### 1.3 Modelo de Mercado

O retorno normal esperado é estimado via regressão OLS do modelo de mercado:

    R_{i,t} = α_i + β_i · R_{m,t} + ε_{i,t}

onde R_{m,t} é o retorno do Ibovespa (^BVSP). O retorno anormal (AR) na janela de
evento é:

    AR_{i,t} = R_{i,t} − (α̂_i + β̂_i · R_{m,t})

O retorno anormal acumulado (CAR) é a soma dos ARs na janela.

### 1.4 Janelas de Evento

| Janela | Definição | Motivação |
|---|---|---|
| Antecipação (45 d.u.) | [−45, −1] antes do 1º turno | Efeito Propaganda / Pricing-in |
| Antecipação (60 d.u.) | [−60, −1] – robustez | Sensibilidade à largura |
| Reação Curta 1º Turno | [−5, +5] em torno do 1º turno | Reação imediata |
| Reação Média | [−10, +10] – robustez | Sensibilidade à largura |
| Reação Ampla | [−20, +20] – robustez | Persistência do efeito |
| Reação 2º Turno | [−5, +5] em torno do 2º turno | Resolução de incerteza |
| Ciclo Interno (1º sem) | Jan–Jun do ano eleitoral | Expectativa |
| Ciclo Interno (2º sem) | Jul–Dez do ano eleitoral | Disputa e resultado |
| Estendida | Últimos 6 meses | Efeito agregado |

### 1.5 Inferência Estatística

Reportamos três testes de significância:

1. **Teste t simples** (longitudinal): t = AR̄ · √n / σ(AR)
2. **Teste t corrigido** (Silva et al., 2015): utiliza desvio padrão corrigido por
   autocorrelação: csd_t = √(t·var + 2·(t−1)·cov)
3. **Teste BMP** (Boehmer, Musumeci & Poulsen, 1991): normaliza CARs pelo sigma
   da estimação, controlando heteroscedasticidade event-induced.

### 1.6 Testes de Robustez

- **Janelas alternativas**: [−60,−1], [−10,+10], [−20,+20]
- **Cenário sem crises**: exclui dados de 2008 e 2020
- **Teste placebo**: 100 pseudo-eventos em anos não-eleitorais

## 2. Limitações

1. **Viés de sobrevivência**: A amostra inclui apenas empresas com dados disponíveis
   no Yahoo Finance (~45% dos tickers da base original). Small caps, empresas
   deslistadas e falidas são sub-representadas, especialmente antes de 2006.

2. **Ponderação equal-weighted**: Amplifica contribuição de small caps, podendo
   inflar volatilidade e betas setoriais.

3. **Clusterização de volatilidade**: Os p-values podem superestimar significância
   em períodos de alta volatilidade agrupada (e.g., crises de 2002 e 2008).

4. **Modelo de fator único**: O modelo de mercado (CAPM simplificado) não controla
   fatores adicionais (tamanho, valor, momentum). Extensões com Fama-French de
   3 fatores são recomendadas.

5. **Endogeneidade**: Não é possível estabelecer causalidade. Os resultados devem
   ser interpretados como **evidência consistente com risco político**, não como
   impacto causal dos ciclos eleitorais.

6. **Benchmarking Selic/IPCA**: Os ganhos/perdas reais reportados são ilustrativos
   e não constituem teste formal de significância.

## 3. Referências

- Boehmer, E., Musumeci, J. & Poulsen, A. (1991). Event-study methodology under
  conditions of event-induced variance. *Journal of Financial Economics*, 30(2).
- MacKinlay, A. C. (1997). Event Studies in Economics and Finance. *Journal of
  Economic Literature*, 35(1), 13-39.
- Nordhaus, W. D. (1975). The political business cycle. *Review of Economic Studies*, 42(2).
- Silva, W. A. M. et al. (2015). Evidências de retornos anormais nos processos de
  IPO na BMF&Bovespa no período de 2004 a 2013.
"""
    path = os.path.join(output_dir, "metodologia_limitacoes.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(texto)
    log.info("  ✓ metodologia_limitacoes.md")


def exportar_resultados(df_resultados: pd.DataFrame,
                        df_volatilidade: pd.DataFrame,
                        df_sobrevivencia: pd.DataFrame,
                        df_composicao: pd.DataFrame,
                        df_diagnostico: pd.DataFrame,
                        df_placebo: pd.DataFrame,
                        output_dir: str):
    """Exporta todos os resultados."""
    log.info("=" * 70)
    log.info("ETAPA 9: EXPORTAÇÃO")
    log.info("=" * 70)

    # CSV consolidado
    df_resultados.to_csv(
        os.path.join(output_dir, "resultados_consolidados_v2.csv"),
        index=False, encoding="utf-8-sig",
    )
    log.info("  ✓ resultados_consolidados_v2.csv")

    # Tabela de sobrevivência
    df_sobrevivencia.to_csv(
        os.path.join(output_dir, "tabela_sobrevivencia.csv"),
        index=False, encoding="utf-8-sig",
    )
    log.info("  ✓ tabela_sobrevivencia.csv")

    # Diagnóstico de download
    df_diagnostico.to_csv(
        os.path.join(output_dir, "diagnostico_download.csv"),
        index=False, encoding="utf-8-sig",
    )
    log.info("  ✓ diagnostico_download.csv")

    # Placebo
    if not df_placebo.empty:
        df_placebo.to_csv(
            os.path.join(output_dir, "placebo_test_results.csv"),
            index=False, encoding="utf-8-sig",
        )
        log.info("  ✓ placebo_test_results.csv")

    # Excel consolidado
    xlsx_path = os.path.join(output_dir, "analise_ciclos_eleitorais_v2.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Resultados completos
        df_resultados.to_excel(writer, sheet_name="Resultados", index=False)

        # Heatmaps tabulares (por janela principal)
        df_comp = df_resultados[df_resultados["cenario"] == "completo"]
        for janela in ["antecipacao_45", "reacao_curta", "reacao_2turno", "estendida"]:
            df_j = df_comp[df_comp["janela"] == janela]
            if df_j.empty:
                continue
            pivot_car = df_j.pivot_table(index="setor", columns="ano", values="car")
            pivot_pv = df_j.pivot_table(index="setor", columns="ano", values="p_value_simples")
            # Merge (converte para object para aceitar strings com asteriscos)
            merged = pivot_car.copy().astype(object)
            for col in merged.columns:
                for idx in merged.index:
                    c = pivot_car.loc[idx, col] if idx in pivot_car.index else np.nan
                    p = pivot_pv.loc[idx, col] if idx in pivot_pv.index else np.nan
                    if pd.notna(c) and pd.notna(p):
                        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                        merged.loc[idx, col] = f"{c:.4f}{star}"
                    elif pd.notna(c):
                        merged.loc[idx, col] = f"{c:.4f}"
            sname = f"CAR_{janela[:15]}"
            merged.to_excel(writer, sheet_name=sname)

        # Volatilidade
        if not df_volatilidade.empty:
            df_volatilidade.to_excel(writer, sheet_name="Volatilidade", index=False)

        # Sobrevivência
        df_sobrevivencia.to_excel(writer, sheet_name="Sobrevivência", index=False)

        # Composição setorial
        df_composicao.to_excel(writer, sheet_name="Composição Setorial", index=False)

        # Sumário executivo
        sumario = criar_sumario(df_resultados)
        sumario.to_excel(writer, sheet_name="Sumário Executivo", index=False)

    log.info("  ✓ %s", xlsx_path)

    # Texto de metodologia
    gerar_texto_metodologia(output_dir)


def criar_sumario(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """Sumário executivo."""
    df = df_resultados[df_resultados["cenario"] == "completo"]
    registros = []
    for janela in sorted(df["janela"].unique()):
        dj = df[df["janela"] == janela]
        if dj.empty:
            continue
        registros.append({
            "Janela": janela,
            "CAR Médio": f"{dj['car'].mean():.4f}",
            "CAR Mediano": f"{dj['car'].median():.4f}",
            "CAR Std": f"{dj['car'].std():.4f}",
            "% Sig 5% (simples)": f"{dj['sig_5pct_simples'].mean() * 100:.1f}%",
            "% Sig 5% (Silva)": f"{dj['sig_5pct_silva'].mean() * 100:.1f}%",
            "% Sig 5% (BMP)": f"{dj['sig_5pct_bmp'].mean() * 100:.1f}%",
            "Setor Maior |CAR|": dj.loc[dj["car"].abs().idxmax(), "setor"] if len(dj) > 0 else "",
            "Maior |CAR|": f"{dj['car'].abs().max():.4f}",
            "N obs": len(dj),
        })
    return pd.DataFrame(registros)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t0 = time.time()
    print()
    print("█" * 72)
    print("█  IMPACTO DOS CICLOS ELEITORAIS NA B3 (2002–2022) — v2 Acadêmica   █")
    print("█  Event Study · Modelo de Mercado · Silva et al. (2015) + BMP       █")
    print("█" * 72)
    print()

    if not os.path.exists(ARQUIVO_ENTRADA):
        log.error("Arquivo não encontrado: %s", ARQUIVO_ENTRADA)
        log.error("Coloque o xlsx na mesma pasta do script ou ajuste ARQUIVO_ENTRADA.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- ETAPA 1: Ingestão -----------------------------------------------
    df_empresas = carregar_lista_empresas(ARQUIVO_ENTRADA)

    # ---- ETAPA 2: Download de preços -------------------------------------
    tickers_yf = df_empresas["TICKER_YF"].unique().tolist()
    log.info("Total de tickers únicos para download: %d", len(tickers_yf))

    cache_precos = os.path.join(OUTPUT_DIR, "_cache_precos_v2.pkl")
    cache_ibov = os.path.join(OUTPUT_DIR, "_cache_ibov_v2.pkl")

    if os.path.exists(cache_precos) and os.path.exists(cache_ibov):
        log.info("Carregando preços do cache ...")
        df_precos = pd.read_pickle(cache_precos)
        ibov = pd.read_pickle(cache_ibov)
        # Diagnóstico do cache
        df_diagnostico = pd.DataFrame({
            "ticker_yf": tickers_yf,
            "status": ["ok" if t in df_precos.columns else "falha" for t in tickers_yf],
        })
    else:
        df_precos, df_diagnostico = baixar_precos_yfinance(tickers_yf, "2000-01-01", "2023-12-31")
        ibov = baixar_ibovespa("2000-01-01", "2023-12-31")
        df_precos.to_pickle(cache_precos)
        ibov.to_pickle(cache_ibov)

    # Enriquecer diagnóstico com setor
    diag_setor = df_empresas[["TICKER_YF", "SETOR_B3", "TICKER_ORIGINAL"]].drop_duplicates("TICKER_YF")
    df_diagnostico = df_diagnostico.merge(diag_setor, left_on="ticker_yf",
                                          right_on="TICKER_YF", how="left")

    # Log de cobertura por setor
    log.info("\nCobertura por setor:")
    for setor in SETORES_B3:
        d = df_diagnostico[df_diagnostico["SETOR_B3"] == setor]
        n_ok = (d["status"] == "ok").sum()
        n_total = len(d)
        log.info("  %s: %d/%d (%.0f%%)", setor, n_ok, n_total,
                 100 * n_ok / max(n_total, 1))

    # ---- ETAPA 3: Filtro de existência -----------------------------------
    df_precos = aplicar_filtro_existencia(df_precos, df_empresas)

    # ---- ETAPA 4: Índices setoriais --------------------------------------
    df_precos_setores, df_ret_setores, df_composicao = construir_indices_setoriais(
        df_precos, df_empresas
    )

    # Retornos Ibovespa
    ret_ibov = np.log(ibov / ibov.shift(1)).dropna()

    # Alinhar
    idx_comum = df_ret_setores.index.intersection(ret_ibov.index)
    df_ret_setores = df_ret_setores.loc[idx_comum]
    ret_ibov = ret_ibov.loc[idx_comum]

    # ---- Tabela de sobrevivência -----------------------------------------
    df_sobrevivencia = gerar_tabela_sobrevivencia(df_empresas, df_precos)

    # ---- ETAPA 5: Event Study (cenário completo) -------------------------
    df_res_completo = executar_analise(df_ret_setores, ret_ibov, "completo")

    # ---- ETAPA 5b: Event Study (cenário sem crises) ----------------------
    log.info("Preparando cenário sem crises (excluindo %s) ...", ANOS_CRISE)
    df_ret_sc = df_ret_setores.copy()
    mask_crise = df_ret_sc.index.year.isin(ANOS_CRISE)
    df_ret_sc.loc[mask_crise] = np.nan
    ret_ibov_sc = ret_ibov.copy()
    ret_ibov_sc.loc[mask_crise] = np.nan
    df_res_sem_crise = executar_analise(df_ret_sc, ret_ibov_sc, "sem_crises")

    # Consolida
    df_resultados = pd.concat([df_res_completo, df_res_sem_crise], ignore_index=True)

    # ---- ETAPA 6: Placebo ------------------------------------------------
    df_placebo = executar_teste_placebo(df_ret_setores, ret_ibov)

    # ---- ETAPA 7: Benchmarking -------------------------------------------
    df_resultados = adicionar_benchmarking(df_resultados)

    # ---- Volatilidade ----------------------------------------------------
    df_volatilidade = calcular_volatilidade_comparativa(df_ret_setores)

    # ---- ETAPA 8: Visualizações ------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("ETAPA 8: VISUALIZAÇÕES")
    log.info("=" * 70)
    gerar_visualizacoes(df_resultados, df_volatilidade, df_placebo, OUTPUT_DIR)

    # ---- ETAPA 9: Exportação ---------------------------------------------
    exportar_resultados(
        df_resultados, df_volatilidade, df_sobrevivencia,
        df_composicao, df_diagnostico, df_placebo, OUTPUT_DIR,
    )

    # ---- RESUMO FINAL ----------------------------------------------------
    elapsed = time.time() - t0
    df_comp = df_resultados[df_resultados["cenario"] == "completo"]

    print()
    print("=" * 72)
    print("RESUMO FINAL")
    print("=" * 72)
    print(f"\nTempo de execução: {elapsed:.1f}s")
    print(f"Observações (cenário completo): {len(df_comp)}")
    print(f"Setores: {df_comp['setor'].nunique()}")
    print(f"Anos eleitorais: {sorted(df_comp['ano'].unique())}")
    print(f"Observações placebo: {len(df_placebo)}")

    print("\nCAR Médio por Janela (cenário completo):")
    for janela in sorted(df_comp["janela"].unique()):
        dj = df_comp[df_comp["janela"] == janela]
        car_m = dj["car"].mean()
        sig_s = dj["sig_5pct_simples"].mean() * 100
        sig_silva = dj["sig_5pct_silva"].mean() * 100
        sig_bmp = dj["sig_5pct_bmp"].mean() * 100
        print(f"  {janela:25s}  CAR={car_m:+.4f}  "
              f"sig_simples={sig_s:.0f}%  sig_silva={sig_silva:.0f}%  sig_bmp={sig_bmp:.0f}%")

    n_ok = (df_diagnostico["status"] == "ok").sum()
    n_total = len(df_diagnostico)
    print(f"\nCobertura de dados: {n_ok}/{n_total} tickers ({100*n_ok/n_total:.0f}%)")
    print(f"\nArquivos em: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 72)
    print("✓ ANÁLISE v2 (ACADÊMICA) FINALIZADA COM SUCESSO!")
    print("=" * 72)


if __name__ == "__main__":
    main()
