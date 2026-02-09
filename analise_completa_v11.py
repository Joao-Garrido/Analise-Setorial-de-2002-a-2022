"""
=============================================================================
 ANÁLISE COMPLETA FINAL
 Risco Político em Anos Eleitorais no Brasil (2002–2022)
=============================================================================
 Metodologia: Fama-French 3 Fatores + BHAR (Buy-and-Hold Abnormal Return)
 Dados: Download automático via yfinance + Fatores NEFIN-USP
=============================================================================

 Saídas:
   1. heatmap_eleitoral.png      — Mapa de Risco Setorial
   2. heatmap_placebo.png        — Teste Placebo
   3. timeline_bhar.png          — Evolução dia-a-dia do BHAR acumulado
   4. did_barras.png             — Difference-in-Differences
   5. resultados_ff3_bhar.xlsx   — Tabelas completas
   6. resultados_eleitoral.csv   — CSV consolidado
   7. metodologia_ff3_bhar.txt   — Texto metodológico

 Uso:
   python analise_completa_final.py

=============================================================================
"""

import os
import sys
import time
import warnings
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Optional, Dict, Tuple, List

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES E CONFIGURAÇÃO
# =============================================================================

ARQUIVO_EMPRESAS = "resultados_analise_b3_com_tickers.xlsx"
ARQUIVO_NEFIN = "nefin_factors.csv"
OUTPUT_DIR = "./output_ff3_bhar"

# Cache de dados (evita re-download)
CACHE_PRECOS = os.path.join(OUTPUT_DIR, "precos.csv")
CACHE_VOLUMES = os.path.join(OUTPUT_DIR, "volumes.csv")
CACHE_DIAGNOSTICO = os.path.join(OUTPUT_DIR, "diagnostico_download.csv")

# Mapeamento de tickers antigos → novos (De-Para)
TICKER_MAPPING = {
    "VVAR3": "BHIA3", "BTOW3": "AMER3", "LAME4": "LAME3", "PCAR4": "PCAR3",
    "KROT3": "COGN3", "ESTC3": "YDUQ3", "RAIL3": "RUMO3",
    "BVMF3": "B3SA3", "CTIP3": "B3SA3",
    "BRIN3": "BRML3", "BRML3": "ALOS3", "SMLE3": "SMFT3",
    "LINX3": "STNE3", "VIVT4": "VIVT3", "TIMP3": "TIMS3",
    "QGEP3": "BRAV3", "GNDI3": "HAPV3", "FIBR3": "SUZB3",
}

# Datas do HGPE (início do Horário Gratuito de Propaganda Eleitoral)
DATAS_HGPE = {
    2002: pd.Timestamp("2002-08-20"),
    2006: pd.Timestamp("2006-08-15"),
    2010: pd.Timestamp("2010-08-17"),
    2014: pd.Timestamp("2014-08-19"),
    2018: pd.Timestamp("2018-08-31"),
    2022: pd.Timestamp("2022-08-26"),
}

# Datas do 1º turno (véspera = fim da janela de evento)
DATAS_PRIMEIRO_TURNO = {
    2002: pd.Timestamp("2002-10-06"),
    2006: pd.Timestamp("2006-10-01"),
    2010: pd.Timestamp("2010-10-03"),
    2014: pd.Timestamp("2014-10-05"),
    2018: pd.Timestamp("2018-10-07"),
    2022: pd.Timestamp("2022-10-02"),
}

# Anos e datas de Placebo (não-eleitorais, mesma semana do calendário)
ANOS_PLACEBO = [2003, 2007, 2011, 2013, 2017, 2019]

DATAS_HGPE_PLACEBO = {
    2003: pd.Timestamp("2003-08-20"),
    2007: pd.Timestamp("2007-08-15"),
    2011: pd.Timestamp("2011-08-17"),
    2013: pd.Timestamp("2013-08-19"),
    2017: pd.Timestamp("2017-08-31"),
    2019: pd.Timestamp("2019-08-26"),
}

DATAS_PRIMEIRO_TURNO_PLACEBO = {
    2003: pd.Timestamp("2003-10-05"),
    2007: pd.Timestamp("2007-10-07"),
    2011: pd.Timestamp("2011-10-02"),
    2013: pd.Timestamp("2013-10-06"),
    2017: pd.Timestamp("2017-10-01"),
    2019: pd.Timestamp("2019-10-06"),
}

# Classificação regulatório para DiD
SETORES_REGULADOS = [
    "Petróleo, Gás e Biocombustíveis",
    "Utilidade Pública",
    "Financeiro",
]

# Parâmetros metodológicos
JANELA_ESTIMACAO_DU = 252       # 1 ano de dias úteis para treino
GAP_SEGURANCA_DU = 30           # gap entre fim da estimação e início do evento
MIN_OBS_REGRESSAO = 60          # mínimo de observações para OLS
MIN_EMPRESAS_SETOR = 1          # mínimo de ativos válidos por setor
MIN_PREGOES_PCT = 0.40          # mínimo de pregões presentes na janela de estimação
DOWNLOAD_START = "2000-01-01"
DOWNLOAD_END = "2023-12-31"


# #############################################################################
#
#  PARTE 1: INGESTÃO DE DADOS (Download Automático + Caching)
#
# #############################################################################

def carregar_empresas(caminho: str) -> pd.DataFrame:
    """
    Carrega lista de empresas do Excel.
    Aplica mapeamento De-Para de tickers e gera coluna TICKER_YF (.SA).
    """
    log.info("=" * 70)
    log.info("ETAPA 1: CARREGAMENTO DE EMPRESAS")
    log.info("=" * 70)

    df = pd.read_excel(caminho, sheet_name="LISTA FINAL (Cont+IPOs-Canc)")
    df = df.dropna(subset=["TICKER", "SETOR_B3"])
    df["DT_REG"] = pd.to_datetime(df["DT_REG"], errors="coerce")

    # Coluna ESTATAL
    if "ESTATAL" in df.columns:
        df["ESTATAL"] = df["ESTATAL"].astype(str).str.strip().str.upper().eq("SIM")
    else:
        df["ESTATAL"] = False

    # Mapeamento De-Para
    df["TICKER_ORIGINAL"] = df["TICKER"].str.strip()
    df["TICKER_MAPEADO"] = df["TICKER_ORIGINAL"].map(TICKER_MAPPING).fillna(df["TICKER_ORIGINAL"])
    df["TICKER_YF"] = df["TICKER_MAPEADO"] + ".SA"

    n_map = (df["TICKER_ORIGINAL"] != df["TICKER_MAPEADO"]).sum()
    n_est = df["ESTATAL"].sum()

    log.info("  %d empresas | %d setores | %d remapeados | %d estatais",
             len(df), df["SETOR_B3"].nunique(), n_map, n_est)
    return df


def carregar_fatores_nefin(caminho: str) -> pd.DataFrame:
    """Carrega e padroniza fatores NEFIN-USP."""
    log.info("Carregando fatores NEFIN de %s ...", caminho)
    df = pd.read_csv(caminho, index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"Risk_Free": "Rf"})

    cols = ["Rm_minus_Rf", "SMB", "HML", "Rf"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' não encontrada. Disponíveis: {df.columns.tolist()}")

    log.info("  NEFIN: %d obs, %s → %s", len(df), df.index.min().date(), df.index.max().date())
    return df[cols]


def garantir_dados_yfinance(df_empresas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Garante que preços e volumes estejam disponíveis.
    1. Se CSVs existem no cache → carrega direto (pula download).
    2. Se não → baixa via yfinance, trata MultiIndex, salva CSVs.

    Retorna: (df_precos, df_volumes) com index=Data, columns=Ticker_YF
    """
    log.info("=" * 70)
    log.info("ETAPA 2: INGESTÃO DE PREÇOS E VOLUMES")
    log.info("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Verificação de Cache ---
    if os.path.exists(CACHE_PRECOS) and os.path.exists(CACHE_VOLUMES):
        log.info("  Cache encontrado! Carregando de CSV...")
        df_precos = pd.read_csv(CACHE_PRECOS, index_col=0, parse_dates=True)
        df_volumes = pd.read_csv(CACHE_VOLUMES, index_col=0, parse_dates=True)
        log.info("  Preços: %s | Volumes: %s", df_precos.shape, df_volumes.shape)
        return df_precos, df_volumes

    # --- 2. Download via yfinance ---
    log.info("  Cache não encontrado. Iniciando download via yfinance...")

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance não instalado. Execute: pip install yfinance")
        sys.exit(1)

    tickers_yf = df_empresas["TICKER_YF"].unique().tolist()
    log.info("  %d tickers para baixar", len(tickers_yf))

    all_close = {}
    all_volume = {}
    diagnostico = []

    # Baixar em blocos de 50 tickers
    blocos = [tickers_yf[i:i + 50] for i in range(0, len(tickers_yf), 50)]

    for idx, bloco in enumerate(blocos):
        log.info("  Bloco %d/%d (%d tickers)...", idx + 1, len(blocos), len(bloco))
        try:
            data = yf.download(
                bloco,
                start=DOWNLOAD_START,
                end=DOWNLOAD_END,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if data.empty:
                log.warning("    Bloco %d retornou vazio.", idx + 1)
                for t in bloco:
                    diagnostico.append({"ticker_yf": t, "status": "falha", "motivo": "dados_vazios"})
                continue

            # Tratamento de MultiIndex (yfinance retorna MultiIndex quando >1 ticker)
            if isinstance(data.columns, pd.MultiIndex):
                # Nível 0: tipo (Close, Volume, etc.), Nível 1: ticker
                if "Close" in data.columns.get_level_values(0):
                    close_df = data["Close"]
                elif "Adj Close" in data.columns.get_level_values(0):
                    close_df = data["Adj Close"]
                else:
                    close_df = pd.DataFrame()

                volume_df = data["Volume"] if "Volume" in data.columns.get_level_values(0) else pd.DataFrame()
            else:
                # Apenas 1 ticker no bloco
                close_df = data[["Close"]].copy()
                close_df.columns = bloco
                volume_df = data[["Volume"]].copy()
                volume_df.columns = bloco

            # Coletar colunas com dados
            for col in close_df.columns:
                if close_df[col].notna().sum() > 0:
                    all_close[col] = close_df[col]
                    if col in volume_df.columns:
                        # Volume financeiro = Volume × Preço
                        vol_fin = volume_df[col] * close_df[col]
                        all_volume[col] = vol_fin
                    else:
                        all_volume[col] = pd.Series(dtype=float)
                    diagnostico.append({"ticker_yf": col, "status": "ok", "motivo": ""})
                else:
                    diagnostico.append({"ticker_yf": col, "status": "falha", "motivo": "sem_dados"})

            # Registrar tickers que não vieram no resultado
            for t in bloco:
                if t not in close_df.columns and not any(d["ticker_yf"] == t for d in diagnostico):
                    diagnostico.append({"ticker_yf": t, "status": "falha", "motivo": "nao_encontrado"})

        except Exception as e:
            log.warning("  Erro no bloco %d: %s", idx + 1, str(e)[:100])
            for t in bloco:
                if not any(d["ticker_yf"] == t for d in diagnostico):
                    diagnostico.append({"ticker_yf": t, "status": "falha", "motivo": f"erro: {str(e)[:60]}"})

        time.sleep(0.5)  # Rate limit

    # --- Montar DataFrames ---
    df_precos = pd.DataFrame(all_close)
    df_precos.index = pd.to_datetime(df_precos.index)
    df_precos = df_precos.sort_index()

    df_volumes = pd.DataFrame(all_volume)
    df_volumes.index = pd.to_datetime(df_volumes.index)
    df_volumes = df_volumes.sort_index()

    # --- Diagnóstico ---
    df_diag = pd.DataFrame(diagnostico)
    n_ok = (df_diag["status"] == "ok").sum() if len(df_diag) > 0 else 0
    n_falha = (df_diag["status"] == "falha").sum() if len(df_diag) > 0 else 0
    log.info("  Download concluído: %d OK | %d falhas (%.1f%%)",
             n_ok, n_falha, 100 * n_falha / max(len(tickers_yf), 1))

    # --- Salvar Cache ---
    df_precos.to_csv(CACHE_PRECOS)
    df_volumes.to_csv(CACHE_VOLUMES)
    df_diag.to_csv(CACHE_DIAGNOSTICO, index=False)
    log.info("  Cache salvo em %s", OUTPUT_DIR)

    return df_precos, df_volumes


def aplicar_filtro_existencia(df_precos: pd.DataFrame, df_empresas: pd.DataFrame) -> pd.DataFrame:
    """
    Invalida (NaN) preços anteriores à data de registro (DT_REG) de cada empresa.
    Garante que não usamos preços de antes da empresa existir na bolsa.
    """
    log.info("Aplicando filtro de existência (DT_REG)...")
    df = df_precos.copy()

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Mapeamento: TICKER_YF → DT_REG
    lookup = {}
    for _, row in df_empresas.iterrows():
        tk_yf = row.get("TICKER_YF", "")
        dt_reg = row.get("DT_REG", pd.NaT)
        if pd.notna(tk_yf) and pd.notna(dt_reg):
            lookup[tk_yf] = dt_reg

    n_filtrado = 0
    for col in df.columns:
        if col in lookup:
            mask = df.index < lookup[col]
            n_antes = df[col].notna().sum()
            df.loc[mask, col] = np.nan
            n_filtrado += n_antes - df[col].notna().sum()

    log.info("  %d observações invalidadas pelo filtro de existência", n_filtrado)
    return df


def preparar_retornos(df_precos: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos diários."""
    log.info("Calculando retornos logarítmicos...")
    df_ret = np.log(df_precos / df_precos.shift(1))
    df_ret = df_ret.replace([np.inf, -np.inf], np.nan)
    df_ret = df_ret.dropna(how="all")
    log.info("  Retornos: %s", df_ret.shape)
    return df_ret


# #############################################################################
#
#  PARTE 2: METODOLOGIA (FF3 + BHAR) — Inalterada
#
# #############################################################################

def definir_janelas(
    ano: int,
    bdates: pd.DatetimeIndex,
    datas_hgpe: dict = None,
    datas_1turno: dict = None,
) -> Optional[Dict]:
    """
    Define janelas de estimação e evento para um dado ano.

    Janela de Evento: do HGPE até a véspera do 1º turno.
    Janela de Estimação: 252 DU, encerrando 30 DU antes do HGPE (gap de segurança).

    Retorna dict com: est_inicio, est_fim, evt_inicio, evt_fim, ano
    ou None se dados insuficientes.
    """
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    if ano not in datas_hgpe or ano not in datas_1turno:
        return None

    hgpe = datas_hgpe[ano]
    turno1 = datas_1turno[ano]
    bdays = bdates.sort_values()

    # Evento: do HGPE até véspera do 1º turno
    evt_inicio_idx = bdays.searchsorted(hgpe, side="left")
    if evt_inicio_idx >= len(bdays):
        return None
    evt_inicio = bdays[evt_inicio_idx]

    evt_fim_idx = bdays.searchsorted(turno1, side="left") - 1
    if evt_fim_idx < 0:
        return None
    evt_fim = bdays[evt_fim_idx]

    # Estimação: termina 30 DU antes do HGPE
    est_fim_idx = evt_inicio_idx - GAP_SEGURANCA_DU
    if est_fim_idx < 0:
        return None
    est_fim = bdays[max(0, est_fim_idx)]

    est_inicio_idx = est_fim_idx - JANELA_ESTIMACAO_DU
    if est_inicio_idx < 0:
        return None
    est_inicio = bdays[max(0, est_inicio_idx)]

    return {
        "est_inicio": est_inicio,
        "est_fim": est_fim,
        "evt_inicio": evt_inicio,
        "evt_fim": evt_fim,
        "ano": ano,
    }


def calcular_bhar_ativo(
    ticker: str,
    ano: int,
    df_retornos: pd.DataFrame,
    df_nefin: pd.DataFrame,
    bdates: pd.DatetimeIndex,
    datas_hgpe: dict = None,
    datas_1turno: dict = None,
) -> Optional[Dict]:
    """
    Calcula o BHAR (Buy-and-Hold Abnormal Return) de um ativo para um ano.

    Passos:
      1. Define janelas dinâmicas (via HGPE)
      2. Extrai retornos do ativo nas janelas
      3. Roda OLS(FF3) na janela de estimação:
         Ri,t - Rf,t = α + β1·(Rm-Rf)t + β2·SMBt + β3·HMLt + εt
      4. Usa betas para prever retorno esperado na janela de evento:
         E[Ri,t] = α̂ + β̂1·(Rm-Rf)t + β̂2·SMBt + β̂3·HMLt + Rf,t
      5. Calcula BHAR = ∏(1 + Ri,t) - ∏(1 + E[Ri,t])

    Retorna dict ou None se dados insuficientes.
    """
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    # 1. Definir janelas
    janelas = definir_janelas(ano, bdates, datas_hgpe, datas_1turno)
    if janelas is None:
        return None

    # 2. Verificar dados do ativo
    if ticker not in df_retornos.columns:
        return None

    ret_ativo = df_retornos[ticker].dropna()

    # --- Janela de Estimação ---
    mask_est = (ret_ativo.index >= janelas["est_inicio"]) & (ret_ativo.index <= janelas["est_fim"])
    ret_est = ret_ativo.loc[mask_est]

    n_esperado_est = len(bdates[(bdates >= janelas["est_inicio"]) & (bdates <= janelas["est_fim"])])
    if len(ret_est) < max(MIN_OBS_REGRESSAO, int(n_esperado_est * MIN_PREGOES_PCT)):
        return None

    # Alinhamento com fatores NEFIN
    fac_est = df_nefin.loc[df_nefin.index.isin(ret_est.index)].copy()
    common_est = ret_est.index.intersection(fac_est.index)
    if len(common_est) < MIN_OBS_REGRESSAO:
        return None

    ret_est = ret_est.loc[common_est]
    fac_est = fac_est.loc[common_est]

    # 3. Regressão OLS (FF3)
    y = ret_est - fac_est["Rf"]
    X = sm.add_constant(fac_est[["Rm_minus_Rf", "SMB", "HML"]])

    try:
        modelo = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    except Exception:
        return None

    betas = modelo.params

    # --- Janela de Evento ---
    mask_evt = (ret_ativo.index >= janelas["evt_inicio"]) & (ret_ativo.index <= janelas["evt_fim"])
    ret_evt = ret_ativo.loc[mask_evt]

    fac_evt = df_nefin.loc[df_nefin.index.isin(ret_evt.index)].copy()
    common_evt = ret_evt.index.intersection(fac_evt.index)
    if len(common_evt) < 5:
        return None

    ret_evt = ret_evt.loc[common_evt]
    fac_evt = fac_evt.loc[common_evt]

    # 4. Retorno esperado na janela de evento
    X_evt = sm.add_constant(fac_evt[["Rm_minus_Rf", "SMB", "HML"]])
    ret_esperado = X_evt.dot(betas) + fac_evt["Rf"]

    # 5. BHAR = ∏(1+Ri) - ∏(1+E[Ri])
    bhar_realizado = (1 + ret_evt).prod()
    bhar_esperado = (1 + ret_esperado).prod()

    if np.isnan(bhar_realizado) or np.isnan(bhar_esperado) or bhar_esperado == 0:
        return None

    bhar = bhar_realizado - bhar_esperado

    return {
        "ticker": ticker,
        "ano": ano,
        "bhar": bhar,
        "bhar_realizado": bhar_realizado,
        "bhar_esperado": bhar_esperado,
        "n_obs_est": len(common_est),
        "n_obs_evt": len(common_evt),
        "r2": modelo.rsquared,
        "alpha": betas.get("const", np.nan),
        "beta_mkt": betas.get("Rm_minus_Rf", np.nan),
        "beta_smb": betas.get("SMB", np.nan),
        "beta_hml": betas.get("HML", np.nan),
        "evt_inicio": janelas["evt_inicio"],
        "evt_fim": janelas["evt_fim"],
    }


def calcular_bhar_diario_ativo(
    ticker: str,
    ano: int,
    df_retornos: pd.DataFrame,
    df_nefin: pd.DataFrame,
    bdates: pd.DatetimeIndex,
    datas_hgpe: dict = None,
    datas_1turno: dict = None,
) -> Optional[pd.Series]:
    """
    Calcula o BHAR ACUMULADO dia-a-dia na janela de evento.
    Retorna Series indexada por dia relativo (0, 1, 2, ...) com o BHAR cumulativo.
    Usado para o gráfico de linha do tempo (Timeline Plot).
    """
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    janelas = definir_janelas(ano, bdates, datas_hgpe, datas_1turno)
    if janelas is None:
        return None

    if ticker not in df_retornos.columns:
        return None

    ret_ativo = df_retornos[ticker].dropna()

    # Estimação
    mask_est = (ret_ativo.index >= janelas["est_inicio"]) & (ret_ativo.index <= janelas["est_fim"])
    ret_est = ret_ativo.loc[mask_est]

    n_esperado_est = len(bdates[(bdates >= janelas["est_inicio"]) & (bdates <= janelas["est_fim"])])
    if len(ret_est) < max(MIN_OBS_REGRESSAO, int(n_esperado_est * MIN_PREGOES_PCT)):
        return None

    fac_est = df_nefin.loc[df_nefin.index.isin(ret_est.index)]
    common_est = ret_est.index.intersection(fac_est.index)
    if len(common_est) < MIN_OBS_REGRESSAO:
        return None

    ret_est = ret_est.loc[common_est]
    fac_est = fac_est.loc[common_est]

    y = ret_est - fac_est["Rf"]
    X = sm.add_constant(fac_est[["Rm_minus_Rf", "SMB", "HML"]])
    try:
        modelo = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    except Exception:
        return None
    betas = modelo.params

    # Evento
    mask_evt = (ret_ativo.index >= janelas["evt_inicio"]) & (ret_ativo.index <= janelas["evt_fim"])
    ret_evt = ret_ativo.loc[mask_evt]
    fac_evt = df_nefin.loc[df_nefin.index.isin(ret_evt.index)]
    common_evt = ret_evt.index.intersection(fac_evt.index)
    if len(common_evt) < 5:
        return None

    ret_evt = ret_evt.loc[common_evt]
    fac_evt = fac_evt.loc[common_evt]

    X_evt = sm.add_constant(fac_evt[["Rm_minus_Rf", "SMB", "HML"]])
    ret_esperado = X_evt.dot(betas) + fac_evt["Rf"]

    # BHAR acumulado dia-a-dia
    cum_real = (1 + ret_evt).cumprod()
    cum_esperado = (1 + ret_esperado).cumprod()
    bhar_cum = cum_real - cum_esperado

    # Reindexar por dia relativo (0, 1, 2, ...)
    bhar_cum.index = range(len(bhar_cum))
    return bhar_cum


def processar_setor(
    setor: str,
    ano: int,
    tickers_setor: List[str],
    df_retornos: pd.DataFrame,
    df_volumes: pd.DataFrame,
    df_nefin: pd.DataFrame,
    bdates: pd.DatetimeIndex,
    datas_hgpe: dict = None,
    datas_1turno: dict = None,
) -> Optional[Dict]:
    """
    Processa todos os ativos de um setor para um ano.

    1. Calcula BHAR individual de cada ativo
    2. Calcula pesos VW (volume médio na janela de estimação)
    3. Agrega: ABHAR_VW = Σ(wi × BHARi)
    4. Roda testes: t-Student e Wilcoxon

    Retorna dict ou None se ativos insuficientes.
    """
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    # Calcular BHAR de cada ativo
    resultados_ativos = []
    for ticker in tickers_setor:
        res = calcular_bhar_ativo(
            ticker, ano, df_retornos, df_nefin, bdates, datas_hgpe, datas_1turno
        )
        if res is not None:
            resultados_ativos.append(res)

    if len(resultados_ativos) < MIN_EMPRESAS_SETOR:
        return None

    df_res = pd.DataFrame(resultados_ativos)

    # AJUSTE 2: Winsorização — clip BHARs entre -100% e +200%
    # Evita que falhas do modelo FF3 (previsões absurdas) destruam a média setorial
    df_res["bhar_raw"] = df_res["bhar"].copy()
    df_res["bhar"] = df_res["bhar"].clip(lower=-1.0, upper=2.0)
    n_clipped = (df_res["bhar"] != df_res["bhar_raw"]).sum()
    if n_clipped > 0:
        log.info("      Winsorização: %d BHARs extremos clipados em %s/%d", n_clipped, setor, ano)

    bhars = df_res["bhar"].values

    # --- Pesos Value-Weighted ---
    janelas = definir_janelas(ano, bdates, datas_hgpe, datas_1turno)
    if janelas is None:
        return None

    pesos = []
    for _, row in df_res.iterrows():
        tk = row["ticker"]
        if tk in df_volumes.columns:
            vol_est = df_volumes.loc[
                (df_volumes.index >= janelas["est_inicio"])
                & (df_volumes.index <= janelas["est_fim"]),
                tk,
            ]
            vol_medio = vol_est.mean() if len(vol_est) > 0 else 0.0
        else:
            vol_medio = 0.0
        pesos.append(max(vol_medio, 0.0))

    pesos = np.array(pesos, dtype=float)
    if pesos.sum() == 0:
        pesos = np.ones(len(pesos))  # Fallback para EW
    pesos_norm = pesos / pesos.sum()

    # ABHAR
    abhar_vw = float(np.dot(pesos_norm, bhars))
    abhar_ew = float(np.mean(bhars))

    # --- Testes Estatísticos ---
    p_ttest = np.nan
    p_wilcoxon = np.nan

    if len(bhars) >= 2:
        try:
            _, p_ttest = stats.ttest_1samp(bhars, 0)
        except Exception:
            pass

    if len(bhars) >= 5:
        try:
            _, p_wilcoxon = stats.wilcoxon(bhars, alternative="two-sided")
        except Exception:
            pass

    return {
        "setor": setor,
        "ano": ano,
        "abhar_vw": abhar_vw,
        "abhar_ew": abhar_ew,
        "n_ativos": len(bhars),
        "bhar_medio": float(np.mean(bhars)),
        "bhar_mediana": float(np.median(bhars)),
        "bhar_std": float(np.std(bhars, ddof=1)) if len(bhars) > 1 else np.nan,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "tickers_validos": df_res["ticker"].tolist(),
        "r2_medio": float(df_res["r2"].mean()),
    }


# #############################################################################
#
#  PARTE 3: LOOPS PRINCIPAIS (Mapa de Risco, Placebo, DiD)
#
# #############################################################################

def gerar_mapa_risco(
    df_retornos: pd.DataFrame,
    df_volumes: pd.DataFrame,
    df_nefin: pd.DataFrame,
    df_empresas: pd.DataFrame,
    anos: list = None,
    datas_hgpe: dict = None,
    datas_1turno: dict = None,
    label: str = "Eleitoral",
) -> pd.DataFrame:
    """
    Loop principal: varre Setores × Anos, calcula ABHAR(VW) de cada combinação.
    Retorna DataFrame completo de resultados.
    """
    if anos is None:
        anos = sorted(DATAS_HGPE.keys())
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    log.info("=" * 70)
    log.info("MAPA DE RISCO (%s) — %d anos × %d setores",
             label, len(anos), df_empresas["SETOR_B3"].nunique())
    log.info("=" * 70)

    bdates = df_retornos.index.sort_values()

    # Mapeamento setor → tickers (usar TICKER_YF para casar com df_retornos)
    setores = df_empresas.groupby("SETOR_B3")["TICKER_YF"].apply(list).to_dict()

    resultados = []
    for ano in anos:
        if ano not in datas_hgpe:
            log.warning("  Ano %d sem data de HGPE, pulando.", ano)
            continue

        log.info("  Ano %d ...", ano)
        for setor, tickers in setores.items():
            res = processar_setor(
                setor, ano, tickers, df_retornos, df_volumes, df_nefin, bdates,
                datas_hgpe, datas_1turno,
            )
            if res is not None:
                res["tipo"] = label
                resultados.append(res)
                sig = (
                    "***" if res["p_ttest"] < 0.01
                    else "**" if res["p_ttest"] < 0.05
                    else "*" if res["p_ttest"] < 0.10
                    else ""
                )
                log.info(
                    "    %-40s ABHAR_VW=%+.4f  N=%2d  p=%.4f %s",
                    setor, res["abhar_vw"], res["n_ativos"], res["p_ttest"], sig,
                )

    df_final = pd.DataFrame(resultados)
    log.info("  Total: %d observações setor×ano", len(df_final))
    return df_final


def gerar_placebo(
    df_retornos: pd.DataFrame,
    df_volumes: pd.DataFrame,
    df_nefin: pd.DataFrame,
    df_empresas: pd.DataFrame,
) -> pd.DataFrame:
    """Roda a mesma lógica FF3+BHAR para anos NÃO-eleitorais (Placebo Test)."""
    return gerar_mapa_risco(
        df_retornos, df_volumes, df_nefin, df_empresas,
        anos=ANOS_PLACEBO,
        datas_hgpe=DATAS_HGPE_PLACEBO,
        datas_1turno=DATAS_PRIMEIRO_TURNO_PLACEBO,
        label="Placebo",
    )


def gerar_diff_in_diff(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Difference-in-Differences: Regulados vs Não Regulados.

    Regulados: Petróleo/Gás, Utilidade Pública, Financeiro
    Não Regulados: demais setores
    """
    if df_resultados.empty:
        return pd.DataFrame()

    log.info("=" * 70)
    log.info("DIFFERENCE-IN-DIFFERENCES (Regulados vs Não Regulados)")
    log.info("=" * 70)

    df = df_resultados.copy()
    df["grupo"] = df["setor"].apply(
        lambda s: "Regulado" if s in SETORES_REGULADOS else "Não Regulado"
    )

    resultados_did = []
    for ano in sorted(df["ano"].unique()):
        df_ano = df[df["ano"] == ano]
        reg = df_ano[df_ano["grupo"] == "Regulado"]["abhar_vw"]
        nreg = df_ano[df_ano["grupo"] == "Não Regulado"]["abhar_vw"]

        if len(reg) == 0 or len(nreg) == 0:
            continue

        mean_reg = reg.mean()
        mean_nreg = nreg.mean()
        diff = mean_reg - mean_nreg

        p_val = np.nan
        if len(reg) >= 2 and len(nreg) >= 2:
            try:
                _, p_val = stats.ttest_ind(reg, nreg, equal_var=False)
            except Exception:
                pass

        resultados_did.append({
            "ano": ano,
            "abhar_regulados": mean_reg,
            "abhar_nao_regulados": mean_nreg,
            "diff": diff,
            "n_reg": len(reg),
            "n_nreg": len(nreg),
            "p_value": p_val,
        })

    df_did = pd.DataFrame(resultados_did).sort_values("ano")

    if not df_did.empty:
        media_geral = {
            "ano": "Média",
            "abhar_regulados": df_did["abhar_regulados"].mean(),
            "abhar_nao_regulados": df_did["abhar_nao_regulados"].mean(),
            "diff": df_did["diff"].mean(),
            "n_reg": "",
            "n_nreg": "",
            "p_value": np.nan,
        }
        df_did = pd.concat([df_did, pd.DataFrame([media_geral])], ignore_index=True)

    log.info("  DiD concluído — %d linhas", len(df_did))
    return df_did


# #############################################################################
#
#  PARTE 4: VISUALIZAÇÕES
#
# #############################################################################

def gerar_heatmap_data(df_resultados: pd.DataFrame, metrica: str = "abhar_vw") -> pd.DataFrame:
    """Pivota resultados para formato Heatmap (Setores × Anos)."""
    if df_resultados.empty:
        return pd.DataFrame()
    return df_resultados.pivot_table(
        index="setor", columns="ano", values=metrica, aggfunc="first"
    )


def gerar_grafico_linha_tempo(
    df_retornos: pd.DataFrame,
    df_volumes: pd.DataFrame,
    df_nefin: pd.DataFrame,
    df_empresas: pd.DataFrame,
    output_dir: str,
):
    """
    TIMELINE PLOT: Evolução dia-a-dia do BHAR acumulado na janela de evento.

    Agrupa setores em "Regulados" vs "Não Regulados".
    Plota uma linha média para cada grupo (média de todas as eleições).
    Mostra visualmente o momento em que o mercado "antecipa" o risco político.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("=" * 70)
    log.info("GERANDO GRÁFICO DE LINHA DO TEMPO (Timeline Plot)")
    log.info("=" * 70)

    bdates = df_retornos.index.sort_values()
    setores = df_empresas.groupby("SETOR_B3")["TICKER_YF"].apply(list).to_dict()
    anos_eleitorais = sorted(DATAS_HGPE.keys())

    # Coletar BHAR diário de cada ativo, classificado por grupo
    series_regulados = []
    series_nao_regulados = []

    for ano in anos_eleitorais:
        log.info("  Timeline — ano %d ...", ano)
        for setor, tickers in setores.items():
            grupo = "Regulado" if setor in SETORES_REGULADOS else "Não Regulado"

            for ticker in tickers:
                bhar_daily = calcular_bhar_diario_ativo(
                    ticker, ano, df_retornos, df_nefin, bdates,
                )
                if bhar_daily is not None and len(bhar_daily) >= 5:
                    if grupo == "Regulado":
                        series_regulados.append(bhar_daily)
                    else:
                        series_nao_regulados.append(bhar_daily)

    if not series_regulados and not series_nao_regulados:
        log.warning("  Sem dados suficientes para o gráfico de linha do tempo.")
        return

    # Construir DataFrames alinhados (cada coluna = 1 ativo×ano)
    def media_por_dia(lista_series: list) -> pd.Series:
        """Calcula a média do BHAR acumulado em cada dia relativo."""
        if not lista_series:
            return pd.Series(dtype=float)
        # Padronizar comprimento (usar o máximo e deixar NaN para os menores)
        max_len = max(len(s) for s in lista_series)
        df_temp = pd.DataFrame({i: s for i, s in enumerate(lista_series)})
        return df_temp.mean(axis=1)

    media_reg = media_por_dia(series_regulados)
    media_nreg = media_por_dia(series_nao_regulados)

    # --- Plotar ---
    fig, ax = plt.subplots(figsize=(14, 7))

    if len(media_reg) > 0:
        x_reg = range(len(media_reg))
        ax.plot(x_reg, media_reg * 100, color="#d62728", linewidth=2.5,
                label=f"Regulados — N = {len(series_regulados)} observações (Total Ativos × Eleições)", zorder=3)
        # Banda de confiança (±1 desvio padrão)
        if len(series_regulados) >= 3:
            df_temp_r = pd.DataFrame({i: s for i, s in enumerate(series_regulados)})
            std_r = df_temp_r.std(axis=1)
            ax.fill_between(x_reg, (media_reg - std_r) * 100, (media_reg + std_r) * 100,
                            alpha=0.15, color="#d62728")

    if len(media_nreg) > 0:
        x_nreg = range(len(media_nreg))
        ax.plot(x_nreg, media_nreg * 100, color="#1f77b4", linewidth=2.5,
                label=f"Não Regulados — N = {len(series_nao_regulados)} observações (Total Ativos × Eleições)", zorder=3)
        if len(series_nao_regulados) >= 3:
            df_temp_n = pd.DataFrame({i: s for i, s in enumerate(series_nao_regulados)})
            std_n = df_temp_n.std(axis=1)
            ax.fill_between(x_nreg, (media_nreg - std_n) * 100, (media_nreg + std_n) * 100,
                            alpha=0.15, color="#1f77b4")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Dias Úteis desde o Início do HGPE", fontsize=12)
    ax.set_ylabel("BHAR Acumulado (%)", fontsize=12)
    ax.set_title(
        "Trajetória do Retorno Anormal Acumulado (BHAR)\n"
        "Janela de Antecipação: HGPE → Véspera do 1º Turno (Média 2002–2022)",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    # Marcadores de referência
    max_x = max(len(media_reg), len(media_nreg))
    if max_x > 10:
        ax.axvline(x=max_x - 1, color="gray", linestyle="--", alpha=0.6)
        ax.text(max_x - 2, ax.get_ylim()[1] * 0.9, "Véspera\n1º Turno",
                ha="right", fontsize=9, color="gray")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6)
    ax.text(1, ax.get_ylim()[1] * 0.9, "Início\nHGPE",
            ha="left", fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(output_dir, "timeline_bhar.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Timeline salvo: %s", path)


def gerar_visualizacoes(
    df_eleitoral: pd.DataFrame,
    df_placebo: pd.DataFrame,
    df_did: pd.DataFrame,
    output_dir: str,
):
    """Gera todos os gráficos estáticos (Heatmaps + DiD)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

    # --- 1. Heatmap Eleitoral ---
    hm_data = gerar_heatmap_data(df_eleitoral)
    if not hm_data.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            hm_data * 100, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, vmin=-40, vmax=40, linewidths=0.5, ax=ax,
            cbar_kws={"label": "ABHAR (%)"},
        )
        ax.set_title(
            "Mapa de Risco Político Setorial — ABHAR (FF3, Value-Weighted)\n"
            "Anos Eleitorais 2002–2022",
            fontsize=14,
        )
        ax.set_ylabel("Setor B3")
        ax.set_xlabel("Ano Eleitoral")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_eleitoral.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ heatmap_eleitoral.png")

    # --- 2. Heatmap Placebo ---
    hm_placebo = gerar_heatmap_data(df_placebo)
    if not hm_placebo.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            hm_placebo * 100, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, vmin=-40, vmax=40, linewidths=0.5, ax=ax,
            cbar_kws={"label": "ABHAR (%)"},
        )
        ax.set_title(
            "Teste Placebo — ABHAR (FF3, Value-Weighted)\nAnos Não-Eleitorais",
            fontsize=14,
        )
        ax.set_ylabel("Setor B3")
        ax.set_xlabel("Ano Placebo")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_placebo.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ heatmap_placebo.png")

    # --- 3. DiD Bar Chart ---
    if not df_did.empty:
        df_plot = df_did[df_did["ano"] != "Média"].copy()
        if not df_plot.empty:
            df_plot["ano"] = df_plot["ano"].astype(int)
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(df_plot))
            w = 0.35
            bars1 = ax.bar(x - w / 2, df_plot["abhar_regulados"] * 100, w,
                           label="Regulados", color="#d62728", edgecolor="white")
            bars2 = ax.bar(x + w / 2, df_plot["abhar_nao_regulados"] * 100, w,
                           label="Não Regulados", color="#1f77b4", edgecolor="white")

            # Marcar significância
            for i, (_, row) in enumerate(df_plot.iterrows()):
                if row["p_value"] < 0.05:
                    ax.text(i, max(row["abhar_regulados"], row["abhar_nao_regulados"]) * 100 + 0.3,
                            "**", ha="center", fontsize=12, color="black")

            ax.set_xticks(x)
            ax.set_xticklabels(df_plot["ano"])
            ax.set_ylabel("ABHAR Médio (%)")
            ax.set_title("Difference-in-Differences: Regulados vs Não Regulados", fontsize=14)
            ax.legend()
            ax.axhline(0, color="black", linewidth=0.8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "did_barras.png"), dpi=300, bbox_inches="tight")
            plt.close()
            log.info("  ✓ did_barras.png")


# #############################################################################
#
#  PARTE 5: EXPORTAÇÃO
#
# #############################################################################

def exportar_resultados(
    df_eleitoral: pd.DataFrame,
    df_placebo: pd.DataFrame,
    df_did: pd.DataFrame,
    output_dir: str,
) -> str:
    """Exporta resultados completos em Excel multi-abas e CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Excel ---
    caminho_xlsx = os.path.join(output_dir, "resultados_ff3_bhar.xlsx")
    with pd.ExcelWriter(caminho_xlsx, engine="openpyxl") as writer:
        if not df_eleitoral.empty:
            # Detalhado (excluir listas para Excel)
            df_exp = df_eleitoral.copy()
            df_exp["tickers_validos"] = df_exp["tickers_validos"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )
            df_exp.to_excel(writer, sheet_name="Eleitoral_Detalhado", index=False)
            gerar_heatmap_data(df_eleitoral).to_excel(writer, sheet_name="Heatmap_Eleitoral")

            # Tabela de Significância
            sig = df_eleitoral[["setor", "ano", "abhar_vw", "p_ttest", "p_wilcoxon", "n_ativos"]].copy()
            sig["sig_ttest"] = sig["p_ttest"].apply(
                lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            )
            sig["sig_wilcoxon"] = sig["p_wilcoxon"].apply(
                lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                if pd.notna(p) else ""
            )
            sig.to_excel(writer, sheet_name="Significancia", index=False)

        if not df_placebo.empty:
            df_plac = df_placebo.copy()
            df_plac["tickers_validos"] = df_plac["tickers_validos"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )
            df_plac.to_excel(writer, sheet_name="Placebo_Detalhado", index=False)
            gerar_heatmap_data(df_placebo).to_excel(writer, sheet_name="Heatmap_Placebo")

        if not df_did.empty:
            df_did.to_excel(writer, sheet_name="DiD", index=False)

    log.info("  ✓ %s", caminho_xlsx)

    # --- CSV consolidado ---
    caminho_csv = os.path.join(output_dir, "resultados_eleitoral.csv")
    if not df_eleitoral.empty:
        df_csv = df_eleitoral.copy()
        df_csv["tickers_validos"] = df_csv["tickers_validos"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x)
        )
        df_csv.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        log.info("  ✓ %s", caminho_csv)

    return caminho_xlsx


def gerar_texto_metodologia(output_dir: str):
    """Gera arquivo com texto metodológico para o artigo."""
    texto = """# Metodologia — Risco Político em Anos Eleitorais no Brasil (2002–2022)
# Versão Final: Fama-French 3 Fatores + BHAR

## 1. Modelo de Retorno Esperado
Fama-French 3 Fatores (dados NEFIN-USP):
  Ri,t − Rf,t = α + β₁·(Rm−Rf)t + β₂·SMBt + β₃·HMLt + εt
  Estimação via OLS com erros robustos HAC (Newey-West, 5 lags).

## 2. Janelas (Dinâmicas via HGPE)
- Janela de Estimação: 252 dias úteis (≈1 ano), encerrando 30 DU antes do HGPE.
- Janela de Evento (Antecipação): do início do HGPE até a véspera do 1º Turno.
- Gap de segurança: 30 DU entre estimação e evento (evita contaminação).

## 3. Retorno Anormal — BHAR (Buy-and-Hold Abnormal Return)
  BHAR = ∏(1 + Ri,t) − ∏(1 + E[Ri,t])
  Onde E[Ri,t] = α̂ + β̂₁·(Rm−Rf)t + β̂₂·SMBt + β̂₃·HMLt + Rf,t
  Captura efeito de juros compostos (superior ao CAR para janelas longas).

## 4. Agregação Setorial — Value-Weighted
  ABHAR_setor = Σ(wi × BHARi)
  Onde wi = Vi / ΣVj  (volume financeiro médio na janela de estimação)

## 5. Testes Estatísticos
- Teste t de Student (H₀: média dos BHARs = 0)
- Teste de Wilcoxon (H₀: mediana dos BHARs = 0) — robustez contra outliers

## 6. Testes de Robustez
- Teste Placebo: anos não-eleitorais (2003, 2007, 2011, 2013, 2017, 2019)
  com datas espelhadas nas mesmas semanas do calendário.
- Difference-in-Differences:
  Regulados (Petróleo/Gás, Utilidade Pública, Financeiro) vs Não Regulados

## 7. Visualizações
- Heatmap Setorial: ABHAR(%) por setor × ano eleitoral
- Timeline Plot: BHAR acumulado dia-a-dia na janela de antecipação
  (Regulados vs Não Regulados, média de todas as eleições)
- DiD Bar Chart: comparação direta entre grupos por ano

## 8. Dados
- Preços e Volumes: Yahoo Finance (ajustados por proventos)
- Fatores de Risco: NEFIN-USP (Rm−Rf, SMB, HML, Rf)
- Classificação Setorial: B3 (Bolsa de Valores)
"""
    path = os.path.join(output_dir, "metodologia_ff3_bhar.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(texto)
    log.info("  ✓ %s", path)


# #############################################################################
#
#  MAIN — EXECUÇÃO COMPLETA
#
# #############################################################################

def main():
    t0 = time.time()

    print("\n" + "█" * 80)
    print("█                                                                              █")
    print("█   RISCO POLÍTICO EM ANOS ELEITORAIS NO BRASIL (2002–2022)                    █")
    print("█   Fama-French 3 Fatores + BHAR — Versão Final Unificada                      █")
    print("█                                                                              █")
    print("█" * 80 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================================
    # ETAPA 1: Carregar empresas e fatores NEFIN
    # =====================================================================
    df_empresas = carregar_empresas(ARQUIVO_EMPRESAS)

    # AJUSTE 1: Remover setor "Outros" (dados sujos, outliers extremos)
    n_antes = len(df_empresas)
    df_empresas = df_empresas[df_empresas["SETOR_B3"] != "Outros"].reset_index(drop=True)
    log.info("  Filtro 'Outros': %d → %d empresas (%d removidas)",
             n_antes, len(df_empresas), n_antes - len(df_empresas))

    df_nefin = carregar_fatores_nefin(ARQUIVO_NEFIN)

    # =====================================================================
    # ETAPA 2: Garantir dados de preços e volumes (download ou cache)
    # =====================================================================
    df_precos, df_volumes = garantir_dados_yfinance(df_empresas)

    # Filtro de existência (invalidar preços antes do DT_REG)
    df_precos = aplicar_filtro_existencia(df_precos, df_empresas)

    # Retornos logarítmicos
    df_retornos = preparar_retornos(df_precos)

    # Impressão das janelas para conferência
    log.info("Janelas Dinâmicas (HGPE):")
    bdates = df_retornos.index.sort_values()
    for ano in sorted(DATAS_HGPE.keys()):
        j = definir_janelas(ano, bdates)
        if j:
            log.info(
                "  %d: Estimação [%s → %s]  Evento [%s → %s]",
                ano, j["est_inicio"].date(), j["est_fim"].date(),
                j["evt_inicio"].date(), j["evt_fim"].date(),
            )

    # =====================================================================
    # ETAPA 3: Mapa de Risco (Anos Eleitorais)
    # =====================================================================
    df_eleitoral = gerar_mapa_risco(df_retornos, df_volumes, df_nefin, df_empresas)

    # =====================================================================
    # ETAPA 4: Teste Placebo (Anos Não-Eleitorais)
    # =====================================================================
    df_placebo = gerar_placebo(df_retornos, df_volumes, df_nefin, df_empresas)

    # =====================================================================
    # ETAPA 5: Difference-in-Differences
    # =====================================================================
    df_did = gerar_diff_in_diff(df_eleitoral)

    # =====================================================================
    # ETAPA 6: Exportação
    # =====================================================================
    log.info("=" * 70)
    log.info("EXPORTAÇÃO DE RESULTADOS")
    log.info("=" * 70)
    exportar_resultados(df_eleitoral, df_placebo, df_did, OUTPUT_DIR)

    # =====================================================================
    # ETAPA 7: Visualizações
    # =====================================================================
    log.info("=" * 70)
    log.info("GERANDO VISUALIZAÇÕES")
    log.info("=" * 70)
    gerar_visualizacoes(df_eleitoral, df_placebo, df_did, OUTPUT_DIR)

    # Timeline Plot (Gráfico de Linha do Tempo)
    gerar_grafico_linha_tempo(df_retornos, df_volumes, df_nefin, df_empresas, OUTPUT_DIR)

    # Metodologia
    gerar_texto_metodologia(OUTPUT_DIR)

    # =====================================================================
    # RESUMO FINAL
    # =====================================================================
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("  CONCLUÍDO em {:.1f}s".format(elapsed))
    print("=" * 80)
    print(f"\n  Outputs em: {os.path.abspath(OUTPUT_DIR)}/")
    print()

    # Listar arquivos gerados
    arquivos = sorted(os.listdir(OUTPUT_DIR))
    for a in arquivos:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, a))
        print(f"    {a:<45s} ({size:>10,} bytes)")

    print()

    # Resumo por setor
    if not df_eleitoral.empty:
        print("  RESUMO POR SETOR (Média dos anos eleitorais):")
        print("  " + "-" * 76)
        resumo = df_eleitoral.groupby("setor").agg(
            ABHAR_VW=("abhar_vw", "mean"),
            N_total=("n_ativos", "sum"),
            p_medio=("p_ttest", "mean"),
            anos_sig=("p_ttest", lambda x: (x < 0.05).sum()),
        ).sort_values("ABHAR_VW")

        for setor, row in resumo.iterrows():
            sig = "✓" if row["anos_sig"] >= 2 else ""
            print(f"    {setor:<42s} ABHAR={row['ABHAR_VW']:+.4f}  "
                  f"N={int(row['N_total']):3d}  sig_5%={int(row['anos_sig'])}x {sig}")

    print("\n" + "█" * 80 + "\n")

    return df_eleitoral, df_placebo, df_did


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
