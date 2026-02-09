"""
=============================================================================
RISCO POLÍTICO EM ANOS ELEITORAIS NO BRASIL (2002–2022)
Metodologia: Fama-French 3 Fatores + BHAR (Buy-and-Hold Abnormal Return)
=============================================================================

Autor: Reestruturação metodológica completa
Dados: NEFIN-USP (fatores), B3 (preços/volumes), Mapeamento setorial

Mudanças em relação à versão anterior:
  - Janelas dinâmicas baseadas no HGPE (não mais 45 dias fixos)
  - Modelo FF3 em vez de CAPM simples
  - BHAR (juros compostos) em vez de CAR (soma simples)
  - Ponderação por volume (Value-Weighted) em vez de média aritmética
  - Testes de robustez: Wilcoxon + Placebo + Diff-in-Diff
=============================================================================
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Optional, Dict, Tuple, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES E CONFIGURAÇÃO
# =============================================================================

OUTPUT_DIR = "./output_ff3_bhar"

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

# Anos de Placebo (não-eleitorais)
ANOS_PLACEBO = [2003, 2007, 2011, 2013, 2017, 2019]

# Datas fictícias de HGPE para placebo (mesma semana do calendário, ano ajustado)
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

# Parâmetros
JANELA_ESTIMACAO_DU = 252       # 1 ano de dias úteis
GAP_SEGURANCA_DU = 30           # gap entre estimação e evento
MIN_OBS_REGRESSAO = 60          # mínimo de obs para OLS
MIN_EMPRESAS_SETOR = 1          # mínimo de ativos válidos por setor
MIN_PREGOES_PCT = 0.40          # mínimo de pregões na janela de estimação


# =============================================================================
# ETAPA 1: CARREGAMENTO DE DADOS
# =============================================================================

def carregar_fatores_nefin(caminho: str) -> pd.DataFrame:
    """Carrega fatores NEFIN e padroniza colunas."""
    log.info("Carregando fatores NEFIN de %s", caminho)
    df = pd.read_csv(caminho, index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Padronizar nomes
    rename_map = {}
    if "Risk_Free" in df.columns:
        rename_map["Risk_Free"] = "Rf"
    if "Rm_minus_Rf" not in df.columns and "Mkt_Rf" in df.columns:
        rename_map["Mkt_Rf"] = "Rm_minus_Rf"
    elif "Rm_minus_Rf" in df.columns and "Mkt_Rf" not in df.columns:
        rename_map["Rm_minus_Rf"] = "Rm_minus_Rf"  # keep as is

    df = df.rename(columns=rename_map)

    # Garantir colunas necessárias
    cols_necessarias = ["Rm_minus_Rf", "SMB", "HML", "Rf"]
    for c in cols_necessarias:
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' não encontrada no NEFIN. Colunas: {df.columns.tolist()}")

    log.info("  NEFIN carregado: %d obs, %s a %s", len(df), df.index.min().date(), df.index.max().date())
    return df[cols_necessarias]


def carregar_empresas(caminho: str) -> pd.DataFrame:
    """Carrega lista de empresas e mapeamento setorial."""
    log.info("Carregando lista de empresas de %s", caminho)
    df = pd.read_excel(caminho, sheet_name="LISTA FINAL (Cont+IPOs-Canc)")
    df = df.dropna(subset=["TICKER", "SETOR_B3"])
    df["DT_REG"] = pd.to_datetime(df["DT_REG"], errors="coerce")
    log.info("  %d empresas carregadas, %d setores", len(df), df["SETOR_B3"].nunique())
    return df


def carregar_precos_volumes(caminho_precos: str, caminho_volumes: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega DataFrames de preços e volumes.
    Espera arquivos CSV ou pickle com índice = Data, colunas = Tickers.
    """
    log.info("Carregando preços e volumes...")

    if caminho_precos.endswith(".csv"):
        df_p = pd.read_csv(caminho_precos, index_col=0, parse_dates=True)
    elif caminho_precos.endswith(".pkl"):
        df_p = pd.read_pickle(caminho_precos)
    else:
        df_p = pd.read_excel(caminho_precos, index_col=0, parse_dates=True)

    if caminho_volumes.endswith(".csv"):
        df_v = pd.read_csv(caminho_volumes, index_col=0, parse_dates=True)
    elif caminho_volumes.endswith(".pkl"):
        df_v = pd.read_pickle(caminho_volumes)
    else:
        df_v = pd.read_excel(caminho_volumes, index_col=0, parse_dates=True)

    log.info("  Preços: %s, Volumes: %s", df_p.shape, df_v.shape)
    return df_p, df_v


# =============================================================================
# ETAPA 2: DEFINIÇÃO DINÂMICA DE JANELAS
# =============================================================================

def definir_janelas(ano: int, bdates: pd.DatetimeIndex,
                    datas_hgpe: dict = None,
                    datas_1turno: dict = None) -> Optional[Dict]:
    """
    Define janelas de estimação e evento para um dado ano.

    Janela de Evento: do HGPE até a véspera do 1º turno.
    Janela de Estimação: 252 DU, encerrando 30 DU antes do HGPE.

    Retorna dict com chaves: est_inicio, est_fim, evt_inicio, evt_fim
    ou None se não houver dados suficientes.
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

    # Véspera do 1º turno (último dia útil antes)
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

    janelas = {
        "est_inicio": est_inicio,
        "est_fim": est_fim,
        "evt_inicio": evt_inicio,
        "evt_fim": evt_fim,
        "ano": ano,
    }
    return janelas


# =============================================================================
# ETAPA 3: REGRESSÃO FF3 E CÁLCULO DE BHAR POR ATIVO
# =============================================================================

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
      3. Roda OLS(FF3) na janela de estimação
      4. Usa betas para prever retorno esperado na janela de evento
      5. Calcula BHAR = prod(1 + Ri) - prod(1 + E[Ri])

    Retorna dict com: ticker, ano, bhar, n_obs_est, n_obs_evt, r2, betas
    ou None se dados insuficientes.
    """
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    # 1. Definir janelas
    janelas = definir_janelas(ano, bdates, datas_hgpe, datas_1turno)
    if janelas is None:
        return None

    # 2. Verificar se ticker existe nos retornos
    if ticker not in df_retornos.columns:
        return None

    ret_ativo = df_retornos[ticker].dropna()

    # --- Janela de Estimação ---
    mask_est = (ret_ativo.index >= janelas["est_inicio"]) & (ret_ativo.index <= janelas["est_fim"])
    ret_est = ret_ativo.loc[mask_est]

    # Verificar mínimo de observações
    n_esperado_est = len(bdates[(bdates >= janelas["est_inicio"]) & (bdates <= janelas["est_fim"])])
    if len(ret_est) < max(MIN_OBS_REGRESSAO, int(n_esperado_est * MIN_PREGOES_PCT)):
        return None

    # Fatores na janela de estimação
    fac_est = df_nefin.loc[df_nefin.index.isin(ret_est.index)].copy()
    common_est = ret_est.index.intersection(fac_est.index)
    if len(common_est) < MIN_OBS_REGRESSAO:
        return None

    ret_est = ret_est.loc[common_est]
    fac_est = fac_est.loc[common_est]

    # 3. Regressão OLS: Ri - Rf = alpha + b1*(Rm-Rf) + b2*SMB + b3*HML + eps
    y = ret_est - fac_est["Rf"]
    X = fac_est[["Rm_minus_Rf", "SMB", "HML"]]
    X = sm.add_constant(X)

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

    if len(common_evt) < 5:  # mínimo para BHAR ter sentido
        return None

    ret_evt = ret_evt.loc[common_evt]
    fac_evt = fac_evt.loc[common_evt]

    # 4. Retorno esperado na janela de evento
    X_evt = fac_evt[["Rm_minus_Rf", "SMB", "HML"]]
    X_evt = sm.add_constant(X_evt)

    # E[Ri,t] = alpha + b1*(Rm-Rf)_t + b2*SMB_t + b3*HML_t + Rf_t
    ret_esperado = X_evt.dot(betas) + fac_evt["Rf"]

    # 5. BHAR = prod(1 + Ri,t) - prod(1 + E[Ri,t])
    bhar_realizado = (1 + ret_evt).prod()
    bhar_esperado = (1 + ret_esperado).prod()

    # Proteção contra valores extremos
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


# =============================================================================
# ETAPA 4: AGREGAÇÃO SETORIAL (VALUE-WEIGHTED)
# =============================================================================

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
    3. Agrega ABHAR = soma(w_i * BHAR_i)
    4. Roda testes estatísticos (T e Wilcoxon)

    Retorna dict com: setor, ano, abhar_vw, abhar_ew, n_ativos, p_ttest, p_wilcoxon
    """
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    resultados_ativos = []
    for ticker in tickers_setor:
        res = calcular_bhar_ativo(
            ticker, ano, df_retornos, df_nefin, bdates,
            datas_hgpe, datas_1turno
        )
        if res is not None:
            resultados_ativos.append(res)

    if len(resultados_ativos) < MIN_EMPRESAS_SETOR:
        return None

    df_res = pd.DataFrame(resultados_ativos)
    bhars = df_res["bhar"].values

    # --- Pesos Value-Weighted (volume médio na janela de estimação) ---
    janelas = definir_janelas(ano, bdates, datas_hgpe, datas_1turno)
    if janelas is None:
        return None

    pesos = []
    for _, row in df_res.iterrows():
        tk = row["ticker"]
        if tk in df_volumes.columns:
            vol_est = df_volumes.loc[
                (df_volumes.index >= janelas["est_inicio"]) &
                (df_volumes.index <= janelas["est_fim"]),
                tk
            ]
            vol_medio = vol_est.mean() if len(vol_est) > 0 else 0
        else:
            vol_medio = 0
        pesos.append(max(vol_medio, 0))

    pesos = np.array(pesos, dtype=float)

    # Fallback para EW se todos os volumes forem zero
    if pesos.sum() == 0:
        pesos = np.ones(len(pesos))

    pesos_norm = pesos / pesos.sum()

    # ABHAR (Value-Weighted)
    abhar_vw = np.dot(pesos_norm, bhars)

    # ABHAR (Equal-Weighted, para referência)
    abhar_ew = np.mean(bhars)

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
        "bhar_medio": np.mean(bhars),
        "bhar_mediana": np.median(bhars),
        "bhar_std": np.std(bhars, ddof=1) if len(bhars) > 1 else np.nan,
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
        "pesos_vw": pesos_norm.tolist(),
        "tickers_validos": df_res["ticker"].tolist(),
        "r2_medio": df_res["r2"].mean(),
    }


# =============================================================================
# ETAPA 5: MAPA DE RISCO (LOOP PRINCIPAL)
# =============================================================================

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
    Loop principal: varre todos os setores × anos e gera DataFrame completo.

    Retorna DataFrame com colunas:
      setor, ano, abhar_vw, abhar_ew, n_ativos, p_ttest, p_wilcoxon, ...
    """
    if anos is None:
        anos = sorted(DATAS_HGPE.keys())
    if datas_hgpe is None:
        datas_hgpe = DATAS_HGPE
    if datas_1turno is None:
        datas_1turno = DATAS_PRIMEIRO_TURNO

    log.info("=" * 70)
    log.info("GERANDO MAPA DE RISCO (%s) — %d anos", label, len(anos))
    log.info("=" * 70)

    # Calcular retornos logarítmicos
    bdates = df_retornos.index.sort_values()

    # Mapeamento setor -> tickers
    setores = df_empresas.groupby("SETOR_B3")["TICKER"].apply(list).to_dict()

    resultados = []
    for ano in anos:
        if ano not in datas_hgpe:
            log.warning("  Ano %d sem data de HGPE definida, pulando.", ano)
            continue

        log.info("  Ano %d ...", ano)
        for setor, tickers in setores.items():
            res = processar_setor(
                setor, ano, tickers, df_retornos, df_volumes, df_nefin, bdates,
                datas_hgpe, datas_1turno
            )
            if res is not None:
                res["tipo"] = label
                resultados.append(res)
                sig = "***" if res["p_ttest"] < 0.01 else ("**" if res["p_ttest"] < 0.05 else ("*" if res["p_ttest"] < 0.10 else ""))
                log.info(
                    "    %s: ABHAR_VW=%.4f, N=%d, p=%.4f %s",
                    setor, res["abhar_vw"], res["n_ativos"], res["p_ttest"], sig
                )

    df_final = pd.DataFrame(resultados)
    log.info("  Total de observações setor×ano: %d", len(df_final))
    return df_final


def gerar_heatmap_data(df_resultados: pd.DataFrame, metrica: str = "abhar_vw") -> pd.DataFrame:
    """Pivota os resultados para formato de Heatmap (Setores × Anos)."""
    if df_resultados.empty:
        return pd.DataFrame()
    return df_resultados.pivot_table(
        index="setor", columns="ano", values=metrica, aggfunc="first"
    )


# =============================================================================
# ETAPA 6: TESTE PLACEBO
# =============================================================================

def gerar_placebo(
    df_retornos: pd.DataFrame,
    df_volumes: pd.DataFrame,
    df_nefin: pd.DataFrame,
    df_empresas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Roda a mesma lógica de BHAR/FF3 para anos NÃO-eleitorais.
    Usa datas de pseudo-HGPE espelhadas nos mesmos dias do calendário.
    """
    log.info("=" * 70)
    log.info("TESTE PLACEBO — Anos não-eleitorais")
    log.info("=" * 70)

    return gerar_mapa_risco(
        df_retornos, df_volumes, df_nefin, df_empresas,
        anos=ANOS_PLACEBO,
        datas_hgpe=DATAS_HGPE_PLACEBO,
        datas_1turno=DATAS_PRIMEIRO_TURNO_PLACEBO,
        label="Placebo",
    )


# =============================================================================
# ETAPA 7: DIFFERENCE-IN-DIFFERENCES
# =============================================================================

def gerar_diff_in_diff(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa setores em 'Regulados' vs 'Não Regulados' e calcula DiD.

    Regulados: Petróleo/Gás, Utilidade Pública, Financeiro
    Não Regulados: todos os demais

    Retorna DataFrame com: ano, abhar_regulados, abhar_nao_regulados, diff, p_value
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
    for ano in df["ano"].unique():
        df_ano = df[df["ano"] == ano]

        reg = df_ano[df_ano["grupo"] == "Regulado"]["abhar_vw"]
        nreg = df_ano[df_ano["grupo"] == "Não Regulado"]["abhar_vw"]

        if len(reg) == 0 or len(nreg) == 0:
            continue

        mean_reg = reg.mean()
        mean_nreg = nreg.mean()
        diff = mean_reg - mean_nreg

        # Teste de diferença de médias
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
        # Média geral
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

    log.info("  DiD concluído")
    return df_did


# =============================================================================
# ETAPA 8: VISUALIZAÇÕES
# =============================================================================

def gerar_visualizacoes(df_eleitoral: pd.DataFrame, df_placebo: pd.DataFrame,
                        df_did: pd.DataFrame, output_dir: str):
    """Gera heatmaps e gráficos de barras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

    # --- Heatmap Eleitoral ---
    hm_data = gerar_heatmap_data(df_eleitoral)
    if not hm_data.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            hm_data * 100, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "ABHAR (%)"}
        )
        ax.set_title("Mapa de Risco Político Setorial — ABHAR (FF3, VW)\nAnos Eleitorais 2002–2022", fontsize=14)
        ax.set_ylabel("Setor B3")
        ax.set_xlabel("Ano Eleitoral")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_eleitoral.png"), dpi=300)
        plt.close()
        log.info("  Heatmap eleitoral salvo.")

    # --- Heatmap Placebo ---
    hm_placebo = gerar_heatmap_data(df_placebo)
    if not hm_placebo.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            hm_placebo * 100, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "ABHAR (%)"}
        )
        ax.set_title("Teste Placebo — ABHAR (FF3, VW)\nAnos Não-Eleitorais", fontsize=14)
        ax.set_ylabel("Setor B3")
        ax.set_xlabel("Ano Placebo")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_placebo.png"), dpi=300)
        plt.close()
        log.info("  Heatmap placebo salvo.")

    # --- DiD Bar Chart ---
    if not df_did.empty:
        df_plot = df_did[df_did["ano"] != "Média"].copy()
        df_plot["ano"] = df_plot["ano"].astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_plot))
        w = 0.35
        ax.bar(x - w/2, df_plot["abhar_regulados"] * 100, w, label="Regulados", color="#d62728")
        ax.bar(x + w/2, df_plot["abhar_nao_regulados"] * 100, w, label="Não Regulados", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["ano"])
        ax.set_ylabel("ABHAR Médio (%)")
        ax.set_title("Difference-in-Differences: Regulados vs Não Regulados")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "did_barras.png"), dpi=300)
        plt.close()
        log.info("  Gráfico DiD salvo.")


# =============================================================================
# ETAPA 9: EXPORTAÇÃO
# =============================================================================

def exportar_resultados(df_eleitoral: pd.DataFrame, df_placebo: pd.DataFrame,
                        df_did: pd.DataFrame, output_dir: str):
    """Exporta todos os resultados para Excel multi-abas."""
    os.makedirs(output_dir, exist_ok=True)
    caminho = os.path.join(output_dir, "resultados_ff3_bhar.xlsx")

    with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
        if not df_eleitoral.empty:
            df_eleitoral.to_excel(writer, sheet_name="Eleitoral_Detalhado", index=False)
            gerar_heatmap_data(df_eleitoral).to_excel(writer, sheet_name="Heatmap_Eleitoral")

            # Tabela de significância
            sig = df_eleitoral[["setor", "ano", "abhar_vw", "p_ttest", "p_wilcoxon", "n_ativos"]].copy()
            sig["sig_ttest"] = sig["p_ttest"].apply(
                lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            )
            sig.to_excel(writer, sheet_name="Significancia", index=False)

        if not df_placebo.empty:
            df_placebo.to_excel(writer, sheet_name="Placebo_Detalhado", index=False)
            gerar_heatmap_data(df_placebo).to_excel(writer, sheet_name="Heatmap_Placebo")

        if not df_did.empty:
            df_did.to_excel(writer, sheet_name="DiD", index=False)

    log.info("  Resultados exportados para %s", caminho)
    return caminho


def gerar_texto_metodologia(output_dir: str):
    """Gera arquivo com texto metodológico para o artigo."""
    texto = """# Metodologia — Risco Político em Anos Eleitorais no Brasil (2002–2022)
# Versão: FF3 + BHAR

## Modelo de Retorno Esperado
Fama-French 3 Fatores (NEFIN-USP):
  Ri,t - Rf,t = α + β1(Rm-Rf)t + β2·SMBt + β3·HMLt + εt

## Janelas
- Estimação: 252 dias úteis, encerrando 30 DU antes do HGPE (gap de segurança)
- Evento (Antecipação): do início do HGPE até a véspera do 1º Turno
- OLS com erros HAC (Newey-West, 5 lags)

## Retorno Anormal
BHAR = ∏(1 + Ri,t) - ∏(1 + E[Ri,t])
Onde E[Ri,t] = α̂ + β̂1(Rm-Rf)t + β̂2·SMBt + β̂3·HMLt + Rf,t

## Agregação Setorial
Value-Weighted pelo volume financeiro médio na janela de estimação:
  ABHAR_setor = Σ wi · BHARi,  onde wi = Vi / ΣVj

## Testes Estatísticos
- Teste t (H0: média dos BHARs = 0)
- Teste de Wilcoxon (H0: mediana dos BHARs = 0)

## Robustez
- Teste Placebo: anos não-eleitorais (2003, 2007, 2011, 2013, 2017, 2019)
- Difference-in-Differences: Regulados vs Não Regulados

## Setores Regulados
Petróleo/Gás/Biocombustíveis, Utilidade Pública, Financeiro
"""
    path = os.path.join(output_dir, "metodologia_ff3_bhar.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(texto)
    log.info("  Metodologia salva em %s", path)


# =============================================================================
# MAIN — EXECUÇÃO COMPLETA
# =============================================================================

def main(
    caminho_nefin: str = "nefin_factors.csv",
    caminho_empresas: str = "resultados_analise_b3_com_tickers.xlsx",
    caminho_precos: str = "precos.csv",
    caminho_volumes: str = "volumes.csv",
):
    """
    Execução completa do estudo.

    Parâmetros:
      caminho_nefin    : CSV dos fatores NEFIN
      caminho_empresas : Excel com lista de empresas e setores
      caminho_precos   : CSV/pkl com preços (index=Data, cols=Tickers)
      caminho_volumes  : CSV/pkl com volumes (index=Data, cols=Tickers)
    """
    import time
    t0 = time.time()

    print("\n" + "█" * 80)
    print("█  RISCO POLÍTICO ELEITORAL — FF3 + BHAR (2002–2022)  █")
    print("█" * 80 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Carregar dados ----
    df_nefin = carregar_fatores_nefin(caminho_nefin)
    df_empresas = carregar_empresas(caminho_empresas)
    df_precos, df_volumes = carregar_precos_volumes(caminho_precos, caminho_volumes)

    # Retornos logarítmicos
    log.info("Calculando retornos logarítmicos...")
    df_retornos = np.log(df_precos / df_precos.shift(1)).dropna(how="all")

    # ---- Análise Eleitoral ----
    df_eleitoral = gerar_mapa_risco(df_retornos, df_volumes, df_nefin, df_empresas)

    # ---- Teste Placebo ----
    df_placebo = gerar_placebo(df_retornos, df_volumes, df_nefin, df_empresas)

    # ---- Difference-in-Differences ----
    df_did = gerar_diff_in_diff(df_eleitoral)

    # ---- Exportação ----
    exportar_resultados(df_eleitoral, df_placebo, df_did, OUTPUT_DIR)
    gerar_visualizacoes(df_eleitoral, df_placebo, df_did, OUTPUT_DIR)
    gerar_texto_metodologia(OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  Concluído em {elapsed:.1f}s")
    print(f"  Resultados em: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")

    return df_eleitoral, df_placebo, df_did


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    # Ajuste os caminhos para seus arquivos:
    main(
        caminho_nefin="nefin_factors.csv",
        caminho_empresas="resultados_analise_b3_com_tickers.xlsx",
        caminho_precos="precos.csv",       # Gere via yfinance antes
        caminho_volumes="volumes.csv",     # Gere via yfinance antes
    )
