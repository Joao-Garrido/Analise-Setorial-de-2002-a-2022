"""
================================================================================
ANÁLISE DE IMPACTO DOS CICLOS ELEITORAIS NO MERCADO DE CAPITAIS BRASILEIRO
B3 – 2002 a 2022
================================================================================

Metodologia: Event Study com Modelo de Mercado (Silva et al., 2015)
Referências teóricas: Nordhaus (1975) – Political Business Cycles

REQUISITOS (pip install):
    pip install pandas numpy openpyxl yfinance statsmodels scipy matplotlib seaborn tqdm

ENTRADA:
    resultados_analise_b3_com_tickers.xlsx
        → aba "LISTA FINAL (Cont+IPOs-Canc)"
        → colunas: TICKER, DENOM_SOCIAL, SETOR_B3, DT_REG, DT_CANCEL

SAÍDA (pasta ./output/):
    analise_ciclos_eleitorais_completa.xlsx   – resultados consolidados
    heatmap_car_antecipacao.png               – mapa de calor (antecipação)
    heatmap_car_reacao.png                    – mapa de calor (reação imediata)
    heatmap_car_ciclo_interno.png             – mapa de calor (ciclo interno)
    heatmap_car_estendida.png                 – mapa de calor (estendida)
    evolucao_temporal_cars.png                – evolução temporal
    volatilidade_eleitoral_vs_nao.png         – comparativo de volatilidade
================================================================================
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS
# ---------------------------------------------------------------------------

# Caminho do arquivo de entrada (ajuste se necessário)
ARQUIVO_ENTRADA = "resultados_analise_b3_com_tickers.xlsx"
SHEET_NAME = "LISTA FINAL (Cont+IPOs-Canc)"

# Diretório de saída
OUTPUT_DIR = "./output"

# Anos eleitorais e datas de 1º turno
DATAS_PRIMEIRO_TURNO = {
    2002: pd.Timestamp("2002-10-06"),
    2006: pd.Timestamp("2006-10-01"),
    2010: pd.Timestamp("2010-10-03"),
    2014: pd.Timestamp("2014-10-05"),
    2018: pd.Timestamp("2018-10-07"),
    2022: pd.Timestamp("2022-10-02"),
}

# Datas de 2º turno
DATAS_SEGUNDO_TURNO = {
    2002: pd.Timestamp("2002-10-27"),
    2006: pd.Timestamp("2006-10-29"),
    2010: pd.Timestamp("2010-10-31"),
    2014: pd.Timestamp("2014-10-26"),
    2018: pd.Timestamp("2018-10-28"),
    2022: pd.Timestamp("2022-10-30"),
}

ANOS_ELEITORAIS = sorted(DATAS_PRIMEIRO_TURNO.keys())

# Setores oficiais da B3 (conforme dados)
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

# Anos de crise (para cenário controlado)
ANOS_CRISE = [2008, 2020]


# ============================================================================
# ETAPA 1 – INGESTÃO DE DADOS
# ============================================================================

def carregar_lista_empresas(caminho: str) -> pd.DataFrame:
    """Carrega e limpa a lista de empresas da B3."""
    log.info("Carregando lista de empresas de %s ...", caminho)
    df = pd.read_excel(caminho, sheet_name=SHEET_NAME)
    df = df.dropna(subset=["TICKER", "SETOR_B3"])
    df["DT_REG"] = pd.to_datetime(df["DT_REG"], errors="coerce")
    df["DT_CANCEL"] = pd.to_datetime(df["DT_CANCEL"], errors="coerce")
    # Ticker com sufixo .SA
    df["TICKER_YF"] = df["TICKER"].str.strip() + ".SA"
    log.info("  → %d empresas em %d setores", len(df), df["SETOR_B3"].nunique())
    return df


def baixar_precos_yfinance(tickers: list, start: str = "2000-01-01",
                           end: str = "2023-12-31") -> pd.DataFrame:
    """
    Baixa preços ajustados (Adj Close) via yfinance.
    Retorna DataFrame com colunas = tickers, índice = datas.
    """
    import yfinance as yf

    log.info("Baixando preços de %d tickers via yfinance ...", len(tickers))
    log.info("  Período: %s a %s", start, end)

    # yfinance aceita download em lote
    # Faz em blocos de 50 para evitar timeout
    all_data = {}
    blocos = [tickers[i:i + 50] for i in range(0, len(tickers), 50)]

    for idx, bloco in enumerate(blocos):
        log.info("  Bloco %d/%d (%d tickers) ...", idx + 1, len(blocos), len(bloco))
        try:
            data = yf.download(
                bloco,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if data.empty:
                continue
            # Se múltiplos tickers, pegar coluna 'Close' (auto_adjust=True → Close = Adj Close)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                # Ticker único
                close = data[["Close"]]
                close.columns = bloco
            for col in close.columns:
                if close[col].notna().sum() > 0:
                    all_data[col] = close[col]
        except Exception as e:
            log.warning("  Erro no bloco %d: %s", idx + 1, e)
        time.sleep(0.5)

    df_precos = pd.DataFrame(all_data)
    df_precos.index = pd.to_datetime(df_precos.index)
    log.info("  → Obtidos preços para %d tickers", df_precos.shape[1])
    return df_precos


def baixar_ibovespa(start: str = "2000-01-01", end: str = "2023-12-31") -> pd.Series:
    """Baixa o Ibovespa (^BVSP) via yfinance."""
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
    Invalida (NaN) preços fora do período de vida da empresa
    (antes de DT_REG ou após DT_CANCEL).
    """
    log.info("Aplicando filtro de existência ...")
    df_filtrado = df_precos.copy()
    ticker_map = df_empresas.set_index("TICKER_YF")[["DT_REG", "DT_CANCEL"]]

    for col in df_filtrado.columns:
        if col in ticker_map.index:
            dt_reg = ticker_map.loc[col, "DT_REG"]
            dt_cancel = ticker_map.loc[col, "DT_CANCEL"]
            if pd.notna(dt_reg):
                df_filtrado.loc[df_filtrado.index < dt_reg, col] = np.nan
            if pd.notna(dt_cancel):
                df_filtrado.loc[df_filtrado.index > dt_cancel, col] = np.nan

    return df_filtrado


# ============================================================================
# ETAPA 2 – CONSTRUÇÃO DE ÍNDICES SETORIAIS (EQUAL-WEIGHTED)
# ============================================================================

def construir_indices_setoriais(df_precos: pd.DataFrame,
                                df_empresas: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói 11 índices setoriais equal-weighted.
    Para cada dia t, calcula o retorno médio simples de todas as empresas
    ativas do setor naquele dia.
    Retorna DataFrame de PREÇOS (nível) dos índices setoriais.
    """
    log.info("Construindo índices setoriais equal-weighted ...")

    # Retornos logarítmicos diários
    ret = np.log(df_precos / df_precos.shift(1))

    # Mapeamento ticker → setor
    t2s = df_empresas.set_index("TICKER_YF")["SETOR_B3"].to_dict()

    indices_ret = {}
    for setor in SETORES_B3:
        cols_setor = [c for c in ret.columns if t2s.get(c) == setor]
        if len(cols_setor) == 0:
            log.warning("  Nenhum ticker para o setor '%s'", setor)
            continue
        # Retorno médio igual-ponderado (ignora NaN)
        ret_setor = ret[cols_setor].mean(axis=1)
        indices_ret[setor] = ret_setor
        n_tickers = ret[cols_setor].notna().sum(axis=1)
        log.info("  %s: %d tickers disponíveis (média de %.0f/dia)",
                 setor, len(cols_setor), n_tickers.mean())

    df_ret_setores = pd.DataFrame(indices_ret)

    # Converter retornos em nível de preço (base 100)
    df_precos_setores = pd.DataFrame(index=df_ret_setores.index)
    for col in df_ret_setores.columns:
        cumret = df_ret_setores[col].fillna(0).cumsum()
        df_precos_setores[col] = 100 * np.exp(cumret)

    return df_precos_setores, df_ret_setores


# ============================================================================
# ETAPA 3 – DEFINIÇÃO DE JANELAS
# ============================================================================

def obter_dias_uteis(datas_index: pd.DatetimeIndex, data_ref: pd.Timestamp,
                     offset: int) -> pd.Timestamp:
    """
    Retorna a data correspondente a `offset` dias úteis a partir de data_ref.
    offset > 0 → futuro; offset < 0 → passado.
    """
    bdays = datas_index.sort_values()
    # Encontra o dia útil mais próximo de data_ref
    pos = bdays.searchsorted(data_ref)
    pos = min(pos, len(bdays) - 1)
    target = pos + offset
    target = max(0, min(target, len(bdays) - 1))
    return bdays[target]


def definir_janelas(datas_index: pd.DatetimeIndex, ano: int) -> dict:
    """
    Define as janelas de análise para um ano eleitoral.
    Retorna dict com chave = nome_janela, valor = (data_inicio, data_fim).
    """
    dt_1turno = DATAS_PRIMEIRO_TURNO[ano]
    dt_2turno = DATAS_SEGUNDO_TURNO[ano]

    # Janela de Estimação: ano anterior completo
    est_inicio = pd.Timestamp(f"{ano - 1}-01-02")
    est_fim = pd.Timestamp(f"{ano - 1}-12-31")

    # Janela de Antecipação: 45 dias úteis antes do 1º turno
    antecip_inicio = obter_dias_uteis(datas_index, dt_1turno, -45)
    antecip_fim = obter_dias_uteis(datas_index, dt_1turno, -1)

    # Janela de Reação Curta (1º turno): [-5, +5] dias úteis
    reacao1_inicio = obter_dias_uteis(datas_index, dt_1turno, -5)
    reacao1_fim = obter_dias_uteis(datas_index, dt_1turno, +5)

    # Janela de Reação Curta (2º turno): [-5, +5] dias úteis
    reacao2_inicio = obter_dias_uteis(datas_index, dt_2turno, -5)
    reacao2_fim = obter_dias_uteis(datas_index, dt_2turno, +5)

    # Ciclo Interno: 1º sem vs 2º sem
    ciclo_1sem_inicio = pd.Timestamp(f"{ano}-01-02")
    ciclo_1sem_fim = pd.Timestamp(f"{ano}-06-30")
    ciclo_2sem_inicio = pd.Timestamp(f"{ano}-07-01")
    ciclo_2sem_fim = pd.Timestamp(f"{ano}-12-31")

    # Janela Estendida: últimos 6 meses do ano eleitoral
    estendida_inicio = pd.Timestamp(f"{ano}-07-01")
    estendida_fim = pd.Timestamp(f"{ano}-12-31")

    return {
        "estimacao": (est_inicio, est_fim),
        "antecipacao": (antecip_inicio, antecip_fim),
        "reacao_1turno": (reacao1_inicio, reacao1_fim),
        "reacao_2turno": (reacao2_inicio, reacao2_fim),
        "ciclo_1sem": (ciclo_1sem_inicio, ciclo_1sem_fim),
        "ciclo_2sem": (ciclo_2sem_inicio, ciclo_2sem_fim),
        "estendida": (estendida_inicio, estendida_fim),
    }


# ============================================================================
# ETAPA 4 – MODELO DE MERCADO E CÁLCULO DE AR / CAR
# ============================================================================

def estimar_ols(ret_ativo: pd.Series, ret_mercado: pd.Series):
    """
    Estima α e β via OLS: R_i = α + β·R_m + ε
    Retorna dict com alpha, beta, sigma_resid, r_squared, n_obs.
    """
    df = pd.DataFrame({"y": ret_ativo, "x": ret_mercado}).dropna()
    if len(df) < 30:
        return None

    X = np.column_stack([np.ones(len(df)), df["x"].values])
    Y = df["y"].values

    try:
        betas = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ betas
        residuos = Y - Y_hat
        ss_res = np.sum(residuos ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        sigma = np.sqrt(ss_res / (len(df) - 2)) if len(df) > 2 else np.nan
        return {
            "alpha": betas[0],
            "beta": betas[1],
            "sigma_resid": sigma,
            "r_squared": r2,
            "n_obs": len(df),
        }
    except Exception:
        return None


def calcular_ar(ret_ativo: pd.Series, ret_mercado: pd.Series,
                alpha: float, beta: float) -> pd.Series:
    """AR_t = R_{i,t} - (α + β·R_{m,t})"""
    df = pd.DataFrame({"y": ret_ativo, "x": ret_mercado}).dropna()
    ar = df["y"] - (alpha + beta * df["x"])
    return ar


def calcular_car(ar_series: pd.Series) -> float:
    """CAR = Σ AR_t"""
    return ar_series.sum()


def calcular_tstat_ar(ar_cross_section: np.ndarray) -> dict:
    """
    t-stat para AR médio cross-sectional (Eq. 6 de Silva et al., 2015).
    ar_cross_section = array de ARs de N setores/ativos no mesmo dia t.
    """
    n = len(ar_cross_section)
    if n < 2:
        return {"t_stat": np.nan, "p_value": np.nan}
    ar_mean = np.mean(ar_cross_section)
    sd = np.std(ar_cross_section, ddof=1)
    if sd == 0:
        return {"t_stat": np.nan, "p_value": np.nan}
    t_stat = ar_mean * np.sqrt(n) / sd
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return {"t_stat": t_stat, "p_value": p_value}


def calcular_tstat_car(car_values: np.ndarray, ar_series_list: list,
                       n_dias: int) -> dict:
    """
    t-stat para CAR com desvio padrão corrigido (Eq. 7-8 de Silva et al., 2015).
    csd_t = sqrt(t · var + 2·(t-1) · cov)
    """
    n = len(car_values)
    if n < 2:
        return {"t_stat": np.nan, "p_value": np.nan}

    car_mean = np.mean(car_values)

    # Calcular variância e covariância longitudinal
    variances = []
    covariances = []
    for ar_s in ar_series_list:
        if len(ar_s) < 2:
            continue
        variances.append(np.var(ar_s, ddof=1))
        if len(ar_s) > 1:
            ar_arr = ar_s.values if hasattr(ar_s, 'values') else np.array(ar_s)
            cov = np.cov(ar_arr[:-1], ar_arr[1:])[0, 1] if len(ar_arr) > 1 else 0
            covariances.append(cov)

    if len(variances) == 0:
        return {"t_stat": np.nan, "p_value": np.nan}

    var_mean = np.mean(variances)
    cov_mean = np.mean(covariances) if covariances else 0

    t = max(n_dias, 1)
    csd_t_sq = t * var_mean + 2 * (t - 1) * cov_mean
    if csd_t_sq <= 0:
        return {"t_stat": np.nan, "p_value": np.nan}
    csd_t = np.sqrt(csd_t_sq)

    t_stat = car_mean * np.sqrt(n) / csd_t
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), max(n - 1, 1)))
    return {"t_stat": t_stat, "p_value": p_value}


# ============================================================================
# ETAPA 5 – EXECUÇÃO DA ANÁLISE POR SETOR E ANO
# ============================================================================

def executar_analise(df_ret_setores: pd.DataFrame,
                     ret_ibov: pd.Series,
                     cenario: str = "completo") -> pd.DataFrame:
    """
    Executa a análise completa de Event Study para todos os setores e anos.
    cenario: "completo" ou "sem_crises" (exclui anos 2008/2020 da estimação).
    """
    log.info("=" * 70)
    log.info("EXECUTANDO ANÁLISE – Cenário: %s", cenario)
    log.info("=" * 70)

    resultados = []
    datas_index = df_ret_setores.dropna(how="all").index

    for ano in ANOS_ELEITORAIS:
        log.info("\n>>> Ano eleitoral: %d", ano)
        janelas = definir_janelas(datas_index, ano)

        est_ini, est_fim = janelas["estimacao"]

        for setor in df_ret_setores.columns:
            # Retornos do setor e do mercado na janela de estimação
            mask_est = (df_ret_setores.index >= est_ini) & (df_ret_setores.index <= est_fim)
            ret_setor_est = df_ret_setores.loc[mask_est, setor].dropna()
            ret_ibov_est = ret_ibov.loc[mask_est].dropna()

            # Estimar OLS
            params = estimar_ols(ret_setor_est, ret_ibov_est)
            if params is None:
                log.warning("  [%s] OLS falhou (dados insuficientes na estimação)", setor)
                continue

            alpha = params["alpha"]
            beta = params["beta"]

            # Para cada janela de evento
            janelas_evento = {
                "antecipacao": janelas["antecipacao"],
                "reacao_1turno": janelas["reacao_1turno"],
                "reacao_2turno": janelas["reacao_2turno"],
                "ciclo_1sem": janelas["ciclo_1sem"],
                "ciclo_2sem": janelas["ciclo_2sem"],
                "estendida": janelas["estendida"],
            }

            for nome_janela, (j_ini, j_fim) in janelas_evento.items():
                mask_j = (df_ret_setores.index >= j_ini) & (df_ret_setores.index <= j_fim)
                ret_setor_j = df_ret_setores.loc[mask_j, setor].dropna()
                ret_ibov_j = ret_ibov.loc[mask_j].dropna()

                if len(ret_setor_j) < 3:
                    continue

                ar = calcular_ar(ret_setor_j, ret_ibov_j, alpha, beta)
                car = calcular_car(ar)
                n_dias = len(ar)

                # t-stat simples (cross-section de 1 → usa longitudinal)
                ar_std = ar.std()
                t_stat = (car / n_dias) * np.sqrt(n_dias) / ar_std if ar_std > 0 else np.nan
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), max(n_dias - 1, 1))) if not np.isnan(t_stat) else np.nan

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
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "significativo_5pct": p_value < 0.05 if not np.isnan(p_value) else False,
                    "significativo_10pct": p_value < 0.10 if not np.isnan(p_value) else False,
                })

        log.info("  → %d observações acumuladas", len(resultados))

    df_res = pd.DataFrame(resultados)
    log.info("Total de resultados: %d", len(df_res))
    return df_res


# ============================================================================
# ETAPA 6 – ANÁLISE DE VOLATILIDADE (ELEITORAL vs NÃO-ELEITORAL)
# ============================================================================

def calcular_volatilidade_comparativa(df_ret_setores: pd.DataFrame) -> pd.DataFrame:
    """
    Compara volatilidade anualizada entre anos eleitorais e não-eleitorais.
    """
    log.info("Calculando volatilidade eleitoral vs. não-eleitoral ...")
    registros = []

    for setor in df_ret_setores.columns:
        for ano in range(2002, 2023):
            mask = (df_ret_setores.index.year == ano)
            ret = df_ret_setores.loc[mask, setor].dropna()
            if len(ret) < 20:
                continue
            vol_anual = ret.std() * np.sqrt(252)
            tipo = "Eleitoral" if ano in ANOS_ELEITORAIS else "Não-Eleitoral"
            registros.append({
                "setor": setor,
                "ano": ano,
                "tipo_ano": tipo,
                "volatilidade_anualizada": vol_anual,
            })

    return pd.DataFrame(registros)


# ============================================================================
# ETAPA 7 – BENCHMARKING REAL (Selic / IPCA)
# ============================================================================

def calcular_benchmarking_real(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona referência de Selic e IPCA (valores aproximados anuais)
    para cálculo de ganho/perda real.
    """
    # Selic acumulada anual aproximada (% a.a.)
    selic_anual = {
        2002: 0.1911, 2006: 0.1513, 2010: 0.0975,
        2014: 0.1115, 2018: 0.0640, 2022: 0.1275
    }
    # IPCA acumulado anual (% a.a.)
    ipca_anual = {
        2002: 0.1253, 2006: 0.0314, 2010: 0.0591,
        2014: 0.0641, 2018: 0.0375, 2022: 0.0562
    }

    df = df_resultados.copy()
    df["selic_anual"] = df["ano"].map(selic_anual)
    df["ipca_anual"] = df["ano"].map(ipca_anual)

    # CAR ajustado pela Selic proporcional ao nº de dias
    df["selic_proporcional"] = df.apply(
        lambda r: r["selic_anual"] * r["n_dias"] / 252 if pd.notna(r["selic_anual"]) else np.nan,
        axis=1
    )
    df["car_vs_selic"] = df["car"] - df["selic_proporcional"]
    df["ipca_proporcional"] = df.apply(
        lambda r: r["ipca_anual"] * r["n_dias"] / 252 if pd.notna(r["ipca_anual"]) else np.nan,
        axis=1
    )
    df["car_vs_ipca"] = df["car"] - df["ipca_proporcional"]

    return df


# ============================================================================
# ETAPA 8 – VISUALIZAÇÕES
# ============================================================================

def gerar_visualizacoes(df_resultados: pd.DataFrame,
                        df_volatilidade: pd.DataFrame,
                        output_dir: str):
    """Gera todos os gráficos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": (16, 9),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    # Filtra cenário completo
    df = df_resultados[df_resultados["cenario"] == "completo"]

    # --- HEATMAPS por janela ---
    janelas_heatmap = {
        "antecipacao": "Antecipação Eleitoral (45 d.u. antes do 1º turno)",
        "reacao_1turno": "Reação Imediata – 1º Turno ([-5, +5] d.u.)",
        "reacao_2turno": "Reação Imediata – 2º Turno ([-5, +5] d.u.)",
        "ciclo_2sem": "Ciclo Interno – 2º Semestre (Disputa)",
        "estendida": "Janela Estendida (últimos 6 meses)",
    }

    for janela_key, titulo in janelas_heatmap.items():
        df_j = df[df["janela"] == janela_key]
        if df_j.empty:
            continue

        pivot = df_j.pivot_table(index="setor", columns="ano", values="car", aggfunc="mean")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor="gray",
            cbar_kws={"label": "CAR Médio"},
            ax=ax,
        )
        ax.set_title(f"CAR por Setor e Ano – {titulo}", fontweight="bold", fontsize=14)
        ax.set_xlabel("Ano Eleitoral")
        ax.set_ylabel("Setor")
        plt.tight_layout()
        fname = f"heatmap_car_{janela_key}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ %s", fname)

    # --- EVOLUÇÃO TEMPORAL dos top 5 setores ---
    df_ant = df[df["janela"] == "antecipacao"]
    if not df_ant.empty:
        magnitude = df_ant.groupby("setor")["car"].apply(lambda x: abs(x).mean())
        top5 = magnitude.nlargest(5).index.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for idx, janela in enumerate(["antecipacao", "reacao_1turno"]):
            ax = axes[idx]
            for setor in top5:
                dados = df[(df["setor"] == setor) & (df["janela"] == janela)].sort_values("ano")
                if not dados.empty:
                    ax.plot(dados["ano"], dados["car"], marker="o", linewidth=2,
                            label=setor, markersize=7)
            ax.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax.set_xlabel("Ano Eleitoral")
            ax.set_ylabel("CAR")
            nome = "Antecipação" if janela == "antecipacao" else "Reação 1º Turno"
            ax.set_title(f"Evolução dos CARs – {nome}", fontweight="bold")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "evolucao_temporal_cars.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ evolucao_temporal_cars.png")

    # --- VOLATILIDADE COMPARATIVA ---
    if not df_volatilidade.empty:
        fig, ax = plt.subplots(figsize=(16, 8))
        vol_agg = df_volatilidade.groupby(["setor", "tipo_ano"])["volatilidade_anualizada"].mean().reset_index()
        pivot_vol = vol_agg.pivot(index="setor", columns="tipo_ano", values="volatilidade_anualizada")
        pivot_vol.plot(kind="barh", ax=ax, alpha=0.8, edgecolor="black")
        ax.set_title("Volatilidade Anualizada Média: Eleitoral vs. Não-Eleitoral", fontweight="bold")
        ax.set_xlabel("Volatilidade Anualizada")
        ax.set_ylabel("")
        ax.legend(title="Tipo de Ano")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volatilidade_eleitoral_vs_nao.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ volatilidade_eleitoral_vs_nao.png")


# ============================================================================
# ETAPA 9 – EXPORTAÇÃO
# ============================================================================

def exportar_resultados(df_resultados: pd.DataFrame,
                        df_volatilidade: pd.DataFrame,
                        output_dir: str):
    """Exporta resultados para Excel e CSV."""
    log.info("Exportando resultados ...")

    # CSV consolidado
    csv_path = os.path.join(output_dir, "resultados_consolidados.csv")
    df_resultados.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("  ✓ %s", csv_path)

    # Excel com múltiplas abas
    xlsx_path = os.path.join(output_dir, "analise_ciclos_eleitorais_completa.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Aba 1: Resultados completos
        df_resultados.to_excel(writer, sheet_name="Resultados Completos", index=False)

        # Aba 2: Resumo por setor (cenário completo, antecipação)
        for janela in ["antecipacao", "reacao_1turno", "reacao_2turno", "estendida"]:
            df_j = df_resultados[
                (df_resultados["cenario"] == "completo") &
                (df_resultados["janela"] == janela)
            ]
            if not df_j.empty:
                pivot = df_j.pivot_table(
                    index="setor", columns="ano",
                    values=["car", "t_stat", "p_value"],
                    aggfunc="mean"
                )
                sheet_name = f"CAR_{janela[:12]}"
                pivot.to_excel(writer, sheet_name=sheet_name)

        # Aba: Volatilidade
        if not df_volatilidade.empty:
            df_volatilidade.to_excel(writer, sheet_name="Volatilidade", index=False)

        # Aba: Sumário Executivo
        sumario = criar_sumario(df_resultados)
        sumario.to_excel(writer, sheet_name="Sumário Executivo", index=False)

    log.info("  ✓ %s", xlsx_path)


def criar_sumario(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """Cria sumário executivo."""
    df = df_resultados[df_resultados["cenario"] == "completo"]
    registros = []

    for janela in df["janela"].unique():
        df_j = df[df["janela"] == janela]
        registros.append({
            "Janela": janela,
            "CAR Médio Geral": f"{df_j['car'].mean():.4f}",
            "CAR Mediano": f"{df_j['car'].median():.4f}",
            "% Significativos (5%)": f"{df_j['significativo_5pct'].mean() * 100:.1f}%",
            "Setor Maior |CAR|": df_j.loc[df_j["car"].abs().idxmax(), "setor"],
            "Maior |CAR| Valor": f"{df_j.loc[df_j['car'].abs().idxmax(), 'car']:.4f}",
            "N Observações": len(df_j),
        })

    return pd.DataFrame(registros)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("█" * 72)
    print("█  ANÁLISE DE CICLOS ELEITORAIS NA B3 (2002-2022)                    █")
    print("█  Metodologia: Event Study · Modelo de Mercado (Silva et al., 2015) █")
    print("█" * 72)
    print()

    # Verificar arquivo de entrada
    if not os.path.exists(ARQUIVO_ENTRADA):
        log.error("Arquivo não encontrado: %s", ARQUIVO_ENTRADA)
        log.error("Coloque o arquivo na mesma pasta do script ou ajuste ARQUIVO_ENTRADA.")
        sys.exit(1)

    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # ETAPA 1: Carregar lista de empresas
    # ------------------------------------------------------------------
    df_empresas = carregar_lista_empresas(ARQUIVO_ENTRADA)

    # ------------------------------------------------------------------
    # ETAPA 2: Baixar preços (yfinance)
    # ------------------------------------------------------------------
    tickers_yf = df_empresas["TICKER_YF"].unique().tolist()

    # Adicionar Ibovespa à lista
    log.info("Total de tickers para download: %d", len(tickers_yf))

    # Cache: se já baixou, carrega do disco
    cache_precos = os.path.join(OUTPUT_DIR, "_cache_precos.pkl")
    cache_ibov = os.path.join(OUTPUT_DIR, "_cache_ibov.pkl")

    if os.path.exists(cache_precos) and os.path.exists(cache_ibov):
        log.info("Carregando preços do cache ...")
        df_precos = pd.read_pickle(cache_precos)
        ibov = pd.read_pickle(cache_ibov)
    else:
        df_precos = baixar_precos_yfinance(tickers_yf, "2000-01-01", "2023-12-31")
        ibov = baixar_ibovespa("2000-01-01", "2023-12-31")
        # Salvar cache
        df_precos.to_pickle(cache_precos)
        ibov.to_pickle(cache_ibov)
        log.info("Cache salvo em %s", OUTPUT_DIR)

    # ------------------------------------------------------------------
    # ETAPA 3: Filtro de existência
    # ------------------------------------------------------------------
    df_precos = aplicar_filtro_existencia(df_precos, df_empresas)

    # ------------------------------------------------------------------
    # ETAPA 4: Construir índices setoriais
    # ------------------------------------------------------------------
    df_precos_setores, df_ret_setores = construir_indices_setoriais(df_precos, df_empresas)

    # Retornos do Ibovespa
    ret_ibov = np.log(ibov / ibov.shift(1)).dropna()

    # Alinhar índices
    idx_comum = df_ret_setores.index.intersection(ret_ibov.index)
    df_ret_setores = df_ret_setores.loc[idx_comum]
    ret_ibov = ret_ibov.loc[idx_comum]

    # ------------------------------------------------------------------
    # ETAPA 5: Análise Event Study
    # ------------------------------------------------------------------
    # Cenário 1: Base completa
    df_res_completo = executar_analise(df_ret_setores, ret_ibov, cenario="completo")

    # Cenário 2: Sem crises (2008, 2020 excluídos)
    # Na prática, só afeta a estimação se o ano anterior for crise.
    # Filtramos os anos eleitorais que têm crise no ano anterior ou no próprio ano.
    df_ret_sem_crise = df_ret_setores.copy()
    mask_crise = df_ret_sem_crise.index.year.isin(ANOS_CRISE)
    df_ret_sem_crise.loc[mask_crise] = np.nan
    ret_ibov_sem_crise = ret_ibov.copy()
    ret_ibov_sem_crise.loc[mask_crise] = np.nan

    df_res_sem_crise = executar_analise(df_ret_sem_crise, ret_ibov_sem_crise,
                                         cenario="sem_crises")

    # Consolidar
    df_resultados = pd.concat([df_res_completo, df_res_sem_crise], ignore_index=True)

    # ------------------------------------------------------------------
    # ETAPA 6: Benchmarking real
    # ------------------------------------------------------------------
    df_resultados = calcular_benchmarking_real(df_resultados)

    # ------------------------------------------------------------------
    # ETAPA 7: Volatilidade comparativa
    # ------------------------------------------------------------------
    df_volatilidade = calcular_volatilidade_comparativa(df_ret_setores)

    # ------------------------------------------------------------------
    # ETAPA 8: Visualizações
    # ------------------------------------------------------------------
    log.info("\nGerando visualizações ...")
    gerar_visualizacoes(df_resultados, df_volatilidade, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # ETAPA 9: Exportação
    # ------------------------------------------------------------------
    exportar_resultados(df_resultados, df_volatilidade, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # RESUMO FINAL
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("RESUMO FINAL")
    print("=" * 72)

    df_comp = df_resultados[df_resultados["cenario"] == "completo"]
    print(f"\nObservações totais (cenário completo): {len(df_comp)}")
    print(f"Setores analisados: {df_comp['setor'].nunique()}")
    print(f"Anos eleitorais: {sorted(df_comp['ano'].unique())}")

    print("\nCAR Médio por Janela (cenário completo):")
    for janela in df_comp["janela"].unique():
        car_m = df_comp[df_comp["janela"] == janela]["car"].mean()
        sig = df_comp[df_comp["janela"] == janela]["significativo_5pct"].mean() * 100
        print(f"  {janela:25s}  CAR={car_m:+.4f}   sig(5%)={sig:.1f}%")

    print(f"\nArquivos salvos em: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 72)
    print("✓ ANÁLISE FINALIZADA COM SUCESSO!")
    print("=" * 72)


if __name__ == "__main__":
    main()
