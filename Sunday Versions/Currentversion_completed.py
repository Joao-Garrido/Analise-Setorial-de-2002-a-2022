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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ElectoralCycles")


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

ARQUIVO_ENTRADA = "resultados_analise_b3_com_tickers.xlsx"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Outputs New")

# ---- Datas eleitorais (1º e 2º turnos) -----------------------------------
DATAS_PRIMEIRO_TURNO = {
    2002: pd.Timestamp("2002-10-06"), 2006: pd.Timestamp("2006-10-01"),
    2010: pd.Timestamp("2010-10-03"), 2014: pd.Timestamp("2014-10-05"),
    2018: pd.Timestamp("2018-10-07"), 2022: pd.Timestamp("2022-10-02"),
}
DATAS_SEGUNDO_TURNO = {
    2002: pd.Timestamp("2002-10-27"), 2006: pd.Timestamp("2006-10-29"),
    2010: pd.Timestamp("2010-10-31"), 2014: pd.Timestamp("2014-10-26"),
    2018: pd.Timestamp("2018-10-28"), 2022: pd.Timestamp("2022-10-30"),
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

# ---- Janela entre turnos (máxima incerteza política) ---------------------
# De [-5 d.u. antes do 1ºT] até [-1 d.u. antes do 2ºT]
# Tamanho variável (~15-18 d.u. conforme ano)
JANELA_ENTRE_TURNOS = True

# ---- Janelas semestrais (ciclo interno) ----------------------------------
JANELAS_CICLO = True          # 1º sem (expectativa) vs 2º sem (disputa)

# ---- Janela estendida ----------------------------------------------------
JANELA_ESTENDIDA = True       # últimos 6 meses do ano eleitoral

# ---- Janelas-espelho (T-1) — comparativo de normalidade ------------------
# Replica todas as janelas no ano anterior (mesmo período calendário)
# Avisos dos espelhos:
#   2002→2001: crise Argentina + 11/Set
#   2006→2005: Mensalão
#   2010→2009: pós-crise subprime
#   2014→2013: Jornadas de Junho
#   2018→2017: Joesley Day
#   2022→2021: boom de IPOs
JANELAS_ESPELHO = True

# ---- Filtragem e robustez ------------------------------------------------
MIN_PREGOES_PCT = 0.80        # Mínimo 80% de pregões no ano para inclusão
MIN_EMPRESAS_SETOR = 5        # Corte: mínimo N empresas por setor/ano
ANOS_CRISE = [2008, 2014, 2020]
N_PLACEBO_EVENTS = 1000
WINSORIZE_PCT = 0.01          # Winsorização a 1%/99%
MIN_PRECO = 1.0               # Filtro penny stocks: preço médio < R$1

# ---- Selic / IPCA — TODOS os anos (2001–2023), fonte BCB -----------------
SELIC_ANUAL = {
    2001: 0.1725, 2002: 0.1911, 2003: 0.2335, 2004: 0.1617,
    2005: 0.1913, 2006: 0.1513, 2007: 0.1185, 2008: 0.1266,
    2009: 0.0966, 2010: 0.0975, 2011: 0.1175, 2012: 0.0840,
    2013: 0.0815, 2014: 0.1115, 2015: 0.1328, 2016: 0.1404,
    2017: 0.0993, 2018: 0.0640, 2019: 0.0596, 2020: 0.0276,
    2021: 0.0440, 2022: 0.1275, 2023: 0.1325,
}
IPCA_ANUAL = {
    2001: 0.0753, 2002: 0.1253, 2003: 0.0930, 2004: 0.0764,
    2005: 0.0569, 2006: 0.0314, 2007: 0.0446, 2008: 0.0590,
    2009: 0.0431, 2010: 0.0591, 2011: 0.0650, 2012: 0.0584,
    2013: 0.0591, 2014: 0.0641, 2015: 0.1067, 2016: 0.0629,
    2017: 0.0295, 2018: 0.0375, 2019: 0.0431, 2020: 0.0452,
    2021: 0.1006, 2022: 0.0562, 2023: 0.0462,
}
# Selic diária (proxy CDI) para cálculo de Sharpe — TODOS os anos
SELIC_DIARIA = {k: (1 + v) ** (1/252) - 1 for k, v in SELIC_ANUAL.items()}

# ---- NEFIN-USP: Fatores Fama-French Brasil --------------------------------
# O NEFIN (Núcleo de Economia Financeira da USP) calcula diariamente os
# fatores de risco Fama-French adaptados para o Brasil:
#   - Mkt-Rf : prêmio de risco de mercado (retorno mercado − CDI)
#   - SMB    : prêmio de tamanho (small caps − large caps)
#   - HML    : prêmio de valor (alto B/M − baixo B/M)
#   - WML    : prêmio de momentum (vencedoras − perdedoras)
#   - IML    : prêmio de iliquidez (ilíquidas − líquidas)
#   - Rf     : taxa livre de risco diária (CDI)
#
# Usamos para rodar um modelo alternativo de robustez (Fama-French 3):
#   R_i − Rf = α + β·MktRf + s·SMB + h·HML + ε
#
# Se o CAR permanece significativo após controlar por tamanho e valor,
# o resultado é mais robusto contra a crítica de que o efeito capturado
# seria apenas exposição a small caps ou empresas de valor.
#
# Fonte: https://nefin.com.br/data/risk_factors.html
# Se o download falhar (site fora, sem internet), o script continua
# normalmente apenas com o Modelo de Mercado (CAPM).
NEFIN_CAMINHO_LOCAL = "nefin_factors.csv"
NEFIN_URL = "https://nefin.com.br/resources/risk_factors/nefin_factors.csv"

# ---- Ticker mapping (De-Para: tickers antigos → atuais) ------------------
TICKER_MAPPING = {
    "VVAR3": "BHIA3", "BTOW3": "AMER3", "LAME4": "LAME3", "PCAR4": "PCAR3",
    "KROT3": "COGN3", "ESTC3": "YDUQ3", "RAIL3": "RUMO3",
    "BVMF3": "B3SA3", "CTIP3": "B3SA3",
    "BRIN3": "BRML3", "BRML3": "ALOS3", "SMLE3": "SMFT3",
    "LINX3": "STNE3", "VIVT4": "VIVT3", "TIMP3": "TIMS3",
    "QGEP3": "BRAV3", "GNDI3": "HAPV3", "FIBR3": "SUZB3",
}

# ---- Alpha Vantage (fallback) -------------------------------------------
AV_LIMITE_DIARIO = 25

try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    if not ALPHA_VANTAGE_KEY:
        ALPHA_VANTAGE_KEY = 'SSU8PQU94ONCSLAF'
except ImportError:
    ALPHA_VANTAGE_KEY = None
    TimeSeries = None


# ============================================================================
# ETAPA 1 – INGESTÃO DE DADOS
# ============================================================================

def carregar_lista_empresas(caminho: str) -> pd.DataFrame:
    """
    Carrega lista de empresas da planilha, aplica mapeamento De-Para,
    e lê coluna ESTATAL direto da planilha.
    """
    log.info("=" * 70)
    log.info("ETAPA 1: INGESTÃO DE DADOS")
    log.info("=" * 70)

    df = pd.read_excel(caminho, sheet_name="LISTA FINAL (Cont+IPOs-Canc)")
    df = df.dropna(subset=["TICKER", "SETOR_B3"])

    df["DT_REG"] = pd.to_datetime(df["DT_REG"], errors="coerce")

    # Coluna ESTATAL: converte "Sim"/"Não" para booleano
    df["ESTATAL"] = df["ESTATAL"].str.strip().str.upper().eq("SIM")

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

    for _, r in df[df["ESTATAL"]].iterrows():
        log.info("    [ESTATAL] %s — %s (%s)",
                 r["TICKER_ORIGINAL"], str(r["DENOM_SOCIAL"])[:45], r["SETOR_B3"])

    return df


def baixar_precos_yfinance(tickers: list, start="2001-01-01", end="2023-12-31"):
    """Baixa preços ajustados e volumes via yfinance (primário) + Alpha Vantage (fallback)."""
    import yfinance as yf

    log.info("Baixando preços de %d tickers (primário: yfinance, fallback: Alpha Vantage)...",
             len(tickers))
    all_close = {}
    all_volume = {}
    diagnostico_list = []

    blocos = [tickers[i:i+50] for i in range(0, len(tickers), 50)]

    # Cliente Alpha Vantage (inicializa só se key válida)
    ts_av = None
    if ALPHA_VANTAGE_KEY and TimeSeries:
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
                close = data[["Close"]]; close.columns = bloco
                volume = data[["Volume"]]; volume.columns = bloco

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

    # Fallback Alpha Vantage para falhas (com limite diário)
    falhas = [d for d in diagnostico_list if d["status"] == "falha"]
    av_n = 0
    if falhas and ts_av:
        log.info("  Fallback Alpha Vantage para %d tickers falhos (limite: %d/dia)...",
                 len(falhas), AV_LIMITE_DIARIO)
        for diag in falhas:
            if av_n >= AV_LIMITE_DIARIO:
                log.warning("  Limite diário Alpha Vantage atingido (%d calls). "
                            "Restantes ficam como falha.", AV_LIMITE_DIARIO)
                break

            t = diag["ticker_yf"]
            try:
                data_av, _ = ts_av.get_daily(symbol=t, outputsize='full')
                data_av = data_av.rename(columns={
                    '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                    '4. close': 'Close', '5. volume': 'Volume'
                })
                av_n += 1

                if data_av['Close'].notna().sum() > 0:
                    all_close[t] = data_av['Close']
                    all_volume[t] = data_av['Volume']
                    diag.update({"status": "ok", "fonte": "alpha_vantage", "motivo": ""})
                    log.info("    ✓ %s recuperado via Alpha Vantage (%d/%d)",
                             t, av_n, AV_LIMITE_DIARIO)

                time.sleep(2)  # 1 req/segundo mínimo, 2s para margem

            except Exception as av_e:
                av_n += 1
                log.warning("  Falha Alpha Vantage %s: %s", t, str(av_e)[:80])
                diag["motivo"] += f"; erro_av: {str(av_e)[:80]}"
                time.sleep(2)

        log.info("  Alpha Vantage: %d chamadas feitas, %d tickers recuperados",
                 av_n, sum(1 for d in falhas if d.get("fonte") == "alpha_vantage"))

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
    Baixa fatores Fama-French brasileiros do NEFIN-USP (CSV único).
    Tenta primeiro arquivo local, depois URL.
    Retorna DataFrame com colunas: Mkt_Rf, SMB, HML, Rf. Ou None.
    """
    log.info("Carregando fatores NEFIN-USP ...")

    df = None
    # Tenta arquivo local primeiro
    if os.path.exists(NEFIN_CAMINHO_LOCAL):
        try:
            df = pd.read_csv(NEFIN_CAMINHO_LOCAL)
            log.info("  → Arquivo local: %s", NEFIN_CAMINHO_LOCAL)
        except Exception:
            pass

    # Fallback: URL
    if df is None:
        try:
            df = pd.read_csv(NEFIN_URL)
            log.info("  → URL NEFIN")
        except Exception as e:
            log.warning("  Falha ao baixar NEFIN: %s. FF3 desabilitado.", e)
            return None

    try:
        # Tratamento de data
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
            df = df.set_index("date")
        elif {"year", "month", "day"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            df = df.set_index("date")
        else:
            log.warning("  Formato de data desconhecido no CSV do NEFIN.")
            return None

        # Renomear para o padrão do script
        df = df.rename(columns={"Rm_minus_Rf": "Mkt_Rf", "Risk_Free": "Rf"})

        cols_necessarias = ["Mkt_Rf", "SMB", "HML", "Rf"]
        if not set(cols_necessarias).issubset(df.columns):
            log.warning("  Colunas faltando no NEFIN. Encontradas: %s", df.columns.tolist())
            return None

        df_factors = df[cols_necessarias]
        log.info("  → Fatores NEFIN: %d obs (de %s a %s)",
                 len(df_factors),
                 df_factors.index.min().strftime('%Y-%m'),
                 df_factors.index.max().strftime('%Y-%m'))
        return df_factors

    except Exception as e:
        log.warning("  Falha ao processar NEFIN: %s. FF3 desabilitado.", e)
        return None


def aplicar_filtro_existencia(df_precos, df_empresas):
    """Invalida (NaN) preços anteriores à data de registro (DT_REG)."""
    log.info("Aplicando filtro de existência (DT_REG)...")
    df = df_precos.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    lookup = {}
    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        reg = pd.to_datetime(row["DT_REG"], errors='coerce')
        if tk in lookup:
            if pd.notna(reg) and (pd.isna(lookup[tk]) or reg < lookup[tk]):
                lookup[tk] = reg
        else:
            lookup[tk] = reg

    total = 0
    for col in df.columns:
        if col in lookup and pd.notna(lookup[col]):
            mask = df.index < lookup[col]
            qtd = df.loc[mask, col].notna().sum()
            if qtd > 0:
                total += qtd
                df.loc[mask, col] = np.nan

    log.info("  → %d obs pré-registro removidas", total)
    return df


# ============================================================================
# ETAPA 2 – ÍNDICES SETORIAIS
# ============================================================================

def winsorizar(series, pct=WINSORIZE_PCT):
    """Winsoriza retornos no percentil pct / (1-pct)."""
    lower = series.quantile(pct)
    upper = series.quantile(1 - pct)
    return series.clip(lower, upper)


def aplicar_filtro_liquidez(df_ret, ano):
    """Retorna lista de tickers com ≥ MIN_PREGOES_PCT pregões no ano."""
    mask_ano = df_ret.index.year == ano
    ret_ano = df_ret.loc[mask_ano]
    if ret_ano.empty:
        return []
    min_obs = int(mask_ano.sum() * MIN_PREGOES_PCT)
    obs_por_ticker = ret_ano.notna().sum()
    return obs_por_ticker[obs_por_ticker >= min_obs].index.tolist()


def aplicar_filtro_penny(df_precos, ano):
    """Retorna lista de tickers com preço médio ≥ MIN_PRECO no ano."""
    mask = df_precos.index.year == ano
    p = df_precos.loc[mask]
    if p.empty:
        return []
    return p.mean().pipe(lambda s: s[s >= MIN_PRECO]).index.tolist()


def construir_indices_setoriais(df_precos, df_volumes, df_empresas):
    """
    Constrói índices setoriais EW e VW com filtros de liquidez, penny e winsorização.
    """
    log.info("=" * 70)
    log.info("ETAPA 2: ÍNDICES SETORIAIS (liquidez≥%.0f%% | preço≥R$%.0f | winsor %.0f%%)",
             MIN_PREGOES_PCT*100, MIN_PRECO, WINSORIZE_PCT*100)
    log.info("=" * 70)

    ret = np.log(df_precos / df_precos.shift(1))
    vol_fin = df_precos * df_volumes
    t2s = {row["TICKER_YF"]: row["SETOR_B3"] for _, row in df_empresas.iterrows()}

    series_ew, series_vw, composicao = {}, {}, []

    for setor in SETORES_B3:
        cols_setor = [c for c in ret.columns if t2s.get(c) == setor]
        if not cols_setor:
            log.warning("  Sem tickers para '%s'", setor)
            continue

        ew_parts, vw_parts = [], []

        for ano in range(2001, 2024):
            mask_ano = ret.index.year == ano
            if mask_ano.sum() == 0:
                continue

            ok_liq = set(aplicar_filtro_liquidez(ret, ano))
            ok_penny = set(aplicar_filtro_penny(df_precos, ano))
            cols_ano = [c for c in cols_setor if c in ok_liq and c in ok_penny]

            composicao.append({
                "setor": setor, "ano": ano,
                "n_tickers_setor": len(cols_setor),
                "n_com_dados_liquidos": len(cols_ano),
                "filtro_removeu": len(cols_setor) - len(cols_ano),
            })

            if len(cols_ano) < MIN_EMPRESAS_SETOR:
                continue

            # Winsorização dos retornos antes de agregar
            ret_ano = ret.loc[mask_ano, cols_ano].apply(winsorizar)
            ew_parts.append(ret_ano.mean(axis=1))

            # VW: ponderado por volume financeiro médio 20d
            vf_ano = vol_fin.loc[mask_ano, cols_ano].rolling(20, min_periods=5).mean()
            vf_sum = vf_ano.sum(axis=1)
            weights = vf_ano.div(vf_sum.replace(0, np.nan), axis=0)
            vw_parts.append((ret_ano * weights).sum(axis=1))

        if ew_parts:
            series_ew[setor] = pd.concat(ew_parts).sort_index()
        if vw_parts:
            series_vw[setor] = pd.concat(vw_parts).sort_index()

        comp_s = [c for c in composicao if c["setor"] == setor]
        anos_val = sum(1 for c in comp_s if c["n_com_dados_liquidos"] >= MIN_EMPRESAS_SETOR)
        log.info("  %s: %d tickers | %d/%d anos válidos",
                 setor, len(cols_setor), anos_val, len(comp_s))

    df_ret_ew = pd.DataFrame(series_ew)
    df_ret_vw = pd.DataFrame(series_vw)
    log.info("  → EW: %d setores × %d datas | VW: %d setores × %d datas",
             df_ret_ew.shape[1], len(df_ret_ew), df_ret_vw.shape[1], len(df_ret_vw))
    return df_ret_ew, df_ret_vw, pd.DataFrame(composicao)


def gerar_tabela_sobrevivencia(df_empresas, df_precos):
    """Tabela de cobertura: N esperado vs. N com dados por setor/ano."""
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
            n_dados = precos_ano[cols].notna().any().sum() if cols else 0
            n_esp = ((df_empresas["SETOR_B3"] == setor) &
                     (df_empresas["DT_REG"].dt.year <= ano)).sum()
            registros.append({
                "ano": ano, "setor": setor, "n_esperado": n_esp,
                "n_com_dados": n_dados,
                "cobertura_pct": round(100 * n_dados / max(n_esp, 1), 1),
            })
    df_sobrev = pd.DataFrame(registros)
    for ano in ANOS_ELEITORAIS:
        sub = df_sobrev[df_sobrev["ano"] == ano]
        if not sub.empty:
            log.info("  %d: cobertura média %.1f%%", ano, sub["cobertura_pct"].mean())
    return df_sobrev


def gerar_estatisticas_descritivas(df_ret_ew, df_ret_vw):
    """Table 1: estatísticas descritivas dos índices setoriais."""
    log.info("Gerando estatísticas descritivas (Table 1) ...")
    reg = []
    for pond, df_r in [("EW", df_ret_ew), ("VW", df_ret_vw)]:
        for setor in df_r.columns:
            r = df_r[setor].dropna()
            if len(r) < 30:
                continue
            reg.append({
                "ponderacao": pond, "setor": setor, "n_obs": len(r),
                "media_diaria": r.mean(), "media_anual": r.mean() * 252,
                "desvio_diario": r.std(), "vol_anualizada": r.std() * np.sqrt(252),
                "assimetria": r.skew(), "curtose": r.kurtosis(),
                "minimo": r.min(), "maximo": r.max(),
                "autocorr_1": r.autocorr(lag=1),
            })
    df_desc = pd.DataFrame(reg)
    log.info("  → %d linhas (Table 1)", len(df_desc))
    return df_desc


# ============================================================================
# ETAPA 3 – JANELAS
# ============================================================================

def offset_du(bdates, data_ref, offset):
    """Dia útil com offset a partir de data_ref."""
    bdays = bdates.sort_values()
    pos = min(bdays.searchsorted(data_ref), len(bdays) - 1)
    target = max(0, min(pos + offset, len(bdays) - 1))
    return bdays[target]


def janela_estimacao(bdates, dt_evento):
    """[-252, -30] d.u. antes do evento."""
    return (offset_du(bdates, dt_evento, ESTIMACAO_INICIO_DU),
            offset_du(bdates, dt_evento, ESTIMACAO_FIM_DU))


def janelas_evento_1t(bdates, ano):
    """Janelas de evento relativas ao 1º turno."""
    dt1 = DATAS_PRIMEIRO_TURNO[ano]
    return {nome: (offset_du(bdates, dt1, di), offset_du(bdates, dt1, df_))
            for nome, (di, df_) in JANELAS_EVENTO_1T.items()}


def janelas_evento_2t(bdates, ano):
    """Janelas de evento relativas ao 2º turno."""
    dt2 = DATAS_SEGUNDO_TURNO.get(ano)
    if dt2 is None:
        return {}
    return {nome: (offset_du(bdates, dt2, di), offset_du(bdates, dt2, df_))
            for nome, (di, df_) in JANELAS_EVENTO_2T.items()}


def janela_entre_turnos_fn(bdates, ano):
    """De [-5 d.u. antes do 1ºT] até [-1 d.u. antes do 2ºT]. Tamanho variável."""
    dt1 = DATAS_PRIMEIRO_TURNO[ano]
    dt2 = DATAS_SEGUNDO_TURNO.get(ano)
    if dt2 is None:
        return {}
    ini = offset_du(bdates, dt1, -5)
    fim = offset_du(bdates, dt2, -1)
    return {"entre_turnos": (ini, fim)}


def janelas_ciclo_fn(bdates, ano):
    """Semestrais com datas em dias úteis reais."""
    bdays = bdates.sort_values()
    d1, d2, d3, d4 = (pd.Timestamp(f"{ano}-01-02"), pd.Timestamp(f"{ano}-06-30"),
                       pd.Timestamp(f"{ano}-07-01"), pd.Timestamp(f"{ano}-12-31"))
    i1 = bdays[bdays >= d1].min() if (bdays >= d1).any() else d1
    f1 = bdays[bdays <= d2].max() if (bdays <= d2).any() else d2
    i2 = bdays[bdays >= d3].min() if (bdays >= d3).any() else d3
    f2 = bdays[bdays <= d4].max() if (bdays <= d4).any() else d4
    return {"ciclo_1sem": (i1, f1), "ciclo_2sem": (i2, f2)}


def janela_estendida_fn(bdates, ano):
    """Últimos 6 meses em dias úteis reais."""
    bdays = bdates.sort_values()
    d1, d2 = pd.Timestamp(f"{ano}-07-01"), pd.Timestamp(f"{ano}-12-31")
    i = bdays[bdays >= d1].min() if (bdays >= d1).any() else d1
    f = bdays[bdays <= d2].max() if (bdays <= d2).any() else d2
    return {"estendida": (i, f)}


def janelas_espelho_fn(bdates, jans_originais):
    """
    Replica janelas no ano T-1 (mesmo período calendário).
    Retorna dict com prefix 'espelho_'.
    """
    espelhos = {}
    bdays = bdates.sort_values()
    for nome, (ini, fim) in jans_originais.items():
        try:
            ini_esp = ini - pd.DateOffset(years=1)
            fim_esp = fim - pd.DateOffset(years=1)
            i_e = bdays[bdays >= ini_esp].min() if (bdays >= ini_esp).any() else ini_esp
            f_e = bdays[bdays <= fim_esp].max() if (bdays <= fim_esp).any() else fim_esp
            if pd.notna(i_e) and pd.notna(f_e) and i_e < f_e:
                espelhos[f"espelho_{nome}"] = (i_e, f_e)
        except Exception:
            pass
    return espelhos


def coletar_todas_janelas(bdates, ano):
    """Coleta todas as janelas ativas para um ano eleitoral."""
    jans = janelas_evento_1t(bdates, ano)
    jans.update(janelas_evento_2t(bdates, ano))
    if JANELA_ENTRE_TURNOS:
        jans.update(janela_entre_turnos_fn(bdates, ano))
    if JANELAS_CICLO:
        jans.update(janelas_ciclo_fn(bdates, ano))
    if JANELA_ESTENDIDA:
        jans.update(janela_estendida_fn(bdates, ano))
    if JANELAS_ESPELHO:
        principais = {k: v for k, v in jans.items() if not k.startswith("espelho_")}
        jans.update(janelas_espelho_fn(bdates, principais))
    return jans


# ============================================================================
# ETAPA 4 – MODELOS E INFERÊNCIA ESTATÍSTICA
# ============================================================================

def estimar_ols_simples(ret_y, ret_x):
    """OLS manual: y = α + β·x. Retorna dict ou None."""
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
            "alpha": coef[0], "beta": coef[1],
            "sigma_resid": np.sqrt(ss_res / (len(df) - 2)),
            "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else 0,
            "n_obs_est": len(df)
        }
    except Exception:
        return None


def estimar_ols_hac(ret_y, ret_x, maxlags=1):
    """OLS com erros padrão HAC (Newey-West) via statsmodels."""
    import statsmodels.api as sm
    df = pd.DataFrame({"y": ret_y, "x": ret_x}).dropna()
    if len(df) < 30:
        return None
    X = sm.add_constant(df["x"])
    try:
        model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        return {
            "alpha": model.params.iloc[0], "beta": model.params.iloc[1],
            "alpha_pv": model.pvalues.iloc[0], "beta_pv": model.pvalues.iloc[1],
            "sigma_resid": np.sqrt(model.mse_resid),
            "r_squared": model.rsquared, "n_obs_est": int(model.nobs)
        }
    except Exception:
        return None


def estimar_ff3(ret_y, df_factors_window):
    """Fama-French 3: R_i - Rf = α + β·MktRf + s·SMB + h·HML + ε"""
    import statsmodels.api as sm
    common = ret_y.index.intersection(df_factors_window.index)
    if len(common) < 30:
        return None
    y = ret_y.loc[common]
    fac = df_factors_window.loc[common]
    y_excess = y - fac["Rf"] if "Rf" in fac.columns else y
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
    except Exception:
        return None


def calcular_ar(ret_y, ret_x, alpha, beta):
    """AR = R_i - (α + β·R_m)"""
    df = pd.DataFrame({"y": ret_y, "x": ret_x}).dropna()
    return df["y"] - (alpha + beta * df["x"])


def calcular_ar_ff3(ret_y, df_factors_window, params_ff3):
    """AR = R_i - Rf - (α + β_mkt·MktRf + β_smb·SMB + β_hml·HML)"""
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
    """t-stat CAR corrigido por autocorrelação (Silva et al., 2015)."""
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
    t = max(n_dias, 1)
    csd_sq = t * np.mean(variances) + 2 * max(t - 1, 0) * (np.mean(covariances) if covariances else 0)
    if csd_sq <= 0:
        return {"t_silva": np.nan, "p_silva": np.nan}
    t_stat = car_mean * np.sqrt(n) / np.sqrt(csd_sq)
    return {"t_silva": t_stat, "p_silva": 2 * (1 - stats.t.cdf(abs(t_stat), max(n - 1, 1)))}


def tstat_car_bmp(car_values, sigma_resids, n_dias):
    """Boehmer-Musumeci-Poulsen (1991). SCAR = CAR / (σ·√T)."""
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
    return {"t_bmp": t_bmp, "p_bmp": 2 * (1 - stats.t.cdf(abs(t_bmp), len(scar) - 1))}


# ============================================================================
# ETAPA 5 – EVENT STUDY (MODELO DE MERCADO)
# ============================================================================

def executar_analise(df_ret_setores, ret_ibov, cenario="completo", ponderacao="EW"):
    """Event Study: Modelo de Mercado com OLS simples + HAC."""
    log.info("  EVENT STUDY [%s / %s] ...", cenario, ponderacao)
    bdates = df_ret_setores.dropna(how="all").index
    resultados = []

    for ano in ANOS_ELEITORAIS:
        dt1 = DATAS_PRIMEIRO_TURNO[ano]
        est_ini, est_fim = janela_estimacao(bdates, dt1)

        # Coletar todas as janelas
        jans = coletar_todas_janelas(bdates, ano)

        for setor in df_ret_setores.columns:
            m_est = (df_ret_setores.index >= est_ini) & (df_ret_setores.index <= est_fim)
            rs_est = df_ret_setores.loc[m_est, setor].dropna()
            rm_est = ret_ibov.loc[m_est].dropna()

            p = estimar_ols_simples(rs_est, rm_est)
            if p is None:
                continue
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
                p_s = 2 * (1 - stats.t.cdf(abs(t_s), max(nd - 1, 1))) if not np.isnan(t_s) else np.nan

                resultados.append({
                    "cenario": cenario, "ponderacao": ponderacao,
                    "ano": ano, "setor": setor, "janela": nome_j,
                    "car": car, "ar_medio": ar.mean(), "ar_std": ar_std, "n_dias": nd,
                    "alpha": p["alpha"], "beta": p["beta"],
                    "r_squared": p["r_squared"], "sigma_resid": p["sigma_resid"],
                    "n_obs_est": p["n_obs_est"],
                    "alpha_hac_pv": p_hac["alpha_pv"] if p_hac else np.nan,
                    "beta_hac_pv": p_hac["beta_pv"] if p_hac else np.nan,
                    "est_inicio": est_ini, "est_fim": est_fim,
                    "janela_inicio": ji, "janela_fim": jf,
                    "t_simples": t_s, "p_simples": p_s,
                    "t_silva": np.nan, "p_silva": np.nan,
                    "t_bmp": np.nan, "p_bmp": np.nan,
                })

    df_res = pd.DataFrame(resultados)

    # Cross-sectional tests por (ano, janela)
    if not df_res.empty:
        for (ano, jan), grp in df_res.groupby(["ano", "janela"]):
            cars = grp["car"].values
            sigmas = grp["sigma_resid"].values
            nd = int(grp["n_dias"].median())
            ar_lists = []
            for _, row in grp.iterrows():
                m = (df_ret_setores.index >= row["janela_inicio"]) & \
                    (df_ret_setores.index <= row["janela_fim"])
                ar_lists.append(calcular_ar(
                    df_ret_setores.loc[m, row["setor"]].dropna(),
                    ret_ibov.loc[m].dropna(), row["alpha"], row["beta"]))

            silva = tstat_car_silva(cars, ar_lists, nd)
            bmp = tstat_car_bmp(cars, sigmas, nd)
            mask = ((df_res["ano"] == ano) & (df_res["janela"] == jan) &
                    (df_res["cenario"] == cenario) & (df_res["ponderacao"] == ponderacao))
            df_res.loc[mask, "t_silva"] = silva["t_silva"]
            df_res.loc[mask, "p_silva"] = silva["p_silva"]
            df_res.loc[mask, "t_bmp"] = bmp["t_bmp"]
            df_res.loc[mask, "p_bmp"] = bmp["p_bmp"]

    for col, pv in [("sig5_simples", "p_simples"), ("sig5_silva", "p_silva"), ("sig5_bmp", "p_bmp")]:
        df_res[col] = df_res[pv] < 0.05
    return df_res


# ============================================================================
# ETAPA 5b – EVENT STUDY (FAMA-FRENCH 3)
# ============================================================================

def executar_analise_ff3(df_ret_setores, df_factors, cenario="completo", ponderacao="EW"):
    """Event Study com Fama-French 3 fatores (NEFIN)."""
    if df_factors is None:
        log.info("  FF3 desabilitado (sem fatores NEFIN)")
        return pd.DataFrame()

    log.info("  EVENT STUDY FF3 [%s / %s] ...", cenario, ponderacao)
    bdates = df_ret_setores.dropna(how="all").index
    resultados = []

    for ano in ANOS_ELEITORAIS:
        dt1 = DATAS_PRIMEIRO_TURNO[ano]
        est_ini, est_fim = janela_estimacao(bdates, dt1)

        # FF3 roda nas janelas de evento (sem ciclo/estendida/espelho)
        jans = janelas_evento_1t(bdates, ano)
        jans.update(janelas_evento_2t(bdates, ano))
        if JANELA_ENTRE_TURNOS:
            jans.update(janela_entre_turnos_fn(bdates, ano))

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
                t_s = (ar.mean() * np.sqrt(nd) / ar_std) if ar_std > 0 else np.nan
                p_s = 2 * (1 - stats.t.cdf(abs(t_s), max(nd - 1, 1))) if not np.isnan(t_s) else np.nan

                resultados.append({
                    "cenario": cenario, "ponderacao": ponderacao, "modelo": "FF3",
                    "ano": ano, "setor": setor, "janela": nome_j,
                    "car_ff3": car, "ar_medio_ff3": ar.mean(), "n_dias": nd,
                    "alpha_ff3": pff["alpha_ff3"], "beta_mkt": pff["beta_mkt"],
                    "beta_smb": pff["beta_smb"], "beta_hml": pff["beta_hml"],
                    "r_squared_ff3": pff["r_squared_ff3"],
                    "t_simples_ff3": t_s, "p_simples_ff3": p_s,
                    "sig5_ff3": p_s < 0.05 if not np.isnan(p_s) else False,
                })

    return pd.DataFrame(resultados)


# ============================================================================
# ETAPA 6 – TESTE PLACEBO
# ============================================================================

def executar_teste_placebo(df_ret, ret_ibov, n_placebos=N_PLACEBO_EVENTS, seed=42):
    log.info("=" * 70)
    log.info("ETAPA 6: TESTE PLACEBO (%d pseudo-eventos)", n_placebos)
    log.info("=" * 70)
    np.random.seed(seed)
    bdates = df_ret.dropna(how="all").index
    anos_ne = [a for a in range(2003, 2022) if a not in ANOS_ELEITORAIS]
    resultados = []
    for i in range(n_placebos):
        ano = np.random.choice(anos_ne)
        dt = pd.Timestamp(f"{ano}-10-05")
        ei, ef = janela_estimacao(bdates, dt)
        windows = [
            ("placebo_antecip", offset_du(bdates, dt, -45), offset_du(bdates, dt, -1)),
            ("placebo_reacao", offset_du(bdates, dt, -5), offset_du(bdates, dt, +5)),
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
                    "placebo_id": i, "ano_placebo": ano, "setor": setor,
                    "janela": nome, "car": ar.sum(), "n_dias": len(ar)
                })
    df_p = pd.DataFrame(resultados)
    log.info("  → %d obs placebo", len(df_p))
    return df_p


# ============================================================================
# ETAPA 6b – DiD (ESTATAIS vs. PRIVADAS — nível empresa)
# ============================================================================

def executar_did_empresas(df_precos, df_empresas, ret_ibov):
    """
    DiD no nível de empresa individual:
      - Calcula CAR por empresa para cada (ano, janela)
      - Classifica cada empresa como Estatal ou Privada (coluna da planilha)
      - Testa: Δ = CAR_estatais − CAR_privadas (Welch t-test)
    """
    log.info("=" * 70)
    log.info("ETAPA 6b: DiD ESTATAIS vs. PRIVADAS (nível empresa)")
    log.info("=" * 70)

    ret_ativos = np.log(df_precos / df_precos.shift(1))

    t2info = {}
    for _, row in df_empresas.iterrows():
        t2info[row["TICKER_YF"]] = {
            "setor": row["SETOR_B3"], "estatal": row["ESTATAL"],
            "denom": row.get("DENOM_SOCIAL", "")
        }

    bdates = ret_ativos.dropna(how="all").index

    # DiD testa janelas principais incluindo 2ºT
    janelas_did = ["antecipacao_45", "reacao_curta_1t", "reacao_curta_2t", "reacao_media_2t"]

    n_est = sum(1 for v in t2info.values() if v["estatal"])
    n_priv = sum(1 for v in t2info.values() if not v["estatal"])
    log.info("  %d tickers (%d estatais, %d privadas)", len(t2info), n_est, n_priv)

    resultados_ind = []
    for ano in ANOS_ELEITORAIS:
        dt1 = DATAS_PRIMEIRO_TURNO[ano]
        est_ini, est_fim = janela_estimacao(bdates, dt1)
        jans = janelas_evento_1t(bdates, ano)
        jans.update(janelas_evento_2t(bdates, ano))
        jans_did = {k: v for k, v in jans.items() if k in janelas_did}

        n_proc = 0
        for ticker in ret_ativos.columns:
            if ticker not in t2info:
                continue
            info = t2info[ticker]
            m_est = (ret_ativos.index >= est_ini) & (ret_ativos.index <= est_fim)
            ret_tk = ret_ativos.loc[m_est, ticker].dropna()
            ret_m = ret_ibov.loc[m_est].dropna()
            if len(ret_tk) < 30:
                continue
            params = estimar_ols_simples(ret_tk, ret_m)
            if params is None:
                continue
            for nome_j, (ji, jf) in jans_did.items():
                m_j = (ret_ativos.index >= ji) & (ret_ativos.index <= jf)
                rt = ret_ativos.loc[m_j, ticker].dropna()
                rm = ret_ibov.loc[m_j].dropna()
                if len(rt) < 3:
                    continue
                ar = calcular_ar(rt, rm, params["alpha"], params["beta"])
                resultados_ind.append({
                    "ano": ano, "janela": nome_j, "ticker": ticker,
                    "setor": info["setor"], "estatal": info["estatal"],
                    "denom_social": str(info["denom"])[:50],
                    "car": ar.sum(), "n_dias": len(ar),
                    "alpha": params["alpha"], "beta": params["beta"],
                })
            n_proc += 1
        log.info("  Ano %d: %d empresas", ano, n_proc)

    df_ind = pd.DataFrame(resultados_ind)
    log.info("  → %d obs individuais", len(df_ind))

    resultados_agg = []
    if not df_ind.empty:
        for (ano, jan), grp in df_ind.groupby(["ano", "janela"]):
            est = grp[grp["estatal"]]["car"].values
            priv = grp[~grp["estatal"]]["car"].values
            if len(est) < 2 or len(priv) < 2:
                continue
            delta = np.mean(est) - np.mean(priv)
            t_did, p_did = stats.ttest_ind(est, priv, equal_var=False)
            resultados_agg.append({
                "ano": ano, "janela": jan,
                "car_medio_estatais": np.mean(est),
                "car_medio_privadas": np.mean(priv),
                "delta_did": delta, "t_did": t_did, "p_did": p_did,
                "sig5_did": p_did < 0.05 if not np.isnan(p_did) else False,
                "sig10_did": p_did < 0.10 if not np.isnan(p_did) else False,
                "n_estatais": len(est), "n_privadas": len(priv),
                "car_std_estatais": np.std(est, ddof=1),
                "car_std_privadas": np.std(priv, ddof=1),
            })

    df_agg = pd.DataFrame(resultados_agg)
    if not df_agg.empty:
        log.info("  → DiD: %d comp, %d sig@5%%", len(df_agg), df_agg["sig5_did"].sum())
    return df_ind, df_agg


# ============================================================================
# ETAPA 7 – VOLATILIDADE E SHARPE
# ============================================================================

def calcular_volatilidade_comparativa(df_ret):
    registros = []
    for setor in df_ret.columns:
        for ano in range(2002, 2023):
            r = df_ret.loc[df_ret.index.year == ano, setor].dropna()
            if len(r) < 20:
                continue
            registros.append({
                "setor": setor, "ano": ano,
                "tipo_ano": "Eleitoral" if ano in ANOS_ELEITORAIS else "Não-Eleitoral",
                "vol_anualizada": r.std() * np.sqrt(252)
            })
    return pd.DataFrame(registros)


def calcular_sharpe_comparativo(df_ret):
    """Sharpe Ratio anualizado. Usa Selic de todos os anos (não só eleitorais)."""
    log.info("Calculando Sharpe Ratio comparativo ...")
    registros = []
    for setor in df_ret.columns:
        for ano in range(2002, 2023):
            r = df_ret.loc[df_ret.index.year == ano, setor].dropna()
            if len(r) < 20:
                continue
            rf_diario = SELIC_DIARIA.get(ano, SELIC_DIARIA.get(2022, 0.0005))
            excess = r - rf_diario
            sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else np.nan
            t_exc, p_exc = stats.ttest_1samp(excess.dropna(), 0)
            registros.append({
                "setor": setor, "ano": ano,
                "tipo_ano": "Eleitoral" if ano in ANOS_ELEITORAIS else "Não-Eleitoral",
                "sharpe_anualizado": sharpe,
                "excesso_medio_diario": excess.mean(),
                "t_excesso": t_exc, "p_excesso": p_exc,
            })
    return pd.DataFrame(registros)


def adicionar_benchmarking(df_res):
    """Benchmarking: CAR vs. Selic e IPCA proporcionais."""
    df = df_res.copy()
    df["selic_proporcional"] = df["ano"].map(SELIC_ANUAL) * df["n_dias"] / 252
    df["ipca_proporcional"] = df["ano"].map(IPCA_ANUAL) * df["n_dias"] / 252
    df["car_exc_selic"] = df["car"] - df["selic_proporcional"]
    df["car_exc_ipca"] = df["car"] - df["ipca_proporcional"]
    return df


# ============================================================================
# ETAPA 8 – VISUALIZAÇÕES
# ============================================================================

def gerar_visualizacoes(df_res, df_vol, df_placebo, df_did, df_sharpe, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.figsize": (16, 9), "font.size": 11, "figure.dpi": 150})

    df = df_res[(df_res["cenario"] == "completo") & (df_res["ponderacao"] == "EW")]
    # Exclui espelhos dos heatmaps principais
    df_main = df[~df["janela"].str.startswith("espelho_")]

    # ---- HEATMAPS --------------------------------------------------------
    hm_map = {
        "antecipacao_45": "Antecipação (−45 a −1 d.u.)",
        "reacao_curta_1t": "Reação 1º Turno [−5,+5]",
        "reacao_curta_2t": "Reação 2º Turno [−5,+5]",
        "reacao_media_2t": "Reação 2º Turno [−10,+10]",
        "entre_turnos": "Entre Turnos [−5 do 1ºT a −1 do 2ºT]",
        "reacao_media_1t": "Reação [−10,+10] (Robustez)",
        "reacao_ampla_1t": "Reação [−20,+20] (Robustez)",
        "antecipacao_60": "Antecipação (−60 d.u.) (Robustez)",
    }
    for jkey, titulo in hm_map.items():
        dj = df_main[df_main["janela"] == jkey]
        if dj.empty:
            continue
        pivot = dj.pivot_table(index="setor", columns="ano", values="car", aggfunc="mean")
        pivot_sig = dj.pivot_table(index="setor", columns="ano", values="sig5_simples", aggfunc="any")
        if pivot.empty:
            continue
        annot = pivot.copy().astype(object)
        for r in annot.index:
            for c in annot.columns:
                v = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
                s = pivot_sig.loc[r, c] if r in pivot_sig.index and c in pivot_sig.columns else False
                annot.loc[r, c] = f"{v:.3f}{'*' if s else ''}" if pd.notna(v) else ""
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=0,
                    linewidths=.5, linecolor="gray",
                    cbar_kws={"label": "CAR (EW)"}, ax=ax)
        ax.set_title(f"CAR por Setor e Ano – {titulo}\n(* p<0.05)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Ano Eleitoral")
        ax.set_ylabel("Setor B3")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{jkey}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ heatmap_%s.png", jkey)

    # ---- EVOLUÇÃO TEMPORAL -----------------------------------------------
    df_ant = df_main[df_main["janela"] == "antecipacao_45"]
    if not df_ant.empty:
        mag = df_ant.groupby("setor")["car"].apply(lambda x: abs(x).mean())
        top5 = mag.nlargest(5).index.tolist()
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for idx, (j, nm) in enumerate([("antecipacao_45", "Antecipação −45"),
                                        ("reacao_curta_1t", "Reação [−5,+5]")]):
            ax = axes[idx]
            for s in top5:
                d = df_main[(df_main["setor"] == s) & (df_main["janela"] == j)].sort_values("ano")
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
        plt.savefig(os.path.join(output_dir, "evolucao_temporal_cars.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ evolucao_temporal_cars.png")

    # ---- VOLATILIDADE COMPARATIVA ----------------------------------------
    if not df_vol.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        va = df_vol.groupby(["setor", "tipo_ano"])["vol_anualizada"].mean().reset_index()
        pv = va.pivot(index="setor", columns="tipo_ano", values="vol_anualizada")
        pv.plot(kind="barh", ax=ax, alpha=.85, edgecolor="k")
        ax.set_title("Volatilidade Anualizada: Eleitoral vs. Não-Eleitoral (EW)", fontweight="bold")
        ax.set_xlabel("σ·√252")
        ax.set_ylabel("")
        ax.legend(title="Tipo de Ano")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volatilidade_comparativa.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ volatilidade_comparativa.png")

    # ---- PLACEBO vs REAL -------------------------------------------------
    if not df_placebo.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (jr, jp, nm) in enumerate([
            ("antecipacao_45", "placebo_antecip", "Antecipação"),
            ("reacao_curta_1t", "placebo_reacao", "Reação [−5,+5]")
        ]):
            ax = axes[idx]
            cp = df_placebo[df_placebo["janela"] == jp]["car"].dropna()
            cr = df_main[df_main["janela"] == jr]["car"].dropna()
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

    # ---- DiD CHART (Estatais vs. Privadas) --------------------------------
    if not df_did.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        for jan in df_did["janela"].unique():
            d = df_did[df_did["janela"] == jan].sort_values("ano")
            if not d.empty:
                ax.plot(d["ano"], d["delta_did"], marker="s", lw=2, ms=8, label=jan)
                for _, r in d.iterrows():
                    if r["sig5_did"]:
                        ax.annotate("*", (r["ano"], r["delta_did"]), fontsize=14,
                                    ha="center", va="bottom", color="red")
        ax.axhline(0, color="k", ls="--", alpha=.3)
        ax.set_title("Difference-in-Differences: Estatais − Privadas (* p<0.05)\n"
                     "(Nível de Empresa Individual · Classificação pela Planilha)",
                     fontweight="bold")
        ax.set_xlabel("Ano Eleitoral")
        ax.set_ylabel("Δ CAR (Estatais − Privadas)")
        ax.legend()
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "did_estatais_vs_privadas.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("  ✓ did_estatais_vs_privadas.png")

    # ---- ESPELHOS: Eleitoral vs T-1 --------------------------------------
    jans_espelho = [j for j in df["janela"].unique() if j.startswith("espelho_")]
    if jans_espelho:
        pares = {}
        for esp in jans_espelho:
            orig = esp.replace("espelho_", "")
            if orig in df["janela"].unique():
                pares[orig] = esp
        if pares:
            n_pares = len(pares)
            fig, axes = plt.subplots(1, min(n_pares, 4), figsize=(6 * min(n_pares, 4), 6))
            if n_pares == 1:
                axes = [axes]
            for idx, (orig, esp) in enumerate(list(pares.items())[:4]):
                ax = axes[idx]
                car_orig = df[df["janela"] == orig].groupby("ano")["car"].mean()
                car_esp = df[df["janela"] == esp].groupby("ano")["car"].mean()
                ax.bar(car_orig.index - 0.15, car_orig.values, width=0.3, label="Eleitoral", color="steelblue")
                ax.bar(car_esp.index + 0.15, car_esp.values, width=0.3, label="Espelho T-1", color="gray", alpha=0.7)
                ax.axhline(0, color="k", ls="--", alpha=.3)
                ax.set_title(orig, fontweight="bold")
                ax.set_xlabel("Ano")
                ax.set_ylabel("CAR Médio")
                ax.legend(fontsize=8)
            plt.suptitle("Eleições vs. Espelhos (T-1)", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "espelhos_vs_eleicoes.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()
            log.info("  ✓ espelhos_vs_eleicoes.png")


# ============================================================================
# ETAPA 9 – EXPORTAÇÃO
# ============================================================================

def exportar_resultados(df_res, df_ff3, df_vol, df_sobrev, df_comp, df_diag,
                        df_placebo, df_did_ind, df_did_agg, df_sharpe, df_desc, output_dir):
    log.info("=" * 70)
    log.info("ETAPA 9: EXPORTAÇÃO")
    log.info("=" * 70)

    # CSVs
    df_res.to_csv(os.path.join(output_dir, "resultados_consolidados_v3.csv"),
                  index=False, encoding="utf-8-sig")
    log.info("  ✓ resultados_consolidados_v3.csv")

    df_sobrev.to_csv(os.path.join(output_dir, "tabela_sobrevivencia.csv"),
                     index=False, encoding="utf-8-sig")
    log.info("  ✓ tabela_sobrevivencia.csv")

    df_diag.to_csv(os.path.join(output_dir, "diagnostico_download.csv"),
                   index=False, encoding="utf-8-sig")
    log.info("  ✓ diagnostico_download.csv")

    if not df_desc.empty:
        df_desc.to_csv(os.path.join(output_dir, "estatisticas_descritivas.csv"),
                       index=False, encoding="utf-8-sig")
        log.info("  ✓ estatisticas_descritivas.csv")

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

    # Excel consolidado
    xlsx = os.path.join(output_dir, "analise_ciclos_eleitorais_v3.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df_res.to_excel(writer, sheet_name="Resultados", index=False)

        # Heatmaps em abas do Excel
        df_ew = df_res[(df_res["cenario"] == "completo") & (df_res["ponderacao"] == "EW")]
        df_ew_main = df_ew[~df_ew["janela"].str.startswith("espelho_")]

        for jan in ["antecipacao_45", "reacao_curta_1t", "reacao_curta_2t",
                    "entre_turnos", "reacao_media_2t"]:
            dj = df_ew_main[df_ew_main["janela"] == jan]
            if dj.empty:
                continue
            pc = dj.pivot_table(index="setor", columns="ano", values="car")
            pp = dj.pivot_table(index="setor", columns="ano", values="p_simples")
            merged = pc.copy().astype(object)
            for r in merged.index:
                for c in merged.columns:
                    v = pc.loc[r, c] if r in pc.index and c in pc.columns else np.nan
                    p = pp.loc[r, c] if r in pp.index and c in pp.columns else np.nan
                    if pd.notna(v):
                        star = ("***" if pd.notna(p) and p < .01
                                else ("**" if pd.notna(p) and p < .05
                                      else ("*" if pd.notna(p) and p < .10 else "")))
                        merged.loc[r, c] = f"{v:.4f}{star}"
            merged.to_excel(writer, sheet_name=f"CAR_{jan[:15]}")

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

        # Sumário
        sumario = []
        for jan in sorted(df_ew["janela"].unique()):
            dj = df_ew[df_ew["janela"]==jan]
            if dj.empty:
                continue
            sumario.append({
                "Janela": jan, "CAR Médio": f"{dj['car'].mean():.4f}",
                "CAR Mediano": f"{dj['car'].median():.4f}",
                "%Sig5 simples": f"{dj['sig5_simples'].mean()*100:.0f}%",
                "%Sig5 Silva": f"{dj['sig5_silva'].mean()*100:.0f}%",
                "%Sig5 BMP": f"{dj['sig5_bmp'].mean()*100:.0f}%",
                "Setor |CAR| max": dj.loc[dj["car"].abs().idxmax(),"setor"] if len(dj)>0 else "",
                "N": len(dj),
            })
        pd.DataFrame(sumario).to_excel(writer, sheet_name="Sumário", index=False)

    log.info("  ✓ %s", xlsx)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("\n" + "█"*72)
    print("█  CICLOS ELEITORAIS NA B3 (2002–2022) — v3 ACADÊMICA              █")
    print("█  CAPM + FF3 · HAC · BMP · Placebo · DiD · Sharpe                 █")
    print("█"*72 + "\n")

    if not os.path.exists(ARQUIVO_ENTRADA):
        log.error("Arquivo não encontrado: %s", ARQUIVO_ENTRADA)
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Ingestão -----------------------------------------------------
    df_emp = carregar_lista_empresas(ARQUIVO_ENTRADA)

    # ---- 2. Download preços + volumes ------------------------------------
    tickers_yf = df_emp["TICKER_YF"].unique().tolist()
    df_precos, df_volumes, df_diag = baixar_precos_yfinance(tickers_yf)
    ibov = baixar_ibovespa()

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
    df_factors = baixar_fatores_nefin()

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
    dfc = df_resultados[(df_resultados["cenario"]=="completo")&(df_resultados["ponderacao"]=="EW")]

    print("\n" + "="*72)
    print("RESUMO FINAL")
    print("="*72)
    print(f"Tempo: {elapsed:.0f}s")
    print(f"Obs EW completo: {len(dfc)} | VW: {len(df_res_vw)} | FF3: {len(df_ff3)} | Placebo: {len(df_placebo)}")
    print(f"Setores: {dfc['setor'].nunique()} | Anos: {sorted(dfc['ano'].unique())}")

    print("\nCAR Médio por Janela (EW completo):")
    for j in sorted(dfc["janela"].unique()):
        d = dfc[dfc["janela"]==j]
        print(f"  {j:25s}  CAR={d['car'].mean():+.4f}  "
              f"sig_s={d['sig5_simples'].mean()*100:.0f}%  "
              f"silva={d['sig5_silva'].mean()*100:.0f}%  "
              f"bmp={d['sig5_bmp'].mean()*100:.0f}%")

    nok = (df_diag["status"]=="ok").sum()
    print(f"\nCobertura: {nok}/{len(df_diag)} tickers ({100*nok/len(df_diag):.0f}%)")
    if not df_ff3.empty:
        print(f"FF3 obs: {len(df_ff3)} | sig5: {df_ff3['sig5_ff3'].mean()*100:.0f}%")
    if not df_did_agg.empty:
        did_sig = df_did_agg["sig5_did"].sum()
        print(f"DiD (Estatais vs Privadas): {len(df_did_agg)} comparações, "
              f"{did_sig} significativas a 5%")
    if not df_did_ind.empty:
        n_est = df_did_ind[df_did_ind["estatal"]]["ticker"].nunique()
        n_priv = df_did_ind[~df_did_ind["estatal"]]["ticker"].nunique()
        print(f"  → {n_est} empresas estatais, {n_priv} privadas analisadas")

    print(f"\nArquivos: {os.path.abspath(OUTPUT_DIR)}/")
    print("="*72)
    print("✓ ANÁLISE v3 FINALIZADA!")
    print("="*72)


if __name__ == "__main__":
    main()
