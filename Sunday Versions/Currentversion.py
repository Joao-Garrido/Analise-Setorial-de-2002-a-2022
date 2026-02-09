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
    "Saúde", "Utilidade Pública",]

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


#Corrigir alguns apenas

TICKER_MAPPING = {
    "VVAR3": "BHIA3", "BTOW3": "AMER3", "LAME4": "LAME3", "PCAR4": "PCAR3",
    "KROT3": "COGN3", "ESTC3": "YDUQ3", "RAIL3": "RUMO3",
    "BVMF3": "B3SA3", "CTIP3": "B3SA3",
    "BRIN3": "BRML3", "BRML3": "ALOS3", "SMLE3": "SMFT3",
    "LINX3": "STNE3", "VIVT4": "VIVT3", "TIMP3": "TIMS3",
    "QGEP3": "BRAV3", "GNDI3": "HAPV3", "FIBR3": "SUZB3",
}

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

    # Log das estatais para conferência
    for _, r in df[df["ESTATAL"]].iterrows():
        log.info("    [ESTATAL] %s — %s (%s)",
                 r["TICKER_ORIGINAL"], str(r["DENOM_SOCIAL"])[:45], r["SETOR_B3"])

    return df

import os
import time
from alpha_vantage.timeseries import TimeSeries

# Forma segura: defina como variável de ambiente (recomendado)
# No terminal: export ALPHA_VANTAGE_KEY=LB3KHS0AN1R2E36B (Linux/Mac) ou set no Windows
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

# Fallback temporário só para teste local (remova em produção!)
if not ALPHA_VANTAGE_KEY:
    ALPHA_VANTAGE_KEY = 'SSU8PQU94ONCSLAF'  # Apague isso depois de configurar env
    print("AVISO: Usando key hardcoded – configure como env var para segurança!")

def baixar_precos_yfinance(tickers: list, start="2001-01-01", end="2023-12-31"):
    """Baixa preços ajustados e volumes via yfinance (primário) + Alpha Vantage (fallback)."""
    import yfinance as yf

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

time.sleep(5)

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
    
    # URL definida globalmente ou localmente
    url_csv = "https://nefin.com.br/resources/risk_factors/nefin_factors.csv"
    
    try:
        # Tenta ler o CSV direto da URL
        df = pd.read_csv(url_csv)

        # 1. Tratamento de Data (O CSV novo tem a coluna 'Date' pronta)
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
            df = df.set_index("date")
        else:
            # Fallback caso o formato mude para year/month/day
            if {"year", "month", "day"}.issubset(df.columns):
                 df["date"] = pd.to_datetime(df[["year", "month", "day"]])
                 df = df.set_index("date")
            else:
                log.warning("  Formato de data desconhecido no CSV do NEFIN.")
                return None

        # 2. Renomear colunas para o padrão do script (Mkt_Rf, SMB, HML, Rf)
        # O CSV vem como: Rm_minus_Rf, SMB, HML, Risk_Free
        rename_map = {
            "Rm_minus_Rf": "Mkt_Rf", 
            "Risk_Free": "Rf"
        }
        df = df.rename(columns=rename_map)

        # 3. Filtrar apenas as colunas necessárias
        cols_necessarias = ["Mkt_Rf", "SMB", "HML", "Rf"]
        
        # Verifica se todas existem
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

def aplicar_filtro_existencia(df_precos, df_empresas):
    """
    Invalida (NaN) precos anteriores a data de registro (DT_REG).
    Assume que DT_REG existe e que nao ha cancelamentos.
    """
    log.info("Aplicando filtro de existencia (apenas DT_REG)...")

    # Cria copia para nao alterar o original
    df = df_precos.copy()

    # Remove fuso horario para evitar erro de comparacao com o Excel
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Dicionario: Ticker -> Data Inicio
    lookup = {}
    col_inicio = "DT_REG"

    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        # Converte direto
        reg = pd.to_datetime(row[col_inicio], errors='coerce')

        # Se houver duplicata de ticker, preserva a data mais antiga (conservador)
        if tk in lookup:
            old_reg = lookup[tk]
            if pd.notna(reg) and (pd.isna(old_reg) or reg < old_reg):
                lookup[tk] = reg
        else:
            lookup[tk] = reg

    total_cortes = 0

    # Aplica o filtro coluna por coluna
    for col in df.columns:
        if col in lookup:
            dt_inicio = lookup[col]

            if pd.notna(dt_inicio):
                # Corta tudo que vier antes da data de registro
                mask = df.index < dt_inicio
                if mask.any():
                    qtd = df.loc[mask, col].notna().sum()
                    if qtd > 0:
                        total_cortes += qtd
                        df.loc[mask, col] = np.nan

    log.info("  -> Filtro aplicado. Observacoes removidas (pre-inicio): %d", total_cortes)
    
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
    
    n_pregoes = mask_ano.sum()  # total de pregões no ano
    min_obs = int(n_pregoes * min_pct)
    
    # Conta pregões com dado válido (não-NaN) por ticker
    obs_por_ticker = ret_ano.notna().sum()
    
    aprovados = obs_por_ticker[obs_por_ticker >= min_obs].index.tolist()
    return aprovados


def construir_indices_setoriais(df_precos, df_volumes, df_empresas):
    """
    Constrói índices setoriais EW e VW com filtro de liquidez.
    
    Etapas internas:
      1. Retornos log de todos os ativos
      2. Para cada (setor, ano): aplica filtro de liquidez
      3. EW = média simples dos retornos dos tickers líquidos
      4. VW = média ponderada por volume financeiro (proxy market cap)
      5. Registra composição real (N após filtro)
    
    Retorna:
        (df_ret_ew, df_ret_vw, df_composicao)
    """
    log.info("=" * 70)
    log.info("ETAPA 2: ÍNDICES SETORIAIS (EW + VW) COM FILTRO DE LIQUIDEZ")
    log.info("=" * 70)

    # Retornos log
    ret = np.log(df_precos / df_precos.shift(1))
    
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
        # Todos os tickers do setor que existem no DataFrame
        cols_setor = [c for c in ret.columns if t2s.get(c) == setor]
        if not cols_setor:
            log.warning("  Sem tickers para '%s'", setor)
            continue

        ew_parts = []  # pedaços anuais do índice EW
        vw_parts = []  # pedaços anuais do índice VW

        for ano in range(2001, 2024):
            mask_ano = ret.index.year == ano
            if mask_ano.sum() == 0:
                continue

            # Filtro de liquidez: quais tickers do setor têm ≥80% pregões neste ano
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
                # Poucos tickers → NaN para este setor/ano
                continue

            ret_ano = ret.loc[mask_ano, cols_ano]
            
            # EW: média simples
            ew_parts.append(ret_ano.mean(axis=1))

            # VW: ponderado por volume financeiro médio 20d
            vf_ano = vol_fin.loc[mask_ano, cols_ano].rolling(20, min_periods=5).mean()
            vf_sum = vf_ano.sum(axis=1)
            # Evita divisão por zero
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
    Sem DT_CANCEL, n_esperado = empresas com DT_REG <= ano.
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

            # N esperado: empresas registradas até esse ano (sem DT_CANCEL)
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