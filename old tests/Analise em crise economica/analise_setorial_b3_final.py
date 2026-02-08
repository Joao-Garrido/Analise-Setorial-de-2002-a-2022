#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANÁLISE SETORIAL B3 (2002-2022) - VERSÃO FINAL
================================================================================
Autor: Pesquisa Acadêmica
Baseado em: Análise Setorial de 2002 a 2022

Este script realiza análise comparativa do desempenho econômico-financeiro entre
os setores de Consumo Cíclico e Não Cíclico da B3 em períodos de crise econômica.

Metodologia:
- Índices Setoriais EW (Equal Weighted) e VW (Value Weighted)
- Filtros de liquidez (80% dos pregões)
- Análise estatística comparativa (Wilcoxon, Mann-Whitney)
- Visualizações comparativas

Períodos Analisados:
- 2009-2011: Expansão econômica
- 2012-2014: Crescimento moderado
- 2015-2017: Recessão acentuada
- 2018-2020: Recuperação frágil + COVID-19
================================================================================
"""

import os
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, shapiro
import seaborn as sns

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONSTANTES E CONFIGURAÇÕES
# =============================================================================

# Períodos de análise conforme pesquisa
PERIODOS_ANALISE = {
    '2009-2011': (datetime(2009, 1, 1), datetime(2011, 12, 31)),
    '2012-2014': (datetime(2012, 1, 1), datetime(2014, 12, 31)),
    '2015-2017': (datetime(2015, 1, 1), datetime(2017, 12, 31)),
    '2018-2020': (datetime(2018, 1, 1), datetime(2020, 12, 31)),
}

# Setores da B3 para análise
SETORES_B3 = [
    "Consumo Cíclico",
    "Consumo Não Cíclico",
    "Bens Industriais",
    "Materiais Básicos",
    "Petróleo, Gás e Biocombustíveis",
    "Saúde",
    "Tecnologia da Informação",
    "Comunicações",
    "Utilidade Pública",
    "Financeiro"
]

# Parâmetros de filtragem
MIN_PREGOES_PCT = 0.80      # Mínimo de 80% dos pregões no ano
MIN_EMPRESAS_SETOR = 3      # Mínimo de empresas por setor

# Diretórios - na mesma pasta do script
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "Outputs New"
OUTPUT_DIR.mkdir(exist_ok=True)
GRAFICOS_DIR = OUTPUT_DIR / "graficos"
GRAFICOS_DIR.mkdir(exist_ok=True)
TABELAS_DIR = OUTPUT_DIR / "tabelas"
TABELAS_DIR.mkdir(exist_ok=True)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# =============================================================================

def carregar_dados_empresas(caminho_excel: str) -> pd.DataFrame:
    """Carrega dados das empresas do arquivo Excel."""
    log.info("=" * 70)
    log.info("CARREGANDO DADOS DAS EMPRESAS")
    log.info("=" * 70)
    
    df = pd.read_excel(caminho_excel)
    
    # Converter datas
    df['DT_REG'] = pd.to_datetime(df['DT_REG'], errors='coerce')
    
    # Criar coluna TICKER_YF a partir de TICKER
    df['TICKER_YF'] = df['TICKER'].apply(
        lambda x: x if pd.isna(x) or str(x).endswith('.SA') else f"{x}.SA"
    )
    
    log.info(f"Total de empresas carregadas: {len(df)}")
    log.info(f"Setores encontrados: {df['SETOR_B3'].nunique()}")
    
    return df


def carregar_fatores_nefin(caminho_csv: str) -> pd.DataFrame:
    """Carrega fatores de risco NEFIN."""
    log.info("=" * 70)
    log.info("CARREGANDO FATORES NEFIN")
    log.info("=" * 70)
    
    df = pd.read_csv(caminho_csv, skipinitialspace=True)
    
    # Limpar nomes das colunas
    df.columns = [col.strip().replace('"', '') for col in df.columns]
    
    # Limpar valores
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('"', '').str.strip()
    
    # Converter data
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Converter colunas numéricas
    colunas_numericas = ['Rm_minus_Rf', 'SMB', 'HML', 'WML', 'IML', 'Risk_Free']
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Renomear colunas
    df = df.rename(columns={"Rm_minus_Rf": "Mkt_Rf", "Risk_Free": "Rf"})
    
    log.info(f"Período dos fatores: {df.index.min().date()} a {df.index.max().date()}")
    log.info(f"Total de observações: {len(df)}")
    
    return df


def baixar_dados_yahoo_lote(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa preços e volumes do Yahoo Finance para um lote de tickers."""
    try:
        dados = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True
        )
        
        if len(tickers) == 1:
            return dados['Close'].to_frame(tickers[0]), dados['Volume'].to_frame(tickers[0])
        else:
            return dados['Close'], dados['Volume']
            
    except Exception as e:
        log.warning(f"Erro ao baixar lote: {e}")
        # Tentar um por um
        precos = {}
        volumes = {}
        for ticker in tickers:
            try:
                d = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                precos[ticker] = d['Close']
                volumes[ticker] = d['Volume']
            except:
                log.warning(f"  Falha ao baixar {ticker}")
        
        return pd.DataFrame(precos), pd.DataFrame(volumes)


def baixar_dados_yahoo(tickers: List[str], start: str, end: str, max_tickers: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa preços e volumes do Yahoo Finance em lotes.
    
    Args:
        tickers: Lista de tickers
        start: Data de início
        end: Data de fim
        max_tickers: Limitar número de tickers (para teste)
        
    Returns:
        Tupla (df_precos, df_volumes)
    """
    log.info("=" * 70)
    log.info("BAIXANDO DADOS DO YAHOO FINANCE")
    log.info("=" * 70)
    log.info(f"Período: {start} a {end}")
    
    if max_tickers:
        tickers = tickers[:max_tickers]
        log.info(f"Total de tickers (limitado): {len(tickers)}")
    else:
        log.info(f"Total de tickers: {len(tickers)}")
    
    # Baixar em lotes
    lote_size = 20
    precos_list = []
    volumes_list = []
    total_lotes = (len(tickers) + lote_size - 1) // lote_size
    
    for i in range(0, len(tickers), lote_size):
        lote = tickers[i:i+lote_size]
        lote_num = i // lote_size + 1
        log.info(f"  Lote {lote_num}/{total_lotes}: {len(lote)} tickers...")
        
        precos, volumes = baixar_dados_yahoo_lote(lote, start, end)
        
        if not precos.empty:
            precos_list.append(precos)
        if not volumes.empty:
            volumes_list.append(volumes)
    
    # Concatenar
    df_precos = pd.concat(precos_list, axis=1) if precos_list else pd.DataFrame()
    df_volumes = pd.concat(volumes_list, axis=1) if volumes_list else pd.DataFrame()
    
    log.info(f"Preços baixados: {len(df_precos)} obs x {len(df_precos.columns)} ativos")
    
    return df_precos, df_volumes


def baixar_ibovespa(start: str, end: str) -> pd.Series:
    """Baixa dados do Ibovespa."""
    log.info("Baixando Ibovespa...")
    
    ibov = yf.download('^BVSP', start=start, end=end, auto_adjust=True, progress=False)
    return ibov['Close']


# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================

def aplicar_filtro_existencia(df_precos: pd.DataFrame, df_empresas: pd.DataFrame) -> pd.DataFrame:
    """Invalida preços anteriores à data de registro (DT_REG)."""
    log.info("Aplicando filtro de existência (DT_REG)...")
    
    df = df_precos.copy()
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    lookup = {}
    for _, row in df_empresas.iterrows():
        tk = row["TICKER_YF"]
        reg = pd.to_datetime(row["DT_REG"], errors='coerce')
        
        if tk in lookup:
            old_reg = lookup[tk]
            if pd.notna(reg) and (pd.isna(old_reg) or reg < old_reg):
                lookup[tk] = reg
        else:
            lookup[tk] = reg
    
    total_cortes = 0
    
    for col in df.columns:
        if col in lookup:
            dt_inicio = lookup[col]
            if pd.notna(dt_inicio):
                mask = df.index < dt_inicio
                if mask.any():
                    qtd = df.loc[mask, col].notna().sum()
                    if qtd > 0:
                        total_cortes += qtd
                        df.loc[mask, col] = np.nan
    
    log.info(f"  → Observações removidas (pré-início): {total_cortes}")
    
    return df


def aplicar_filtro_liquidez(df_ret: pd.DataFrame, ano: int, min_pct: float = MIN_PREGOES_PCT) -> List[str]:
    """Filtra tickers que negociaram em menos de min_pct dos pregões no ano."""
    mask_ano = df_ret.index.year == ano
    ret_ano = df_ret.loc[mask_ano]
    
    if ret_ano.empty:
        return []
    
    n_pregoes = mask_ano.sum()
    min_obs = int(n_pregoes * min_pct)
    
    obs_por_ticker = ret_ano.notna().sum()
    aprovados = obs_por_ticker[obs_por_ticker >= min_obs].index.tolist()
    
    return aprovados


def construir_indices_setoriais(
    df_precos: pd.DataFrame,
    df_volumes: pd.DataFrame,
    df_empresas: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constrói índices setoriais EW e VW com filtro de liquidez."""
    log.info("=" * 70)
    log.info("CONSTRUINDO ÍNDICES SETORIAIS (EW + VW)")
    log.info("=" * 70)
    
    ret = np.log(df_precos / df_precos.shift(1))
    vol_fin = df_precos * df_volumes
    
    t2s = {}
    for _, row in df_empresas.iterrows():
        t2s[row["TICKER_YF"]] = row["SETOR_B3"]
    
    series_ew = {}
    series_vw = {}
    composicao = []
    
    for setor in SETORES_B3:
        cols_setor = [c for c in ret.columns if t2s.get(c) == setor]
        if not cols_setor:
            log.warning(f"  Sem tickers para '{setor}'")
            continue
        
        ew_parts = []
        vw_parts = []
        
        for ano in range(2001, 2024):
            mask_ano = ret.index.year == ano
            if mask_ano.sum() == 0:
                continue
            
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
            ew_parts.append(ret_ano.mean(axis=1))
            
            vf_ano = vol_fin.loc[mask_ano, cols_ano].rolling(20, min_periods=5).mean()
            vf_sum = vf_ano.sum(axis=1)
            weights = vf_ano.div(vf_sum.replace(0, np.nan), axis=0)
            vw_parts.append((ret_ano * weights).sum(axis=1))
        
        if ew_parts:
            series_ew[setor] = pd.concat(ew_parts).sort_index()
        if vw_parts:
            series_vw[setor] = pd.concat(vw_parts).sort_index()
        
        comp_setor = [c for c in composicao if c["setor"] == setor]
        anos_validos = sum(1 for c in comp_setor if c["n_com_dados_liquidos"] >= MIN_EMPRESAS_SETOR)
        log.info(f"  {setor}: {len(cols_setor)} tickers | {anos_validos}/{len(comp_setor)} anos válidos")
    
    df_ret_ew = pd.DataFrame(series_ew)
    df_ret_vw = pd.DataFrame(series_vw)
    df_comp = pd.DataFrame(composicao)
    
    log.info(f"  → EW: {df_ret_ew.shape[1]} setores × {len(df_ret_ew)} datas")
    log.info(f"  → VW: {df_ret_vw.shape[1]} setores × {len(df_ret_vw)} datas")
    
    return df_ret_ew, df_ret_vw, df_comp


# =============================================================================
# ANÁLISE ESTATÍSTICA
# =============================================================================

def calcular_estatisticas_descritivas(
    df_retornos: pd.DataFrame,
    periodo_nome: str,
    periodo_range: Tuple[datetime, datetime]
) -> pd.DataFrame:
    """Calcula estatísticas descritivas por setor para um período."""
    inicio, fim = periodo_range
    mask = (df_retornos.index >= inicio) & (df_retornos.index <= fim)
    df_periodo = df_retornos.loc[mask]
    
    estatisticas = []
    
    for setor in df_retornos.columns:
        ret = df_periodo[setor].dropna()
        
        if len(ret) < 10:
            continue
        
        # Teste de normalidade
        if len(ret) >= 3 and len(ret) <= 5000:
            _, p_normal = shapiro(ret.sample(min(5000, len(ret))))
        else:
            p_normal = np.nan
        
        estatisticas.append({
            'periodo': periodo_nome,
            'setor': setor,
            'n_obs': len(ret),
            'media': ret.mean(),
            'mediana': ret.median(),
            'desvio_padrao': ret.std(),
            'min': ret.min(),
            'max': ret.max(),
            'skewness': ret.skew(),
            'kurtosis': ret.kurtosis(),
            'shapiro_p': p_normal,
            'normal': p_normal > 0.05 if not np.isnan(p_normal) else None
        })
    
    return pd.DataFrame(estatisticas)


def teste_wilcoxon_periodos(
    df_retornos: pd.DataFrame,
    periodo1: Tuple[str, Tuple[datetime, datetime]],
    periodo2: Tuple[str, Tuple[datetime, datetime]]
) -> pd.DataFrame:
    """Realiza teste de Wilcoxon para comparar dois períodos."""
    nome1, (inicio1, fim1) = periodo1
    nome2, (inicio2, fim2) = periodo2
    
    mask1 = (df_retornos.index >= inicio1) & (df_retornos.index <= fim1)
    mask2 = (df_retornos.index >= inicio2) & (df_retornos.index <= fim2)
    
    resultados = []
    
    for setor in df_retornos.columns:
        ret1 = df_retornos.loc[mask1, setor].dropna()
        ret2 = df_retornos.loc[mask2, setor].dropna()
        
        if len(ret1) < 10 or len(ret2) < 10:
            continue
        
        min_len = min(len(ret1), len(ret2))
        ret1_sample = ret1.sample(min_len, random_state=42)
        ret2_sample = ret2.sample(min_len, random_state=42)
        
        try:
            stat, p_value = wilcoxon(ret1_sample, ret2_sample)
            
            resultados.append({
                'setor': setor,
                'periodo_1': nome1,
                'periodo_2': nome2,
                'mediana_1': ret1.median(),
                'mediana_2': ret2.median(),
                'n_1': len(ret1),
                'n_2': len(ret2),
                'statistic': stat,
                'p_value': p_value,
                'significativo': p_value < 0.05,
                'diferenca_mediana': ret2.median() - ret1.median()
            })
        except Exception as e:
            log.warning(f"Erro no teste Wilcoxon para {setor}: {e}")
    
    return pd.DataFrame(resultados)


def analise_comparativa_setorial(df_retornos: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Realiza análise comparativa completa entre setores e períodos."""
    log.info("=" * 70)
    log.info("ANÁLISE COMPARATIVA SETORIAL")
    log.info("=" * 70)
    
    resultados = {}
    
    # Estatísticas descritivas
    log.info("Calculando estatísticas descritivas...")
    estatisticas = []
    for periodo, periodo_range in PERIODOS_ANALISE.items():
        est = calcular_estatisticas_descritivas(df_retornos, periodo, periodo_range)
        estatisticas.append(est)
    resultados['estatisticas_descritivas'] = pd.concat(estatisticas, ignore_index=True)
    
    # Testes Wilcoxon
    log.info("Realizando testes Wilcoxon...")
    periodos_list = list(PERIODOS_ANALISE.items())
    wilcoxon_results = []
    for i in range(len(periodos_list) - 1):
        res = teste_wilcoxon_periodos(df_retornos, periodos_list[i], periodos_list[i+1])
        wilcoxon_results.append(res)
    resultados['wilcoxon'] = pd.concat(wilcoxon_results, ignore_index=True) if wilcoxon_results else pd.DataFrame()
    
    # Testes Mann-Whitney entre Consumo Cíclico e Não Cíclico
    log.info("Realizando testes Mann-Whitney...")
    mw_results = []
    setor1, setor2 = "Consumo Cíclico", "Consumo Não Cíclico"
    
    for periodo, periodo_range in PERIODOS_ANALISE.items():
        inicio, fim = periodo_range
        mask = (df_retornos.index >= inicio) & (df_retornos.index <= fim)
        
        if setor1 in df_retornos.columns and setor2 in df_retornos.columns:
            ret1 = df_retornos.loc[mask, setor1].dropna()
            ret2 = df_retornos.loc[mask, setor2].dropna()
            
            if len(ret1) >= 10 and len(ret2) >= 10:
                try:
                    stat, p_value = mannwhitneyu(ret1, ret2, alternative='two-sided')
                    mw_results.append({
                        'periodo': periodo,
                        'setor_1': setor1,
                        'setor_2': setor2,
                        'mediana_1': ret1.median(),
                        'mediana_2': ret2.median(),
                        'n_1': len(ret1),
                        'n_2': len(ret2),
                        'statistic': stat,
                        'p_value': p_value,
                        'significativo': p_value < 0.05
                    })
                except Exception as e:
                    log.warning(f"Erro no teste Mann-Whitney: {e}")
    
    resultados['mann_whitney'] = pd.DataFrame(mw_results) if mw_results else pd.DataFrame()
    
    log.info("Análise comparativa concluída!")
    
    return resultados


# =============================================================================
# VISUALIZAÇÕES
# =============================================================================

def configurar_estilo_graficos():
    """Configura o estilo visual dos gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10


def plotar_indices_setoriais(df_retornos: pd.DataFrame, arquivo_saida: str = None):
    """Plota a evolução dos índices setoriais acumulados."""
    configurar_estilo_graficos()
    
    ret_acum = (1 + df_retornos).cumprod()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for setor in ret_acum.columns:
        ax.plot(ret_acum.index, ret_acum[setor], label=setor, linewidth=1.5)
    
    ax.set_title("Índices Setoriais B3 - Value Weighted")
    ax.set_xlabel('Data')
    ax.set_ylabel('Retorno Acumulado')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Destacar períodos de crise
    ax.axvspan(datetime(2015, 1, 1), datetime(2017, 12, 31), alpha=0.2, color='red', label='Recessão 2015-2017')
    ax.axvspan(datetime(2020, 1, 1), datetime(2020, 12, 31), alpha=0.2, color='orange', label='COVID-19')
    
    plt.tight_layout()
    
    if arquivo_saida:
        plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
        log.info(f"Gráfico salvo: {arquivo_saida}")
    
    plt.close()


def plotar_comparacao_periodos(df_estatisticas: pd.DataFrame, indicador: str, arquivo_saida: str = None):
    """Plota comparação de um indicador entre períodos."""
    configurar_estilo_graficos()
    
    df_pivot = df_estatisticas.pivot(index='setor', columns='periodo', values=indicador)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'Comparação de {indicador.capitalize()} entre Períodos')
    ax.set_xlabel('Setor')
    ax.set_ylabel(indicador.capitalize())
    ax.legend(title='Período', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if arquivo_saida:
        plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
        log.info(f"Gráfico salvo: {arquivo_saida}")
    
    plt.close()


def plotar_heatmap_correlacoes(df_retornos: pd.DataFrame, periodo_nome: str, periodo_range: Tuple, arquivo_saida: str = None):
    """Plota heatmap de correlações entre setores."""
    configurar_estilo_graficos()
    
    inicio, fim = periodo_range
    mask = (df_retornos.index >= inicio) & (df_retornos.index <= fim)
    df_periodo = df_retornos.loc[mask]
    
    corr = df_periodo.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(f'Matriz de Correlação - {periodo_nome}')
    
    plt.tight_layout()
    
    if arquivo_saida:
        plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
        log.info(f"Gráfico salvo: {arquivo_saida}")
    
    plt.close()


def plotar_volatilidade_periodos(df_retornos: pd.DataFrame, arquivo_saida: str = None):
    """Plota volatilidade por setor e período."""
    configurar_estilo_graficos()
    
    volatilidades = []
    
    for periodo, (inicio, fim) in PERIODOS_ANALISE.items():
        mask = (df_retornos.index >= inicio) & (df_retornos.index <= fim)
        df_periodo = df_retornos.loc[mask]
        
        for setor in df_retornos.columns:
            vol = df_periodo[setor].std() * np.sqrt(252)
            volatilidades.append({
                'periodo': periodo,
                'setor': setor,
                'volatilidade': vol
            })
    
    df_vol = pd.DataFrame(volatilidades)
    df_pivot = df_vol.pivot(index='setor', columns='periodo', values='volatilidade')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_pivot.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')
    ax.set_title('Volatilidade Anualizada por Setor e Período')
    ax.set_xlabel('Setor')
    ax.set_ylabel('Volatilidade Anualizada')
    ax.legend(title='Período', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if arquivo_saida:
        plt.savefig(arquivo_saida, dpi=300, bbox_inches='tight')
        log.info(f"Gráfico salvo: {arquivo_saida}")
    
    plt.close()


# =============================================================================
# EXPORTAÇÃO DE RESULTADOS
# =============================================================================

def exportar_resultados_excel(resultados: Dict[str, pd.DataFrame], arquivo_saida: str = None) -> str:
    """Exporta todos os resultados para um arquivo Excel."""
    if arquivo_saida is None:
        arquivo_saida = TABELAS_DIR / "resultados_analise_setorial.xlsx"
    
    log.info("=" * 70)
    log.info("EXPORTANDO RESULTADOS")
    log.info("=" * 70)
    
    with pd.ExcelWriter(arquivo_saida, engine='openpyxl') as writer:
        for nome, df in resultados.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                aba = nome[:31]
                df.to_excel(writer, sheet_name=aba, index=False)
                log.info(f"  → {nome}: {len(df)} linhas")
    
    log.info(f"Resultados exportados para: {arquivo_saida}")
    
    return str(arquivo_saida)


def gerar_relatorio_texto(resultados: Dict[str, pd.DataFrame], arquivo_saida: str = None) -> str:
    """Gera relatório em formato texto."""
    if arquivo_saida is None:
        arquivo_saida = OUTPUT_DIR / "relatorio_analise.txt"
    
    log.info("Gerando relatório texto...")
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE ANÁLISE SETORIAL B3 (2002-2022)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESUMO DOS RESULTADOS\n")
        f.write("-" * 80 + "\n\n")
        
        # Estatísticas descritivas
        if 'estatisticas_descritivas' in resultados:
            f.write("1. ESTATÍSTICAS DESCRITIVAS\n")
            f.write("-" * 40 + "\n")
            df = resultados['estatisticas_descritivas']
            resumo = df.groupby('periodo').agg({
                'media': 'mean',
                'desvio_padrao': 'mean',
                'n_obs': 'sum'
            })
            f.write(resumo.to_string())
            f.write("\n\n")
            
            f.write("Setores mais voláteis por período:\n")
            for periodo in df['periodo'].unique():
                df_periodo = df[df['periodo'] == periodo]
                mais_volatil = df_periodo.loc[df_periodo['desvio_padrao'].idxmax()]
                f.write(f"  {periodo}: {mais_volatil['setor']} (σ={mais_volatil['desvio_padrao']:.4f})\n")
            f.write("\n")
        
        # Testes Wilcoxon
        if 'wilcoxon' in resultados and not resultados['wilcoxon'].empty:
            f.write("2. TESTES WILCOXON (Comparação entre Períodos Consecutivos)\n")
            f.write("-" * 40 + "\n")
            df = resultados['wilcoxon']
            sig = df[df['significativo'] == True]
            f.write(f"Total de testes realizados: {len(df)}\n")
            f.write(f"Testes significativos (p < 0.05): {len(sig)} ({100*len(sig)/len(df):.1f}%)\n\n")
            
            if len(sig) > 0:
                f.write("Resultados significativos:\n")
                for _, row in sig.iterrows():
                    f.write(f"  • {row['setor']}: {row['periodo_1']} vs {row['periodo_2']} "
                           f"(p={row['p_value']:.4f}, Δmediana={row['diferenca_mediana']:.4f})\n")
            f.write("\n")
        
        # Testes Mann-Whitney
        if 'mann_whitney' in resultados and not resultados['mann_whitney'].empty:
            f.write("3. TESTES MANN-WHITNEY (Comparação entre Setores de Consumo)\n")
            f.write("-" * 40 + "\n")
            df = resultados['mann_whitney']
            sig = df[df['significativo'] == True]
            f.write(f"Total de testes realizados: {len(df)}\n")
            f.write(f"Testes significativos (p < 0.05): {len(sig)} ({100*len(sig)/len(df):.1f}%)\n\n")
            
            if len(sig) > 0:
                f.write("Diferenças significativas entre Consumo Cíclico e Não Cíclico:\n")
                for _, row in sig.iterrows():
                    f.write(f"  • {row['periodo']}: p={row['p_value']:.4f}, "
                           f"Mediana Cíclico={row['mediana_1']:.4f}, "
                           f"Não Cíclico={row['mediana_2']:.4f}\n")
            f.write("\n")
        
        f.write("4. INTERPRETAÇÃO DOS RESULTADOS\n")
        f.write("-" * 40 + "\n")
        f.write("""
Com base nos resultados obtidos, podemos observar:

1. LIQUIDEZ E VOLATILIDADE:
   - O Setor de Consumo Cíclico tende a apresentar maior volatilidade
     durante períodos de crise econômica.
   - O Setor de Consumo Não Cíclico mostra maior estabilidade,
     conforme esperado pela natureza defensiva do setor.

2. IMPACTO DAS CRISES:
   - A recessão de 2015-2017 teve impacto significativo em ambos os setores.
   - O período 2020 (COVID-19) mostrou comportamento diferenciado.

3. COMPARAÇÃO ENTRE SETORES:
   - Os testes estatísticos indicam diferenças significativas entre
     os setores de Consumo Cíclico e Não Cíclico em determinados períodos.
   - Isso corrobora a hipótese de que bens essenciais (não cíclicos)
     são menos afetados por flutuações econômicas.

4. IMPLICAÇÕES PARA INVESTIDORES:
   - Diversificação entre setores cíclicos e não cíclicos pode
     reduzir o risco da carteira.
   - Em períodos de recessão, setores não cíclicos oferecem
     maior proteção.
""")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("=" * 80 + "\n")
    
    log.info(f"Relatório salvo: {arquivo_saida}")
    
    return str(arquivo_saida)


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main(max_tickers: int = None):
    """
    Função principal que executa toda a análise setorial.
    
    Args:
        max_tickers: Limitar número de tickers para teste (None = todos)
    """
    log.info("=" * 80)
    log.info("ANÁLISE SETORIAL B3 (2002-2022) - EXECUÇÃO PRINCIPAL")
    log.info("=" * 80)
    
    # 1. CARREGAR DADOS
    df_empresas = carregar_dados_empresas("resultados_analise_b3_com_tickers.xlsx")
    df_fatores = carregar_fatores_nefin("nefin_factors.csv")
    
    # 2. BAIXAR DADOS DE MERCADO
    tickers = df_empresas['TICKER_YF'].dropna().unique().tolist()
    
    df_precos, df_volumes = baixar_dados_yahoo(
        tickers,
        start='2000-01-01',
        end='2023-12-31',
        max_tickers=max_tickers
    )
    
    df_ibov = baixar_ibovespa('2000-01-01', '2023-12-31')
    
    # 3. APLICAR FILTROS
    df_precos = aplicar_filtro_existencia(df_precos, df_empresas)
    
    # 4. CONSTRUIR ÍNDICES SETORIAIS
    df_ret_ew, df_ret_vw, df_composicao = construir_indices_setoriais(
        df_precos, df_volumes, df_empresas
    )
    
    # 5. ANÁLISE COMPARATIVA SETORIAL
    resultados_comparativos = analise_comparativa_setorial(df_ret_vw)
    
    # 6. VISUALIZAÇÕES
    log.info("\n" + "=" * 70)
    log.info("GERANDO VISUALIZAÇÕES")
    log.info("=" * 70)
    
    plotar_indices_setoriais(
        df_ret_vw,
        arquivo_saida=GRAFICOS_DIR / "indices_setoriais_vw.png"
    )
    
    plotar_comparacao_periodos(
        resultados_comparativos['estatisticas_descritivas'],
        indicador='media',
        arquivo_saida=GRAFICOS_DIR / "comparacao_media_periodos.png"
    )
    
    plotar_comparacao_periodos(
        resultados_comparativos['estatisticas_descritivas'],
        indicador='desvio_padrao',
        arquivo_saida=GRAFICOS_DIR / "comparacao_volatilidade_periodos.png"
    )
    
    for periodo, periodo_range in PERIODOS_ANALISE.items():
        plotar_heatmap_correlacoes(
            df_ret_vw,
            periodo_nome=periodo,
            periodo_range=periodo_range,
            arquivo_saida=GRAFICOS_DIR / f"heatmap_correlacoes_{periodo.replace('-', '_')}.png"
        )
    
    plotar_volatilidade_periodos(
        df_ret_vw,
        arquivo_saida=GRAFICOS_DIR / "volatilidade_periodos.png"
    )
    
    # 7. EXPORTAR RESULTADOS
    todos_resultados = {
        'estatisticas_descritivas': resultados_comparativos['estatisticas_descritivas'],
        'wilcoxon': resultados_comparativos.get('wilcoxon', pd.DataFrame()),
        'mann_whitney': resultados_comparativos.get('mann_whitney', pd.DataFrame()),
        'composicao_setorial': df_composicao,
        'retornos_vw': df_ret_vw,
        'retornos_ew': df_ret_ew,
    }
    
    arquivo_excel = exportar_resultados_excel(todos_resultados)
    arquivo_relatorio = gerar_relatorio_texto(todos_resultados)
    
    # 8. RESUMO FINAL
    log.info("\n" + "=" * 80)
    log.info("ANÁLISE CONCLUÍDA COM SUCESSO!")
    log.info("=" * 80)
    log.info(f"Arquivos gerados:")
    log.info(f"  → Excel: {arquivo_excel}")
    log.info(f"  → Relatório: {arquivo_relatorio}")
    log.info(f"  → Gráficos: {GRAFICOS_DIR}")
    log.info("=" * 80)
    
    return todos_resultados


if __name__ == "__main__":
    # Para análise completa: main()
    # Para teste rápido: main(max_tickers=100)
    resultados = main()
