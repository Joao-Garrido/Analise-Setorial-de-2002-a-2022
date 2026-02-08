#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DEMONSTRAÇÃO RÁPIDA - ANÁLISE SETORIAL B3
================================================================================
Versão simplificada para demonstração rápida (50 tickers)
================================================================================
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Períodos
PERIODOS = {
    '2009-2011': (datetime(2009, 1, 1), datetime(2011, 12, 31)),
    '2012-2014': (datetime(2012, 1, 1), datetime(2014, 12, 31)),
    '2015-2017': (datetime(2015, 1, 1), datetime(2017, 12, 31)),
    '2018-2020': (datetime(2018, 1, 1), datetime(2020, 12, 31)),
}

OUTPUT_DIR = Path("/mnt/okcomputer/output/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    log.info("=" * 70)
    log.info("DEMONSTRAÇÃO RÁPIDA - ANÁLISE SETORIAL B3")
    log.info("=" * 70)
    
    # 1. Carregar dados
    log.info("Carregando dados das empresas...")
    df_empresas = pd.read_excel("/mnt/okcomputer/upload/resultados_analise_b3_com_tickers.xlsx")
    df_empresas['DT_REG'] = pd.to_datetime(df_empresas['DT_REG'], errors='coerce')
    df_empresas['TICKER_YF'] = df_empresas['TICKER'].apply(
        lambda x: x if pd.isna(x) or str(x).endswith('.SA') else f"{x}.SA"
    )
    
    # Selecionar apenas 50 tickers para demonstração rápida
    tickers = df_empresas['TICKER_YF'].dropna().unique()[:50].tolist()
    log.info(f"Usando {len(tickers)} tickers para demonstração")
    
    # 2. Baixar dados
    log.info("Baixando dados do Yahoo Finance...")
    dados = yf.download(tickers=tickers, start='2008-01-01', end='2021-12-31', 
                        auto_adjust=True, progress=False, threads=True)
    df_precos = dados['Close']
    df_volumes = dados['Volume']
    
    log.info(f"Dados baixados: {len(df_precos)} observações x {len(df_precos.columns)} ativos")
    
    # 3. Calcular retornos
    ret = np.log(df_precos / df_precos.shift(1))
    
    # 4. Construir índices setoriais simples (EW)
    t2s = dict(zip(df_empresas['TICKER_YF'], df_empresas['SETOR_B3']))
    
    setores = ["Consumo Cíclico", "Consumo Não Cíclico", "Bens Industriais", 
               "Financeiro", "Materiais Básicos"]
    
    indices_setoriais = {}
    for setor in setores:
        cols = [c for c in ret.columns if t2s.get(c) == setor]
        if cols:
            indices_setoriais[setor] = ret[cols].mean(axis=1)
    
    df_indices = pd.DataFrame(indices_setoriais)
    log.info(f"Índices construídos: {df_indices.shape[1]} setores")
    
    # 5. Estatísticas descritivas
    log.info("Calculando estatísticas descritivas...")
    estatisticas = []
    for periodo, (inicio, fim) in PERIODOS.items():
        mask = (df_indices.index >= inicio) & (df_indices.index <= fim)
        df_periodo = df_indices.loc[mask]
        
        for setor in df_indices.columns:
            ret_setor = df_periodo[setor].dropna()
            if len(ret_setor) > 10:
                estatisticas.append({
                    'periodo': periodo,
                    'setor': setor,
                    'media': ret_setor.mean(),
                    'mediana': ret_setor.median(),
                    'desvio_padrao': ret_setor.std(),
                    'vol_anualizada': ret_setor.std() * np.sqrt(252)
                })
    
    df_est = pd.DataFrame(estatisticas)
    
    # 6. Teste Mann-Whitney entre Consumo Cíclico e Não Cíclico
    log.info("Realizando testes estatísticos...")
    mw_results = []
    for periodo, (inicio, fim) in PERIODOS.items():
        mask = (df_indices.index >= inicio) & (df_indices.index <= fim)
        
        if 'Consumo Cíclico' in df_indices.columns and 'Consumo Não Cíclico' in df_indices.columns:
            ret1 = df_indices.loc[mask, 'Consumo Cíclico'].dropna()
            ret2 = df_indices.loc[mask, 'Consumo Não Cíclico'].dropna()
            
            if len(ret1) >= 10 and len(ret2) >= 10:
                stat, p_value = mannwhitneyu(ret1, ret2, alternative='two-sided')
                mw_results.append({
                    'periodo': periodo,
                    'mediana_ciclico': ret1.median(),
                    'mediana_nao_ciclico': ret2.median(),
                    'p_value': p_value,
                    'significativo': p_value < 0.05
                })
    
    df_mw = pd.DataFrame(mw_results)
    
    # 7. Visualizações
    log.info("Gerando visualizações...")
    
    # Gráfico 1: Índices acumulados
    fig, ax = plt.subplots(figsize=(12, 6))
    ret_acum = (1 + df_indices).cumprod()
    for setor in ret_acum.columns:
        ax.plot(ret_acum.index, ret_acum[setor], label=setor, linewidth=1.5)
    ax.set_title('Índices Setoriais B3 - Demonstração (50 tickers)')
    ax.set_xlabel('Data')
    ax.set_ylabel('Retorno Acumulado')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axvspan(datetime(2015, 1, 1), datetime(2017, 12, 31), alpha=0.2, color='red')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "demo_indices.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Volatilidade por período
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot = df_est.pivot(index='setor', columns='periodo', values='vol_anualizada')
    df_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Volatilidade Anualizada por Setor e Período')
    ax.set_ylabel('Volatilidade')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "demo_volatilidade.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Exportar resultados
    log.info("Exportando resultados...")
    
    with pd.ExcelWriter(OUTPUT_DIR / "demo_resultados.xlsx", engine='openpyxl') as writer:
        df_est.to_excel(writer, sheet_name='estatisticas', index=False)
        df_mw.to_excel(writer, sheet_name='mann_whitney', index=False)
        df_indices.to_excel(writer, sheet_name='retornos')
    
    # Relatório texto
    with open(OUTPUT_DIR / "demo_relatorio.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DEMONSTRAÇÃO RÁPIDA - ANÁLISE SETORIAL B3\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("RESUMO DOS RESULTADOS\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("1. ESTATÍSTICAS DESCRITIVAS\n")
        f.write("-" * 40 + "\n")
        resumo = df_est.groupby('periodo').agg({
            'media': 'mean',
            'vol_anualizada': 'mean'
        })
        f.write(resumo.to_string())
        f.write("\n\n")
        
        f.write("2. TESTES MANN-WHITNEY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de testes: {len(df_mw)}\n")
        sig = df_mw[df_mw['significativo'] == True]
        f.write(f"Significativos (p < 0.05): {len(sig)}\n\n")
        
        for _, row in df_mw.iterrows():
            status = "*" if row['significativo'] else ""
            f.write(f"  {row['periodo']}: p={row['p_value']:.4f} {status}\n")
            f.write(f"    Cíclico: {row['mediana_ciclico']:.4f}")
            f.write(f" | Não Cíclico: {row['mediana_nao_ciclico']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("=" * 70 + "\n")
    
    log.info("=" * 70)
    log.info("DEMONSTRAÇÃO CONCLUÍDA!")
    log.info("=" * 70)
    log.info(f"Arquivos gerados em: {OUTPUT_DIR}")
    log.info("  → demo_indices.png")
    log.info("  → demo_volatilidade.png")
    log.info("  → demo_resultados.xlsx")
    log.info("  → demo_relatorio.txt")
    
    return df_est, df_mw


if __name__ == "__main__":
    estatisticas, testes = main()
    
    # Mostrar resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    print("\nEstatísticas por Período:")
    print(estatisticas.groupby('periodo')[['media', 'vol_anualizada']].mean())
    
    print("\n\nTestes Mann-Whitney (Cíclico vs Não Cíclico):")
    print(testes[['periodo', 'p_value', 'significativo']])
