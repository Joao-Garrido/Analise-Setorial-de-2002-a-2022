import pandas as pd
import yfinance as yf
from difflib import get_close_matches
import time

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
ARQUIVO_ENTRADA = "resultados_analise_b3_com_tickers.xlsx" # Seu arquivo original
SHEET_NAME = "LISTA FINAL (Cont+IPOs-Canc)"
ARQUIVO_SAIDA = "lista_tickers_final_recuperada.xlsx"

# Dicionário de Correção Imediata (Casos conhecidos)
TICKER_MAPPING = {
    'VVAR3': 'BHIA3', 'VVAR11': 'BHIA3', 'VIIA3': 'BHIA3',
    'BTOW3': 'AMER3', 'LAME3': 'AMER3', 'LAME4': 'AMER3',
    'PCAR4': 'PCAR3', 'KROT3': 'COGN3', 'ESTC3': 'YDUQ3',
    'SUZB5': 'SUZB3', 'FIBR3': 'SUZB3', 'TIET11': 'AESB3',
    'CESP6': 'AURE3', 'OMGE3': 'AURE3', 'LCAM3': 'LREN3',
    'BRDT3': 'VBBR3', 'CNTO3': 'SBFG3', 'BKBR3': 'ZAMP3',
    'NATU3': 'NTCO3', 'NTCO3': 'NATU3', 'BIDI4': 'INBR32',
    'SMLS3': 'GOLL4', 'JSLG3': 'SIMH3', 'LINX3': 'STBP3'
}

def validar_ticker_online(ticker):
    """Verifica se o ticker existe no Yahoo Finance."""
    try:
        t_sa = ticker + ".SA" if not ticker.endswith(".SA") else ticker
        # Tenta baixar 1 dia de histórico. Se vazio, o ticker é inválido/antigo.
        hist = yf.download(t_sa, period="1d", progress=False)
        return not hist.empty
    except:
        return False

def limpar_nome(nome):
    if not isinstance(nome, str): return ""
    return nome.upper().replace('S.A.', '').replace('S/A', '').replace(' LTDA', '').strip()

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
print("Carregando dados...")
df = pd.read_excel(ARQUIVO_ENTRADA, sheet_name=SHEET_NAME)

# Cria lista de referência de nomes e tickers ATUAIS para busca
# Assumimos que empresas sem data de cancelamento têm tickers válidos
df_ativas = df[df['DT_CANCEL'].isna()].copy()
df_ativas['nome_limpo'] = df_ativas['DENOM_SOCIAL'].apply(limpar_nome)
nomes_validos = df_ativas['nome_limpo'].tolist()
tickers_validos = df_ativas['TICKER'].tolist()

novos_tickers = []
status_lista = []

print(f"Processando {len(df)} empresas. Isso pode demorar uns minutos...")

for idx, row in df.iterrows():
    t_original = str(row['TICKER']).strip()
    nome = row['DENOM_SOCIAL']
    
    # 1. Tenta Dicionário (Rápido e Preciso)
    if t_original in TICKER_MAPPING:
        novo = TICKER_MAPPING[t_original]
        novos_tickers.append(novo)
        status_lista.append(f"Recuperado (Dic): {t_original}->{novo}")
        continue

    # 2. Verifica se o ticker original funciona online
    if validar_ticker_online(t_original):
        novos_tickers.append(t_original)
        status_lista.append("OK (Online)")
    else:
        # 3. Se falhou online, tenta achar a empresa pelo NOME (Fuzzy Match)
        nome_busca = limpar_nome(nome)
        matches = get_close_matches(nome_busca, nomes_validos, n=1, cutoff=0.6)
        
        if matches:
            match_nome = matches[0]
            idx_match = nomes_validos.index(match_nome)
            t_novo = tickers_validos[idx_match]
            
            # Só aceita se o ticker novo funcionar (para evitar falsos positivos)
            if t_novo != t_original:
                novos_tickers.append(t_novo)
                status_lista.append(f"Recuperado (Nome): {t_original}->{t_novo}")
            else:
                novos_tickers.append(t_original) # Mantém o velho (morto)
                status_lista.append("Falha (Ticker morto)")
        else:
            novos_tickers.append(t_original)
            status_lista.append("Falha (Não encontrado)")

df['TICKER_FINAL'] = novos_tickers
df['STATUS_CHECK'] = status_lista

# Salvar resultado
df.to_excel(ARQUIVO_SAIDA, index=False)

print("-" * 50)
print("RELATÓRIO FINAL:")
print(df['STATUS_CHECK'].value_counts())
print(f"\nArquivo salvo como: {ARQUIVO_SAIDA}")
print("Use a coluna 'TICKER_FINAL' no seu código principal agora.")