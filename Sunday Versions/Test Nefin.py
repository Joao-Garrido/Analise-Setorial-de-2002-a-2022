import pandas as pd
import logging

# Configuração simples de log para ver o que acontece
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TesteNEFIN")

# --- COLE AQUI A SUA FUNÇÃO CORRIGIDA ---
def baixar_fatores_nefin():
    log.info("Testando download NEFIN...")
    url_csv = "https://nefin.com.br/resources/risk_factors/nefin_factors.csv"
    
    try:
        # Tenta ler
        df = pd.read_csv(url_csv)

        # 1. Arrumar Data
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
            df = df.set_index("date")
        elif {"year", "month", "day"}.issubset(df.columns):
             df["date"] = pd.to_datetime(df[["year", "month", "day"]])
             df = df.set_index("date")
        
        # 2. Renomear
        rename_map = {"Rm_minus_Rf": "Mkt_Rf", "Risk_Free": "Rf"}
        df = df.rename(columns=rename_map)

        # 3. Filtrar
        cols = ["Mkt_Rf", "SMB", "HML", "Rf"]
        if not set(cols).issubset(df.columns):
            log.error(f"Colunas faltando! Tem: {df.columns.tolist()}")
            return None

        return df[cols]

    except Exception as e:
        log.error(f"Erro: {e}")
        return None

# --- EXECUÇÃO DO TESTE ---
print("-" * 50)
print("INICIANDO TESTE...")
df_resultado = baixar_fatores_nefin()

if df_resultado is not None:
    print("\n✅ SUCESSO! O DataFrame foi gerado.")
    print(f"Dimensões: {df_resultado.shape} (Linhas, Colunas)")
    print(f"Período: {df_resultado.index.min().date()} até {df_resultado.index.max().date()}")
    print("\n🔍 Primeiras 5 linhas:")
    print(df_resultado.head())
    
    # Verifica se tem valores nulos
    if df_resultado.isnull().values.any():
        print("\n⚠️ AVISO: Existem valores nulos (NaN) nos dados.")
    else:
        print("\n✅ Dados limpos (sem NaN).")
else:
    print("\n❌ FALHA: A função retornou None.")
print("-" * 50)