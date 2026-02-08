# Metodologia - Impacto dos Ciclos Eleitorais na B3 (2002-2022)

## Desenho do Estudo
Estudo de Evento aplicado aos 6 ciclos eleitorais presidenciais brasileiros.

### Principais Características
- Janela de estimação: [-252, -30] dias úteis (MacKinlay, 1997)
- Modelos: CAPM + Fama-French 3 Fatores (NEFIN)
- Ponderação: Equal-Weighted (principal) e Volume-Weighted (robustez)
- Filtros: Liquidez (80%), Penny stocks (R$1,00), Winsorização (1%/99%)
- Testes: t simples, Silva et al. (2015), BMP (1991), HAC-Newey-West
- Robustez: Placebo, DiD (Estatais vs Privadas), Sharpe Ratio

## Limitações
- Viés de sobrevivência (mitigado com mapeamento e filtros)
- Fonte yfinance (lacunas em tickers antigos)
- Não estabelece causalidade, apenas evidência associativa

(Continue com mais detalhes conforme necessário)
