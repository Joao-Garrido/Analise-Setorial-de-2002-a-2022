# Metodologia e Limitações — v3

## 1. Desenho do Estudo

Estudo de Evento (MacKinlay, 1997) aplicado aos ciclos eleitorais presidenciais
brasileiros: 2002, 2006, 2010, 2014, 2018, 2022.

### 1.1 Índices Setoriais

11 setores B3, em dois esquemas de ponderação:
- **Equal-Weighted (EW)**: retorno médio simples das empresas ativas.
  *Nota:* Amplifica small caps. Reportado como análise principal.
- **Volume-Weighted (VW)**: retorno ponderado por volume financeiro médio 20 dias.
  Proxy de market cap. Reportado como robustez.

### 1.2 Janela de Estimação

**[-252, -30] dias úteis antes do 1º turno** (~222 d.u., ~10,5 meses).
Janela móvel que evita contaminação por choques pré-evento (MacKinlay, 1997).

### 1.3 Modelos de Retorno Normal

**(A) Modelo de Mercado:** R_{i,t} = α + β·R_{m,t} + ε
  - Benchmark: Ibovespa (^BVSP)
  - Estimação via OLS com erros padrão HAC-Newey-West (1 lag)

**(B) Fama-French 3 Fatores (BR):** R_i−Rf = α + β·MktRf + s·SMB + h·HML + ε
  - Fatores: NEFIN-USP (nefin.com.br)
  - Controla por tamanho (SMB) e valor (HML)
  - Estimação via OLS-HAC

### 1.4 Janelas de Evento

| Janela | Definição | Motivação |
|---|---|---|
| Antecipação 45 d.u. | [−45, −1] 1ºT | Efeito Propaganda / Pricing-in |
| Antecipação 60 d.u. | [−60, −1] 1ºT | Robustez |
| Reação Curta 1ºT | [−5, +5] | Reação imediata |
| Reação Média | [−10, +10] | Robustez |
| Reação Ampla | [−20, +20] | Robustez |
| Reação 2ºT | [−5, +5] 2ºT | Resolução de incerteza |
| Ciclo Interno | 1ºSem vs 2ºSem | Expectativa vs. Disputa |
| Estendida | Jul–Dez | Efeito agregado |

### 1.5 Inferência Estatística

1. **Teste t simples** (longitudinal): t = AR̄·√n / σ(AR)
2. **Teste t Silva et al. (2015)**: csd_t = √(t·var + 2(t−1)·cov) — corrige autocorrelação
3. **Teste BMP (1991)**: SCAR = CAR/(σ·√T) — controla event-induced variance
4. **OLS-HAC**: Newey-West com 1 lag na estimação dos parâmetros

### 1.6 Testes de Robustez

- Janelas alternativas: [−60,−1], [−10,+10], [−20,+20]
- Cenário sem crises (exclui 2008/2020)
- Teste placebo: 100 pseudo-eventos em anos não-eleitorais
- DiD: Regulados vs. Não-Regulados (teste Welch)
- Sharpe Ratio + teste formal de excesso de retorno vs. Selic
- Ponderação alternativa: Volume-Weighted (VW)
- Modelo alternativo: Fama-French 3 fatores (NEFIN)

## 2. Limitações

1. **Viés de sobrevivência**: ~45% dos tickers não possuem dados no yfinance.
   Small caps e deslistadas sub-representadas, especialmente pré-2006.
   Mitigação: mapeamento De-Para + filtro de liquidez.

2. **Ponderação**: EW amplifica small caps. VW reportado como robustez.

3. **Clusterização de volatilidade**: Mitigada por HAC + BMP, mas não eliminada.

4. **Endogeneidade**: Resultados são evidência consistente com risco político —
   NÃO causalidade. Placebo e DiD oferecem validação parcial.

5. **Fonte de dados (yfinance)**: API gratuita, sujeita a falhas e lacunas.
   Solução definitiva: COTAHIST/B3 (séries históricas brutas).

6. **Benchmarking Selic/IPCA**: Ilustrativo. O teste formal de excesso usa
   Sharpe Ratio + t-test do excesso de retorno diário.

## 3. Referências

- Boehmer, Musumeci & Poulsen (1991). JFE 30(2).
- Fama & French (1993). JFE 33(1), 3-56.
- MacKinlay (1997). JEL 35(1), 13-39.
- Newey & West (1987). Econometrica 55(3), 703-708.
- Nordhaus (1975). RES 42(2), 169-190.
- Silva et al. (2015). Retornos anormais IPO BMF&Bovespa.
