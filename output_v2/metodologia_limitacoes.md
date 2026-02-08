# Metodologia e Limitações

## 1. Desenho do Estudo

Este trabalho emprega a metodologia de **Estudo de Evento** (Event Study), conforme
formalizada por MacKinlay (1997) e adaptada por Silva et al. (2015) para o mercado
brasileiro. O evento de interesse é o ciclo eleitoral presidencial brasileiro, analisado
em seis eleições: 2002, 2006, 2010, 2014, 2018 e 2022.

### 1.1 Construção dos Índices Setoriais

Para cada um dos 11 setores da classificação oficial da B3, construímos um **índice
equal-weighted** (pesos iguais) com base nos retornos logarítmicos diários de todas
as empresas ativas em cada dia de negociação. A ponderação igual visa isolar o
comportamento médio do setor, diluindo riscos idiossincráticos de empresas individuais.

**Nota:** A ponderação equal-weighted tende a amplificar a contribuição de empresas
de menor capitalização. Resultados com ponderação por valor de mercado podem diferir.

### 1.2 Janela de Estimação

A janela de estimação compreende os **[-252, -30] dias úteis** antes da data do
1º turno de cada eleição, totalizando aproximadamente 222 dias úteis (~10,5 meses).
Esta janela móvel evita contaminação por choques que ocorram imediatamente antes
do evento e é consistente com a recomendação de MacKinlay (1997).

### 1.3 Modelo de Mercado

O retorno normal esperado é estimado via regressão OLS do modelo de mercado:

    R_{i,t} = α_i + β_i · R_{m,t} + ε_{i,t}

onde R_{m,t} é o retorno do Ibovespa (^BVSP). O retorno anormal (AR) na janela de
evento é:

    AR_{i,t} = R_{i,t} − (α̂_i + β̂_i · R_{m,t})

O retorno anormal acumulado (CAR) é a soma dos ARs na janela.

### 1.4 Janelas de Evento

| Janela | Definição | Motivação |
|---|---|---|
| Antecipação (45 d.u.) | [−45, −1] antes do 1º turno | Efeito Propaganda / Pricing-in |
| Antecipação (60 d.u.) | [−60, −1] – robustez | Sensibilidade à largura |
| Reação Curta 1º Turno | [−5, +5] em torno do 1º turno | Reação imediata |
| Reação Média | [−10, +10] – robustez | Sensibilidade à largura |
| Reação Ampla | [−20, +20] – robustez | Persistência do efeito |
| Reação 2º Turno | [−5, +5] em torno do 2º turno | Resolução de incerteza |
| Ciclo Interno (1º sem) | Jan–Jun do ano eleitoral | Expectativa |
| Ciclo Interno (2º sem) | Jul–Dez do ano eleitoral | Disputa e resultado |
| Estendida | Últimos 6 meses | Efeito agregado |

### 1.5 Inferência Estatística

Reportamos três testes de significância:

1. **Teste t simples** (longitudinal): t = AR̄ · √n / σ(AR)
2. **Teste t corrigido** (Silva et al., 2015): utiliza desvio padrão corrigido por
   autocorrelação: csd_t = √(t·var + 2·(t−1)·cov)
3. **Teste BMP** (Boehmer, Musumeci & Poulsen, 1991): normaliza CARs pelo sigma
   da estimação, controlando heteroscedasticidade event-induced.

### 1.6 Testes de Robustez

- **Janelas alternativas**: [−60,−1], [−10,+10], [−20,+20]
- **Cenário sem crises**: exclui dados de 2008 e 2020
- **Teste placebo**: 100 pseudo-eventos em anos não-eleitorais

## 2. Limitações

1. **Viés de sobrevivência**: A amostra inclui apenas empresas com dados disponíveis
   no Yahoo Finance (~45% dos tickers da base original). Small caps, empresas
   deslistadas e falidas são sub-representadas, especialmente antes de 2006.

2. **Ponderação equal-weighted**: Amplifica contribuição de small caps, podendo
   inflar volatilidade e betas setoriais.

3. **Clusterização de volatilidade**: Os p-values podem superestimar significância
   em períodos de alta volatilidade agrupada (e.g., crises de 2002 e 2008).

4. **Modelo de fator único**: O modelo de mercado (CAPM simplificado) não controla
   fatores adicionais (tamanho, valor, momentum). Extensões com Fama-French de
   3 fatores são recomendadas.

5. **Endogeneidade**: Não é possível estabelecer causalidade. Os resultados devem
   ser interpretados como **evidência consistente com risco político**, não como
   impacto causal dos ciclos eleitorais.

6. **Benchmarking Selic/IPCA**: Os ganhos/perdas reais reportados são ilustrativos
   e não constituem teste formal de significância.

## 3. Referências

- Boehmer, E., Musumeci, J. & Poulsen, A. (1991). Event-study methodology under
  conditions of event-induced variance. *Journal of Financial Economics*, 30(2).
- MacKinlay, A. C. (1997). Event Studies in Economics and Finance. *Journal of
  Economic Literature*, 35(1), 13-39.
- Nordhaus, W. D. (1975). The political business cycle. *Review of Economic Studies*, 42(2).
- Silva, W. A. M. et al. (2015). Evidências de retornos anormais nos processos de
  IPO na BMF&Bovespa no período de 2004 a 2013.
