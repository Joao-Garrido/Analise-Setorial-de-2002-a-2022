# Refatoração Acadêmica – Explicação Detalhada das Alterações

## Visão Geral

O script foi refatorado de uma **análise exploratória funcional** para uma **análise de nível acadêmico** (padrão working paper). Abaixo, cada alteração é justificada com base nas críticas recebidas.

---

## 1. Viés de Dados: Mapeamento De-Para de Tickers (CRÍTICO)

**Problema:** ~55% dos tickers falhavam no yfinance por serem códigos antigos, deslistados ou renomeados. Isso criava viés de sobrevivência severo.

**Solução implementada:**
- Dicionário `TICKER_MAPPING` com ~35 pares De-Para dos tickers mais conhecidos (VVAR3→BHIA3, KROT3→COGN3, BTOW3→AMER3, FIBR3→SUZB3, etc.)
- Coluna `TICKER_MAPEADO` no DataFrame de empresas
- Log detalhado: quantos tickers mapeados, % de perda por setor
- **Tabela de diagnóstico de download** (`diagnostico_download.csv`) com status ok/falha + setor
- **Disclaimer forte** no docstring e no arquivo `metodologia_limitacoes.md`

**Impacto esperado:** Recuperação de ~10-20 tickers adicionais de empresas relevantes.

---

## 2. Inferência Estatística: Silva et al. (2015) + BMP (CRÍTICO)

**Problema:** O t-stat anterior era univariado e otimista (`car/n_dias * sqrt(n) / std`), ignorando correlação cross-setorial e autocorrelação serial.

**Solução implementada:**
- Função `tstat_car_silva()`: implementa Eq. 6-8 do artigo de referência
  - `csd_t = sqrt(t · var_média + 2·(t−1) · cov_média)`
  - Corrige autocorrelação de primeira ordem na série de ARs
- Função `tstat_car_bmp()`: implementa Boehmer-Musumeci-Poulsen (1991)
  - Normaliza CARs pelo sigma da estimação (SCAR = CAR / σ·√T)
  - Controla heteroscedasticidade event-induced
- **Três colunas de p-value** reportadas: simples, Silva, BMP
- Cross-sectional tests calculados por grupo (ano × janela), usando os 11 setores como unidades

**Impacto esperado:** p-values mais conservadores e defensáveis.

---

## 3. Janela de Estimação Móvel (CRÍTICO)

**Problema:** Janela fixa no ano calendário anterior capturava choques (2001 para 2002, 2007 para 2008, 2019 para 2020).

**Solução implementada:**
- Janela móvel: **[-252, -30] dias úteis antes do 1º turno**
- ~222 dias úteis de estimação, isolados do evento
- Parâmetros configuráveis: `ESTIMACAO_INICIO_DU`, `ESTIMACAO_FIM_DU`
- Colunas `est_inicio` e `est_fim` nos resultados para transparência

**Impacto esperado:** α e β mais limpos, especialmente para 2002 e 2018.

---

## 4. Janelas Alternativas de Robustez

**Problema:** Apenas duas janelas (antecipação 45d, reação 10d) – insuficiente para robustez.

**Solução implementada:**
- **7 janelas de evento** + 2 de ciclo interno:
  - Antecipação: [-45,-1] e [-60,-1] (robustez)
  - Reação 1º turno: [-5,+5], [-10,+10], [-20,+20]
  - Reação 2º turno: [-5,+5]
  - Ciclo interno: 1º sem / 2º sem
  - Estendida: últimos 6 meses
- Todas configuráveis via dicionários `JANELAS_EVENTO` e `JANELAS_2TURNO`

---

## 5. Filtro de Liquidez

**Problema:** Empresas com dados esparsos (poucos pregões) distorcem retornos.

**Solução implementada:**
- Função `aplicar_filtro_liquidez()`: exige ≥80% dos pregões do ano
- Parâmetro configurável: `MIN_PREGOES_PCT = 0.80`

---

## 6. Corte por N Mínimo de Empresas

**Problema:** Setores com N < 5 empresas em dado ano geram inferência instável.

**Solução implementada:**
- Parâmetro `MIN_EMPRESAS_SETOR = 5`
- Tabela de composição setorial (`df_composicao`) registra N por setor/ano

---

## 7. Teste Placebo (Pseudo-Eventos)

**Problema:** Sem contrafactual, não é possível avaliar se os CARs observados são diferentes do "ruído normal" do mercado.

**Solução implementada:**
- Função `executar_teste_placebo()`: gera 100 datas aleatórias em anos não-eleitorais
- Calcula CARs para antecipação e reação em cada pseudo-evento
- Gráfico comparativo: distribuição de CARs eleitorais vs. placebo
- CSV exportado: `placebo_test_results.csv`

---

## 8. Tabela de Sobrevivência

**Problema:** Sem transparência sobre quantas empresas existem por setor/ano.

**Solução implementada:**
- Função `gerar_tabela_sobrevivencia()`: cruza DT_REG/DT_CANCEL com dados obtidos
- Colunas: n_esperado, n_com_dados, cobertura_pct
- Exportada como `tabela_sobrevivencia.csv` e aba no Excel

---

## 9. Ponderação e Nomenclatura

**Problema:** Gráficos não explicitavam que os índices são equal-weighted.

**Solução:** Todos os títulos de gráficos e tabelas incluem "(Equal-Weighted)" ou "(Índice Setorial Equal-Weighted)". Nota de rodapé na metodologia explica a escolha e suas implicações.

---

## 10. Exclusão de Crises e Benchmarking

**Problema:** Mask de crise era assimétrica; Selic/IPCA misturavam métricas.

**Solução:**
- Cenário "sem_crises" aplica NaN aos anos 2008 e 2020 inteiros (retornos setoriais e Ibovespa)
- Benchmarking Selic/IPCA marcado como "ilustrativo, não constitui teste formal" no código e na metodologia

---

## 11. Linguagem Acadêmica

- "impacto causal" → "evidência consistente com risco político"
- Adicionado arquivo `metodologia_limitacoes.md` com seções prontas para artigo
- Disclaimer forte no docstring do script

---

## 12. Estrutura Modular e Logs

- Cada etapa é uma função independente, com logging estruturado
- Cache de preços (pickle) evita re-download
- Diagnóstico de download exportado para análise de perda de dados

---

## Resumo: v1 → v2

| Aspecto | v1 | v2 |
|---|---|---|
| Mapeamento de tickers | Nenhum | 35+ pares De-Para |
| Janela de estimação | Ano calendário anterior | [-252, -30] d.u. móvel |
| Testes de significância | t simples (1 teste) | Simples + Silva + BMP (3 testes) |
| Janelas de evento | 2 | 9 (incluindo robustez) |
| Teste placebo | Não | 100 pseudo-eventos |
| Filtro de liquidez | Não | ≥80% pregões |
| Tabela de sobrevivência | Não | Sim (N por setor/ano) |
| Disclaimer acadêmico | Não | Sim (metodologia + limitações) |
| Heatmaps com significância | Não | Sim (* para p<0.05) |
| Benchmarking | "teste" | "ilustrativo" |
