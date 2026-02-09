<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Model-Fama--French%203%20Fatores-1D3557?style=for-the-badge" alt="Fama-French">
  <img src="https://img.shields.io/badge/Status-Conclu%C3%ADdo-2ca02c?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Dados-NEFIN--USP%20%7C%20Yahoo%20Finance-informational?style=for-the-badge" alt="Dados">
</p>

---

# 🗳️ Risco Político e Ciclos Eleitorais no Brasil (2002–2022)

> **Análise quantitativa setorial do impacto das eleições presidenciais no mercado de capitais brasileiro utilizando Fama-French 3 Fatores e Buy-and-Hold Abnormal Returns (BHAR).**

---

## 📖 Sobre o Projeto

Este projeto investiga a hipótese de que o **Risco Político** em anos eleitorais no Brasil não afeta a bolsa de valores de forma homogênea, mas sim através de canais específicos — setores regulados versus não regulados.

Utilizando a metodologia de **Estudo de Eventos** (*Event Study*), analisamos o comportamento anormal das ações durante a **janela de antecipação eleitoral** (início do Horário Gratuito de Propaganda Eleitoral até o 1º Turno) nas eleições de **2002, 2006, 2010, 2014, 2018 e 2022**.

---

## 🎯 Principais Descobertas

A análise estatística revelou uma clara dicotomia setorial:

| Setor | ABHAR Médio | Interpretação |
|:------|:-----------:|:--------------|
| **Utilidade Pública** (Elétricas / Saneamento) | **+6,07%** | Atuou como *hedge*, impulsionado por pautas de privatização |
| **Setor Financeiro** | **+0,08%** | Neutralidade estatística — alta resiliência a choques políticos |
| **Consumo e Varejo** | **−8,11%** | Maior penalização — sensível à inflação e volatilidade cambial |

---

## ⚙️ Metodologia

O diferencial deste estudo é a **robustez econométrica** para isolar o "Risco Político" do "Risco de Mercado".

### 1. Modelo de Precificação — Fama-French 3 Fatores (1993)

O retorno esperado de cada ativo é estimado via regressão OLS na janela de estimação:

$$R_{i,t} - R_{f,t} = \alpha_i + \beta_{1,i}(R_m - R_f)_t + \beta_{2,i} \cdot SMB_t + \beta_{3,i} \cdot HML_t + \varepsilon_{i,t}$$

- **Dados:** Fatores de risco do [NEFIN-USP](https://nefin.com.br/).
- **Estimação:** 252 dias úteis (≈ 1 ano), com erros robustos HAC (Newey-West, 5 lags).

### 2. Retorno Anormal — BHAR (Buy-and-Hold)

Em vez da soma aritmética (CAR), utilizamos **BHAR** para capturar o efeito dos juros compostos na riqueza do investidor ao longo da janela de ~45 dias:

$$BHAR_i = \prod_{t=1}^{T}(1 + R_{i,t}) - \prod_{t=1}^{T}(1 + E[R_{i,t}])$$

Onde o retorno esperado na janela de evento é:

$$E[R_{i,t}] = \hat{\alpha}_i + \hat{\beta}_{1,i}(R_m - R_f)_t + \hat{\beta}_{2,i} \cdot SMB_t + \hat{\beta}_{3,i} \cdot HML_t + R_{f,t}$$

### 3. Janelas Dinâmicas (via HGPE)

As janelas **não são fixas**. São determinadas pelo calendário oficial do TSE:

```
Janela de Estimação              Gap     Janela de Evento
[───── 252 DU (treino) ─────]  30 DU  [── HGPE → Véspera 1ºT ──]
```

| Eleição | Início HGPE | 1º Turno | Dias Úteis (Evento) |
|:-------:|:-----------:|:--------:|:-------------------:|
| 2002 | 20/ago | 06/out | ~30 |
| 2006 | 15/ago | 01/out | ~32 |
| 2010 | 17/ago | 03/out | ~32 |
| 2014 | 19/ago | 05/out | ~32 |
| 2018 | 31/ago | 07/out | ~25 |
| 2022 | 26/ago | 02/out | ~25 |

### 4. Agregação e Testes

- **Value-Weighted:** Índices setoriais ponderados pelo volume financeiro médio na janela de estimação.

$$ABHAR_{setor} = \sum_i w_i \cdot BHAR_i \quad \text{onde} \quad w_i = \frac{\bar{V}_i}{\sum_j \bar{V}_j}$$

- **Teste t-Student:** $H_0$: Média dos BHARs = 0.
- **Teste de Wilcoxon:** $H_0$: Mediana dos BHARs = 0 (robustez contra *outliers*).

### 5. Robustez

| Teste | Descrição |
|:------|:----------|
| **Placebo** | Mesma metodologia aplicada em anos não-eleitorais (2003, 2007, 2011, 2013, 2017, 2019) |
| **Diff-in-Diff** | Comparação de médias entre setores Regulados (Petróleo, Utilidade Pública, Financeiro) e Não Regulados |

---

## 📊 Visualizações Geradas

O script produz automaticamente os seguintes outputs na pasta `output_ff3_bhar/`:

### Mapa de Calor — Risco Político Setorial

Magnitude do retorno anormal (ABHAR %) por setor × ano eleitoral, com escala fixa em ±40% para foco na variação relevante.

### Linha do Tempo — BHAR Acumulado

Evolução dia-a-dia do retorno anormal na janela de antecipação, comparando Regulados vs. Não Regulados (média de todas as eleições, com bandas de ±1σ).

### Ranking Setorial

Barras horizontais com o ABHAR médio de cada setor ao longo dos 6 ciclos eleitorais.

### Mapa Risco × Retorno

Scatter plot cruzando o retorno médio (eixo Y) com a volatilidade entre eleições (eixo X).

---

## 🚀 Como Executar

O projeto foi desenhado para ser **plug-and-play**. Ele baixa os dados do Yahoo Finance automaticamente se não os encontrar localmente.

### Pré-requisitos

- Python 3.8+
- Acesso à internet (apenas na primeira execução)

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/risco-politico-b3.git
cd risco-politico-b3

# 2. Instale as dependências
pip install pandas numpy matplotlib seaborn statsmodels scipy yfinance openpyxl
```

### Execução

```bash
python analise_completa_final.py
```

Na primeira execução, o script irá:

1. Baixar cotações de ~650 empresas (2000–2023) via Yahoo Finance.
2. Salvar os dados em cache (`precos.csv`, `volumes.csv`) para reutilização.
3. Estimar os modelos Fama-French para cada ativo × eleição.
4. Gerar tabelas (`.xlsx`, `.csv`) e gráficos (`.png`) na pasta de saída.

> **Execuções seguintes** carregam do cache e pulam o download.

---

## 📂 Estrutura do Projeto

```
risco-politico-b3/
│
├── analise_completa_final.py               # Script principal (ETL + Modelagem + Viz)
├── resultados_analise_b3_com_tickers.xlsx  # Input: mapeamento de tickers e setores (B3)
├── nefin_factors.csv                       # Input: fatores de risco NEFIN-USP
│
├── output_ff3_bhar/                        # Output: resultados gerados
│   ├── resultados_ff3_bhar.xlsx            #   Tabelas completas (multi-abas)
│   ├── resultados_eleitoral.csv            #   CSV consolidado
│   ├── heatmap_eleitoral.png               #   Mapa de calor (anos eleitorais)
│   ├── heatmap_placebo.png                 #   Mapa de calor (teste placebo)
│   ├── timeline_bhar.png                   #   Evolução temporal do BHAR
│   ├── did_barras.png                      #   Diff-in-Diff (Regulados vs Não Regulados)
│   ├── conclusao_ranking_setorial.png      #   Ranking final por setor
│   ├── conclusao_risco_retorno.png         #   Scatter Risco × Retorno
│   ├── metodologia_ff3_bhar.txt            #   Descrição metodológica
│   ├── precos.csv                          #   Cache de preços (gerado automaticamente)
│   └── volumes.csv                         #   Cache de volumes (gerado automaticamente)
│
└── README.md
```

---

## 🛡️ Tratamento de Dados

Para garantir a integridade dos resultados, o código aplica filtros rigorosos:

| Filtro | Regra | Justificativa |
|:-------|:------|:--------------|
| **Filtro de Existência** | Invalida preços anteriores à data de registro (DT_REG) | Evita viés de sobrevivência |
| **Filtro de Liquidez** | Exige presença em ≥ 40% dos pregões da janela de estimação | Descarta ativos ilíquidos |
| **Winsorização** | Clip de BHARs individuais em [−100%, +200%] | Protege contra falhas do modelo |
| **Exclusão "Outros"** | Remove setor "Outros" do processamento | Holdings heterogêneas causavam distorções |
| **Mapeamento De-Para** | Atualiza tickers antigos (ex: VVAR3 → BHIA3) | Garante continuidade dos dados |

---

## 🧰 Stack Tecnológica

| Categoria | Ferramenta |
|:----------|:-----------|
| Linguagem | Python 3.8+ |
| Dados de Mercado | `yfinance` (Yahoo Finance) |
| Fatores de Risco | NEFIN-USP |
| Econometria | `statsmodels` (OLS, HAC) |
| Testes Estatísticos | `scipy.stats` (t-Student, Wilcoxon) |
| Manipulação | `pandas`, `numpy` |
| Visualização | `matplotlib`, `seaborn` |

---

## 📚 Referências

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13–39.
- NEFIN — Núcleo de Pesquisa em Economia Financeira da USP. Disponível em: [https://nefin.com.br/](https://nefin.com.br/)

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Sinta-se livre para utilizar os códigos para fins acadêmicos ou profissionais, desde que citada a fonte.

---

<p align="center">
  <b>Autor:</b> [Seu Nome]<br>
  <i>Pesquisa desenvolvida como parte de [TCC / Dissertação / Estudo Pessoal] em Finanças Quantitativas.</i>
</p>
