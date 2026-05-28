# Prevendo o Sucesso de Videogames a partir de Texto Pré-Lançamento

Classificação binária do sucesso comercial de videogames usando posts do Reddit e comentários de trailers do YouTube coletados **antes da data de lançamento do jogo**.

**Relatório completo:** [`reports/report.md`](reports/report.md)

---

## Visão geral

Um classificador treinado apenas no que a comunidade diz antes de um jogo sair consegue prever se ele será bem avaliado e bem-vendido? Este projeto constrói esse classificador usando o discurso pré-lançamento de comunidades do Reddit e seções de comentários de trailers no YouTube, comparando um baseline TF-IDF + Regressão Logística contra DistilBERT com pesos congelados e DistilBERT fine-tuned.

Dois targets de predição são explorados separadamente:
- **Avaliação** — se o jogo acaba bem avaliado na Steam (`rating_ratio ≥ 0,80` = sucesso)
- **Vendas** — se ultrapassa 150 mil donos estimados na Steam

Todo texto usado para treinamento e avaliação foi postado **antes da data de lançamento na Steam**, garantindo uma restrição de não-leakage que torna o modelo aplicável como sinal real pré-lançamento.

---

## Resultados

| Target | Modelo | F1-macro |
|---|---|---|
| Avaliação | TF-IDF + LR (baseline) | **0,899** |
| Avaliação | DistilBERT frozen | 0,699 |
| Avaliação | DistilBERT fine-tuned | 0,880 |
| Vendas | TF-IDF + LR (baseline) | **0,728** |
| Vendas | DistilBERT frozen | 0,482 |
| Vendas | DistilBERT fine-tuned | 0,685 |

Achado principal: TF-IDF supera DistilBERT fine-tuned no target de avaliação (0,899 vs. 0,880). Posts pré-lançamento têm mediana de aproximadamente 73 caracteres — frases curtas e densas em palavras-chave onde representações bag-of-words se igualam ou superam embeddings contextuais.

---

## Estrutura do projeto

```
.
├── data/
│   ├── raw/
│   │   └── games.csv                        # 145 jogos selecionados com rótulos
│   └── processed/
│       ├── dataset_rating_granular.csv
│       ├── dataset_rating_aggregated.csv
│       ├── dataset_sales_granular.csv
│       └── dataset_sales_aggregated.csv
├── notebooks/
│   ├── 01_select_games.ipynb                # Seleção dos jogos do dataset HuggingFace
│   ├── 02_nlp_pipeline_colab.ipynb          # Pipeline limpo para re-execução no Colab
│   └── 02_nlp_pipeline_colab_executed.ipynb # Executado com todos os outputs
├── src/
│   ├── collect/
│   │   ├── reddit.py                        # Coleta do Reddit via PullPush.io
│   │   ├── youtube.py                       # Comentários de trailers via YouTube Data API v3
│   │   └── run_collection.py               # Orquestra os dois coletores
│   └── dataset/
│       └── build.py                         # Constrói os datasets processados a partir dos CSVs brutos
├── reports/
│   ├── report.md                            # Relatório completo do projeto
│   └── figures/                             # Gráficos de EDA e curvas de aprendizado
├── pyproject.toml
└── uv.lock
```

Os arquivos de texto bruto (`data/raw/reddit/`, `data/raw/youtube/`) não estão incluídos no repositório. Veja [Coleta de dados](#coleta-de-dados) para instruções de reprodução.

---

## Configuração

Este projeto usa [uv](https://docs.astral.sh/uv/) para gerenciamento de dependências.

```bash
# Instalar uv (se ainda não instalado)
pip install uv

# Instalar todas as dependências
uv sync

# Executar qualquer script
uv run python src/collect/reddit.py
```

**Versão Python:** 3.12+

---

## Reproduzindo o dataset

### 1. Lista de jogos

[`data/raw/games.csv`](data/raw/games.csv) já está incluído no repositório. Contém os 145 jogos selecionados com `appID`, `release_date`, `label_rating`, `label_sales`, `tier` e demais metadados da Steam.

| Dimensão | Valores |
|---|---|
| Total de jogos | 145 |
| Avaliação: sucesso / fracasso | 73 / 72 |
| Vendas: alto / baixo | 93 / 41 |
| Tiers | indie: 83, AA: 41, AAA: 21 |
| Janela temporal | 2020 – 2024 |

Para regenerar a partir do dataset HuggingFace, execute `notebooks/01_select_games.ipynb`.

### 2. Coleta do Reddit

```bash
uv run python src/collect/reddit.py
```

Usa a API [PullPush.io](https://pullpush.io) (sem autenticação). Para cada jogo em `games.csv`, busca posts mencionando o título do jogo em subreddits como `r/Games`, `r/gaming` e o subreddit próprio do jogo, filtrados para a janela de 90 dias antes de `release_date`. Saída: `data/raw/reddit/{appid}_reddit.csv`.

**Observação:** a cobertura do PullPush.io é melhor para o período 2020–2023. Jogos mais recentes podem ter dados limitados.

### 3. Coleta do YouTube

Requer uma chave da YouTube Data API v3. Crie um arquivo `.env` na raiz do projeto:

```
YOUTUBE_API_KEY=sua_chave_aqui
```

Para obter a chave: Google Cloud Console → ativar YouTube Data API v3 → Credenciais → Criar chave de API. A cota gratuita é de 10.000 unidades por dia.

```bash
uv run python src/collect/youtube.py
```

Busca trailers e teasers oficiais de cada jogo publicados antes de `release_date` e coleta todos os comentários postados antes dessa data. Saída: `data/raw/youtube/{appid}_youtube.csv`.

### 4. Construção dos datasets processados

```bash
uv run python src/dataset/build.py
```

Junta os CSVs do Reddit e YouTube com os rótulos por jogo, aplica detecção de idioma (apenas inglês), deduplica e gera os quatro CSVs em `data/processed/`.

---

## Executando o pipeline NLP

O notebook principal é `notebooks/02_nlp_pipeline_colab.ipynb`, projetado para rodar no **Google Colab com GPU T4** (o fine-tuning do DistilBERT leva aproximadamente 30 minutos por target em CPU; com GPU, cerca de 5 minutos).

Passos:
1. Fazer upload do notebook para o [Google Colab](https://colab.research.google.com)
2. Fazer upload da pasta `data/processed/` para a sessão do Colab
3. Configurar o runtime para GPU (Runtime → Alterar tipo de runtime → GPU T4)
4. Executar todas as células

A versão executada com todos os outputs está disponível em `notebooks/02_nlp_pipeline_colab_executed.ipynb`.

---

## Fontes de dados

- **Metadata e rótulos dos jogos:** [FronkonGames/steam-games-dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) (HuggingFace, CC-BY-4.0, ~124k jogos da Steam)
- **Posts do Reddit:** [PullPush.io](https://pullpush.io) — arquivo público da API do Reddit
- **Comentários do YouTube:** [YouTube Data API v3](https://developers.google.com/youtube/v3)

---

## Observações sobre disponibilidade de dados

Dados confiáveis de vendas de videogames são estruturalmente difíceis de obter. A Valve não publica números de vendas da Steam; a coluna `estimated_owners` no dataset é uma aproximação derivada de amostragem de perfis públicos (metodologia SteamSpy). Dados de vendas de console (PlayStation, Xbox, Nintendo Switch) não estão disponíveis publicamente por título, de modo que o target de vendas deste projeto reflete desempenho na Steam especificamente. Para títulos AAA com forte presença em console, isso pode subestimar ou representar de forma incompleta o desempenho comercial total.
