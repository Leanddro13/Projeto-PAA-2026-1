# Projeto-PAA-2026-1

## Sistema de QA de Filmes - PAA 1/2026

Este projeto de disciplina implementa um sistema onde o usuário faz uma pergunta em linguagem natural e recebe uma resposta formatada. 

## Base de Dados
O banco de dados utilizado é o "CMU Movie Summary Corpus". 

## Metodologias Implementadas
O sistema realiza buscas semânticas nas sinopses e avalia o desempenho e complexidade das seguintes abordagens:
- Similaridade de Cosseno
- Word2Vec Average
- Sentence Embeddings
- HNSW Search

A formatação final é realizada por um LLM local mais simples.

## Como Executar
1. Instale as dependências: `pip install -r requirements.txt`
2. Baixe o corpus e coloque na pasta `data/`
3. Execute o script principal: `python src/main.py`

## Integrantes



| Nome | 
| --- | 
| Leandro Souza da Silva | 
| Luan |
| Alexandre |
|Debora Goncalves|
|Lucas Santos|
|Rychard Ryan| 