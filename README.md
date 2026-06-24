# Projeto-PAA-2026-1

## Sistema de QA de Filmes - PAA 1/2026

[cite_start]Este projeto de disciplina implementa um sistema onde o usuário faz uma pergunta em linguagem natural e recebe uma resposta formatada[cite: 6]. 

## Base de Dados
[cite_start]O banco de dados utilizado é o "CMU Movie Summary Corpus"[cite: 7]. 

## Metodologias Implementadas
[cite_start]O sistema realiza buscas semânticas nas sinopses e avalia o desempenho e complexidade das seguintes abordagens[cite: 8, 9]:
- [cite_start]Similaridade de Cosseno [cite: 9]
- [cite_start]Word2Vec Average [cite: 9]
- [cite_start]Sentence Embeddings [cite: 9]
- [cite_start]HNSW Search [cite: 9]

[cite_start]A formatação final é realizada por um LLM local mais simples.

## Como Executar
1. Instale as dependências: `pip install -r requirements.txt`
2. Baixe o corpus e coloque na pasta `data/`
3. Execute o script principal: `python src/main.py`