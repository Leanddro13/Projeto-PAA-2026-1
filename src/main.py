def main():
    # 1. Receber a pergunta em linguagem natural do usuário [cite: 6]
    user_query = input("Faça uma pergunta sobre filmes: ")

    # 2. Realizar busca semântica no banco de dados [cite: 8]
    # Aqui o sistema deve comparar o uso de:
    # - Similaridade de Cosseno [cite: 9]
    # - Word2Vec Average [cite: 9]
    # - Sentence Embeddings [cite: 9]
    # - HNSW Search [cite: 9]
    
    # 3. Formatar a resposta utilizando um LLM local 
    # Exemplo de fluxo: passar o contexto recuperado na etapa 2 para o modelo
    
    # 4. Retornar a resposta formatada [cite: 6]
    print("A resposta formatada do LLM aparecerá aqui.")

if __name__ == "__main__":
    main()