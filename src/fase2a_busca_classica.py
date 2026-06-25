import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pre_processing

class BuscaVetorialClassica:
    def __init__(self, dataset_path, vector_size=100, window=5, min_count=1):
        """
        Inicializa a classe carregando os dados e treinando o modelo Word2Vec.
        """
        self.dataset_path = dataset_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        
        self.movies = []
        self.model = None
        self.movie_vectors = []
        
    def load_data(self):
        """
        Carrega o dataset limpo (JSON gerado pela Pessoa 1).
        """
        print(f"Carregando dados de {self.dataset_path}...")
        try:
            if not os.path.exists(self.dataset_path):
                print(f"Arquivo {self.dataset_path} não encontrado. Executando pré-processamento...")
                pre_processing.preprocessar_dados()
                
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.movies = json.load(f)
            print(f"{len(self.movies)} filmes carregados com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")
            self.movies = []

    def preprocess_text(self, text):
        """
        Função simples para limpar e tokenizar o texto (transformar em minúsculas e extrair palavras).
        """
        if not isinstance(text, str):
            return []
        # Remove pontuação e converte para minúsculas
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def train_word2vec(self):
        """
        Treina o modelo Word2Vec com as sinopses dos filmes.
        """
        if not self.movies:
            print("Nenhum dado para treinar. Carregue os dados primeiro.")
            return

        print("Tokenizando sinopses...")
        tokenized_synopses = [self.preprocess_text(movie.get('plot_limpo', '')) for movie in self.movies]
        
        print("Treinando modelo Word2Vec...")
        # Treinando o modelo Word2Vec
        self.model = Word2Vec(
            sentences=tokenized_synopses, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=4
        )
        print("Modelo treinado com sucesso!")

    def _get_average_vector(self, tokens):
        """
        Calcula o vetor médio de uma lista de tokens (palavras).
        """
        if not tokens:
            return np.zeros(self.vector_size)
            
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
            
        # Retorna a média dos vetores das palavras
        return np.mean(vectors, axis=0)

    def compute_movie_vectors(self):
        """
        Calcula o vetor médio para todas as sinopses do dataset.
        """
        if not self.model:
            print("Treine o modelo Word2Vec primeiro.")
            return

        print("Calculando vetores médios para todos os filmes...")
        self.movie_vectors = []
        for movie in self.movies:
            tokens = self.preprocess_text(movie.get('plot_limpo', ''))
            avg_vector = self._get_average_vector(tokens)
            self.movie_vectors.append(avg_vector)
            
        # Converte para uma matriz numpy para facilitar o cálculo de cosseno
        self.movie_vectors = np.array(self.movie_vectors)
        print("Vetores calculados com sucesso.")

    def search(self, query, top_k=5):
        """
        Recebe uma pergunta em linguagem natural e retorna os top_k filmes mais similares.
        """
        if len(self.movie_vectors) == 0:
            print("Vetores de filmes não calculados. Execute compute_movie_vectors() primeiro.")
            return []

        # 1. Processa a pergunta
        query_tokens = self.preprocess_text(query)
        query_vector = self._get_average_vector(query_tokens)
        
        # O vetor da query precisa estar no formato correto para o sklearn (1, vector_size)
        query_vector = query_vector.reshape(1, -1)
        
        # 2. Calcula a similaridade de cosseno entre a query e todos os filmes
        # cosine_similarity retorna uma matriz, pegamos a primeira (e única) linha
        similarities = cosine_similarity(query_vector, self.movie_vectors)[0]
        
        # 3. Encontra os índices dos maiores valores de similaridade
        # argsort retorna ordem crescente, então pegamos os últimos top_k e invertemos a ordem
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 4. Retorna os filmes correspondentes
        results = []
        for idx in top_indices:
            movie = self.movies[idx]
            results.append({
                'titulo': movie.get('movie_name', 'Desconhecido'),
                'similaridade': round(float(similarities[idx]), 4),
                'sinopse_trecho': movie.get('plot_limpo', '')[:200] + '...' # Retorna só um pedaço para visualização
            })
            
        return results

# Exemplo de uso isolado do script
if __name__ == "__main__":
    # Caminho do dataset limpo gerado pela Pessoa 1
    # Substitua pelo caminho real quando for rodar
    caminho_dataset = "data/dataset_limpo.json"
    
    # 1. Instanciar a classe
    buscador = BuscaVetorialClassica(dataset_path=caminho_dataset)
    
    # 2. Carregar dados
    buscador.load_data()
    
    # 3. Treinar Word2Vec
    buscador.train_word2vec()
    
    # 4. Calcular vetores médios
    buscador.compute_movie_vectors()
    
    # 5. Fazer uma busca
    pergunta = "I want a sci-fi movie about space and survival"
    print(f"\nBuscando por: '{pergunta}'")
    
    resultados = buscador.search(pergunta, top_k=2)
    
    print("\nResultados encontrados:")
    for i, r in enumerate(resultados):
        print(f"{i+1}. {r['titulo']} (Score de Similaridade: {r['similaridade']})")
        print(f"   Sinopse: {r['sinopse_trecho']}\n")
