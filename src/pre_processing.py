import pandas as pd
import re
import os

def preprocessar_dados():
    print("Iniciando pré-processamento...")
    
    # Caminhos apontando para a pasta data/
    caminho_plots = 'data/plot_summaries.txt'
    caminho_meta = 'data/movie.metadata.tsv'
    caminho_saida = 'data/dataset_limpo.json'
    
    # Garantir que a pasta data/ existe
    os.makedirs('data', exist_ok=True)
    
    print("Carregando arquivos do CMU Movie Summary Corpus...")
    plots = pd.read_csv(caminho_plots, sep='\t', names=['wikipedia_id', 'plot'])
    
    metadata_cols = ['wikipedia_id', 'freebase_id', 'movie_name', 'release_date',
                     'box_office', 'runtime', 'languages', 'countries', 'genres']
    metadata = pd.read_csv(caminho_meta, sep='\t', names=metadata_cols)
    
    print("Fazendo o Merge (juntando os filmes com as sinopses)...")
    df = pd.merge(plots, metadata[['wikipedia_id', 'movie_name', 'genres']], on='wikipedia_id', how='inner')
    
    print("Iniciando limpeza de texto...")
    def limpar_texto(texto):
        texto = str(texto).lower()
        texto = re.sub(r'[^a-z0-9\s.,]', '', texto)
        return re.sub(r'\s+', ' ', texto).strip()

    df['plot_limpo'] = df['plot'].apply(limpar_texto)
    df = df.dropna(subset=['plot_limpo', 'movie_name'])
    
    print("Salvando dataset_limpo.json...")
    df_final = df[['wikipedia_id', 'movie_name', 'genres', 'plot_limpo']]
    df_final.to_json(caminho_saida, orient="records", force_ascii=False)
    
    print(f"Sucesso! {len(df_final)} filmes foram processados e salvos em {caminho_saida}.")

if __name__ == "__main__":
    preprocessar_dados()