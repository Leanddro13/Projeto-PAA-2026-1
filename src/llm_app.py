import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Variável global para manter a pipeline em cache e evitar recarregar na memória RAM
_llm_pipeline = None

def carregar_llm_local(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Carrega o modelo de linguagem local leve (LLM) na CPU e cria a pipeline de geração de texto.
    """
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline
        
    print(f"Inicializando LLM local '{model_name}' em CPU...")
    print("Nota: Na primeira execução, o download dos arquivos (~950MB) ocorrerá de forma automatizada.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Carregar modelo otimizado para CPU pura em float32 (formato nativo)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Definir a pipeline para geração textual autoregressiva
    _llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.6,
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print("LLM local carregado com sucesso na RAM e pronto para inferência!")
    return _llm_pipeline

def responder_pergunta_rag(pergunta, resultados_filmes, pipeline_llm):
    """
    Gera a resposta em linguagem natural com base nos metadados e sinopses dos filmes recuperados.
    Complexidade: Depende do tamanho do prompt (S) e da quantidade de tokens gerados (T),
    sendo o custo do self-attention O(S^2) e da decodificação autorregressiva O(T * S).
    """
    # Construir o contexto para o prompt de RAG
    contexto = ""
    for idx, (_, row) in enumerate(resultados_filmes.iterrows(), 1):
        contexto += f"Filme {idx}: {row['movie_name']}\n"
        # Truncar sinopse a 500 caracteres para evitar estourar o limite de tokens/tempo da CPU
        sinopse = row['plot_limpo'][:500] + "..." if len(row['plot_limpo']) > 500 else row['plot_limpo']
        contexto += f"Sinopse: {sinopse}\n\n"
        
    # Instrução estruturada para guiar o LLM pequeno (Qwen-0.5B)
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente especialista em cinema. Responda em português de forma clara e resumida (máximo 3 frases). "
                "Baseie-se APENAS nas sinopses dos filmes fornecidos como contexto."
            )
        },
        {
            "role": "user",
            "content": f"Contexto das sinopses de filmes recuperadas:\n{contexto}\n\nPergunta: {pergunta}"
        }
    ]
    
    # Formatar mensagem usando o chat template do modelo
    prompt = pipeline_llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Gerar a resposta
    outputs = pipeline_llm(prompt)
    
    generated_text = outputs[0]["generated_text"]
    
    # Extrair apenas a nova resposta gerada pelo modelo (após a indicação de assistente no template)
    # O Qwen usa `<|im_start|>assistant\n` para demarcar a resposta do assistente
    marcador_resposta = "<|im_start|>assistant\n"
    if marcador_resposta in generated_text:
        resposta = generated_text.split(marcador_resposta)[-1].replace("<|im_end|>", "").strip()
    else:
        # Fallback simples caso não use o chat template corretamente
        resposta = generated_text[len(prompt):].strip()
        
    return resposta

if __name__ == "__main__":
    # Teste unitário independente para a Pessoa 4 (Orquestrador do LLM Local)
    # Permite rodar o código simulando sinopses na mão antes da integração final
    import pandas as pd
    
    print("=" * 60)
    print("TESTE UNITÁRIO AUTO-SUFICIENTE - LLM LOCAL (PESSOA 4)")
    print("=" * 60)
    
    # 1. Simular os dados que a Pessoa 2 ou 3 entregariam
    dados_simulados = [
        {
            "movie_name": "Inception (A Origem)",
            "plot_limpo": "um ladrao que rouba segredos corporativos por meio do uso de tecnologia de compartilhamento de sonhos tem a tarefa de plantar uma ideia na mente de um c.e.o."
        },
        {
            "movie_name": "Interstellar (Interestelar)",
            "plot_limpo": "uma equipe de exploradores viaja atraves de um buraco de minhoca no espaco na tentativa de garantir a sobrevivencia da humanidade."
        }
    ]
    df_contexto = pd.DataFrame(dados_simulados)
    
    # 2. Inicializar o modelo na CPU
    try:
        pipeline = carregar_llm_local()
    except Exception as e:
        print(f"Erro ao carregar o LLM local: {e}")
        print("Certifique-se de que os pacotes torch, transformers e accelerate estão instalados no venv.")
        exit(1)
        
    # 3. Testar a geração da resposta
    pergunta_usuario = "Qual filme conta a história de uma viagem no espaço por um buraco de minhoca?"
    print(f"\n[Pergunta do Usuário]: '{pergunta_usuario}'")
    
    print("\nExecutando Prompt RAG e inferência...")
    resposta = responder_pergunta_rag(pergunta_usuario, df_contexto, pipeline)
    
    print("\n" + "-" * 60)
    print("[Resposta gerada pelo LLM]:")
    print("-" * 60)
    print(resposta)
    print("-" * 60)
