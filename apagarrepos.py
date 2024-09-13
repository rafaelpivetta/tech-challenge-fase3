import os
from huggingface_hub import delete_repo

def ler_urls_e_apagar(caminho_arquivo_urls, username_hf):
    # Verificar se o arquivo existe
    if not os.path.exists(caminho_arquivo_urls):
        print(f"O arquivo {caminho_arquivo_urls} não foi encontrado.")
        return
    
    # Ler os URLs do arquivo
    with open(caminho_arquivo_urls, 'r') as arquivo_urls:
        urls = arquivo_urls.readlines()
    
    for url in urls:
        url = url.strip()
        
        # Extrair o nome do repositório a partir do URL
        # Exemplo de URL: "https://huggingface.co/datasets/username/repository_name"
        repo_id = url.replace("https://huggingface.co/datasets/", "")
        
        # Apagar o dataset no Hugging Face
        try:
            delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"Dataset {repo_id} foi apagado com sucesso.")
        except Exception as e:
            print(f"Erro ao apagar o dataset {repo_id}: {str(e)}")

# Exemplo de uso
caminho_arquivo_urls = './datasets/urls.txt'  # Caminho para o arquivo gerado com os links dos datasets
username_hf = 'LuizfvFonseca'  # Seu nome de usuário no Hugging Face

ler_urls_e_apagar(caminho_arquivo_urls, username_hf)
