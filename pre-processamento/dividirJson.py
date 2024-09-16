import jsonlines
import os
import time  # Importar a função sleep
from huggingface_hub import HfApi, create_repo, upload_file

def criar_datasets_sequenciais(caminho_arquivo_json, n_linhas, username_hf):
    # Verificar se a pasta datasets existe, caso contrário, criá-la
    pasta_datasets = './datasets'
    if not os.path.exists(pasta_datasets):
        os.makedirs(pasta_datasets)
    
    # Carregar todos os dados do arquivo JSON original
    dados = []
    with jsonlines.open(caminho_arquivo_json, 'r') as arquivo:
        for item in arquivo:
            dados.append(item)
    
    total_linhas = len(dados)
    num_arquivos = (total_linhas + n_linhas - 1) // n_linhas  # Calcula o número de arquivos necessários

    nome_arquivo_original = os.path.basename(caminho_arquivo_json)  # Nome do arquivo original
    urls = []

    # Inicializar a API da Hugging Face
    api = HfApi()

    for i in range(num_arquivos):
        inicio = i * n_linhas
        fim = min((i + 1) * n_linhas, total_linhas)
        
        dados_parte = dados[inicio:fim]
        
        # Nomear o novo arquivo com um índice
        nome_novo_arquivo = f"{os.path.splitext(nome_arquivo_original)[0]}_{i+1}_de_{num_arquivos}.json"
        caminho_novo_arquivo = os.path.join(pasta_datasets, nome_novo_arquivo)
        
        # Escrever os dados no novo arquivo JSON
        with jsonlines.open(caminho_novo_arquivo, 'w') as novo_arquivo:
            novo_arquivo.write_all(dados_parte)

        # Nome do repositório de dataset no Hugging Face
        repo_name = f"{os.path.splitext(nome_arquivo_original)[0]}_parte_{i+1}_de_{num_arquivos}"
        repo_id = f"{username_hf}/{repo_name}"

        # Criar o dataset no Hugging Face (marcando como dataset)
        create_repo(repo_id, repo_type="dataset", private=False)

        # Fazer o upload do arquivo para o dataset no Hugging Face
        upload_file(
            path_or_fileobj=caminho_novo_arquivo,
            path_in_repo=nome_novo_arquivo,
            repo_id=repo_id,
            repo_type="dataset"  # Especificar que é um dataset
        )

        # Adicionar o URL do dataset à lista
        url = f"https://huggingface.co/datasets/{repo_id}"
        urls.append(url)
        print(f"Dataset {nome_novo_arquivo} criado e enviado para o Hugging Face: {url}")
        
        # Pausa de 3 segundos
        time.sleep(3)
    
    # Criar arquivo txt com os URLs
    caminho_urls_txt = os.path.join(pasta_datasets, 'urls.txt')
    with open(caminho_urls_txt, 'w') as arquivo_urls:
        for url in urls:
            arquivo_urls.write(url + '\n')
    
    print(f"Arquivo 'urls.txt' criado com os links dos datasets: {caminho_urls_txt}")

# Exemplo de uso
caminho_arquivo_json = './Treino100K.json'
n_linhas = 20000  # Número de linhas desejado
username_hf = 'LuizfvFonseca'  # Substitua pelo seu nome de usuário no Hugging Face
criar_datasets_sequenciais(caminho_arquivo_json, n_linhas, username_hf)
