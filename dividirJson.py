import jsonlines
import os

def criar_json_com_n_linhas(caminho_arquivo_json, n_linhas):
    # Carregar o arquivo JSON original usando jsonlines
    dados = []
    with jsonlines.open(caminho_arquivo_json, 'r') as arquivo:
        for item in arquivo:
            dados.append(item)
            if len(dados) >= n_linhas:
                break
    
    # Garantir que a quantidade de linhas não exceda o total de dados
    n_linhas = min(n_linhas, len(dados))
    
    # Nomear o novo arquivo com o sufixo nlinhas
    nome_arquivo_original = os.path.basename(caminho_arquivo_json)  # Nome do arquivo original
    pasta = os.path.dirname(caminho_arquivo_json)  # Pasta do arquivo original
    nome_novo_arquivo = f"{os.path.splitext(nome_arquivo_original)[0]}_{n_linhas/1000}K_linhas.json"
    caminho_novo_arquivo = os.path.join(pasta, nome_novo_arquivo)
    
    # Escrever os dados no novo arquivo JSON
    with jsonlines.open(caminho_novo_arquivo, 'w') as novo_arquivo:
        novo_arquivo.write_all(dados[:n_linhas])
    
    print(f"Novo arquivo criado: {caminho_novo_arquivo}")

# Exemplo de uso
caminho_arquivo_json = './Treino100K.json'
n_linhas = 20000  # Número de linhas desejado
criar_json_com_n_linhas(caminho_arquivo_json, n_linhas)
