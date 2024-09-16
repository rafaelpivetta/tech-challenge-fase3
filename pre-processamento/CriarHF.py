from huggingface_hub import login
from huggingface_hub import create_repo
from huggingface_hub import HfApi,HfFolder

TOKEN_Huggingface ='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' #read

TOKEN_Huggingface_write ='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' #write

login(TOKEN_Huggingface_write)

Nome_repositorio = 'Dados_Tech_Challenge_F3'
Tipo="dataset"
Privado=False

url=create_repo(repo_id=Nome_repositorio,repo_type=Tipo,private=Privado)
print(f"Repositório criado: {url}")

api = HfApi()

caminho_local = "./Treino100K_20.0K_linhas.json"

caminho_repo = 'LuizfvFonseca/Dados_Tech_Challenge_F3'

api.upload_file(
    path_or_fileobj=caminho_local,
    path_in_repo="Treino100K.json",
    repo_id=caminho_repo,
    repo_type="dataset"
)


readme_path = "./README.md"

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="readme.md",
    repo_id=caminho_repo,
    repo_type="dataset"
)