import pandas as pd

df = pd.read_json("hf://datasets/LuizfvFonseca/Dados_Tech_Challenge_F3/Treino100K.json", lines=True)
print(df.head())