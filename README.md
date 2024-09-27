# Tech Challenge 3

O Tech Challenge desta fase demandou a execução do fine-tuning de um foundation model utilizando o dataset "The AmazonTitles-1.3MM". O principal objetivo foi treinar um modelo capaz de fornecer respostas contextualizadas a partir de dados relacionados a produtos da Amazon. A solução exigia que o modelo fosse ajustado para lidar com esse fluxo de trabalho, garantindo respostas baseadas em contexto e fornecendo as fontes das informações. Optamos por treinar dois modelos e avaliar os resultados.

O trabalho foi dividido em três partes: (1) seleção e preparação do dataset, (2) fine-tuning utilizando o modelo Llama com Unsloth, e (3) fine-tuning com o gpt-4o-mini-2024-07-18. Para cada uma dessas partes, foram feitos ajustes específicos e adotadas estratégias para otimizar o treinamento, sempre levando em consideração o tamanho do dataset e as limitações computacionais.

## Parte 1: Seleção e Preparação do Dataset

Devido ao seu tamanho considerável foi decidido dividir o dataset em 70 partes menores para facilitar o processamento e o fine-tuning. Trabalhou-se com a primeira parte do dataset, que continha cerca de 20 mil linhas, representando uma amostra significativa do conjunto total de dados, sem comprometer os recursos computacionais.

### Etapas de Preparação:

- **Limpeza**: Foram removidos registros com entradas incompletas para garantir a integridade dos dados e embora tenha sido identificado possível duplicidade de entradas, optou-se por mantê-las, pois não invalidaram o treino.
- **Carregamento dos Dados**: Os datasets divididos foram carregados via Hugging Face no repo https://huggingface.co/LuizfvFonseca. Como o volume total era muito grande, 2 milhões de linhas, e com o fim didático do desafio, optou-se por carregar e processar alguns arquivos dentre as 70 partes disponíveis, com cada parte contendo 20 mil registros.
- **Tokenização**: Utilizou-se o tokenizer adequado ao modelo selecionado. Esse processo converteu as consultas e títulos em tokens, que são sequências numéricas compatíveis com os modelos de linguagem. O comprimento máximo de sequência foi definido para garantir que o modelo pudesse processar as informações sem sobrecarregar a memória, mantendo a eficiência do treinamento.

### Conclusão da Parte 1

O pré-processamento cuidadoso, a divisão do dataset em partes menores, e a tokenização foram etapas cruciais para garantir que o modelo pudesse lidar com o vasto volume de dados. Ao trabalhar com uma amostra de 20 mil linhas, foi possível otimizar a carga computacional e garantir que o treinamento fosse realizado de maneira eficiente. Essa preparação garantiu que o modelo estivesse pronto para o fine-tuning.

## Parte 2: Fine-Tuning Usando Llama com Unsloth

Na fase 2 do desafio, o foco foi o fine-tuning de um modelo LLaMA, especificamente utilizando o modelo quantizado unsloth/tinyllama-chat-bnb-4bit. Essa abordagem é significativa, pois a quantização reduz o tamanho do modelo e melhora sua eficiência computacional, possibilitando o treinamento e a inferência em ambientes com recursos limitados, sem sacrificar drasticamente a precisão.

### 1. Escolha do Modelo LLaMA

O modelo LLaMA (Large Language Model Meta AI) foi selecionado por sua arquitetura avançada, que é capaz de gerar respostas de alta qualidade e manter um bom desempenho em diversas tarefas de linguagem natural. A flexibilidade do LLaMA permite que ele seja ajustado para diferentes contextos e estilos de resposta, tornando-o ideal para aplicações que requerem um alto grau de precisão e relevância nas respostas.

### 2. Quantização com Unsloth

A quantização é uma técnica crucial que ajuda a reduzir o tamanho do modelo e a aumentar a velocidade de inferência, o que é especialmente útil em ambientes com recursos limitados. O modelo unsloth/tinyllama-chat-bnb-4bit utiliza uma representação de 4 bits, permitindo uma economia significativa de memória enquanto mantém uma performance comparável à de versões de maior capacidade. Isso é alcançado sem sacrificar a qualidade das respostas, o que é vital para aplicações práticas.

### 3. LoRA (Low-Rank Adaptation)

A técnica de LoRA é uma abordagem inovadora que permite o fine-tuning do modelo com um número reduzido de parâmetros. Em vez de atualizar todos os pesos do modelo durante o treinamento, o LoRA realiza as atualizações apenas em partes específicas (subconjuntos) do modelo, focando nas adaptações que têm maior impacto no desempenho. Isso não só torna o fine-tuning mais econômico em termos de recursos computacionais, mas também acelera o processo, permitindo que modelos de grande porte sejam ajustados rapidamente para tarefas específicas.

### 4. Implementação do Fine-Tuning

O processo de fine-tuning utilizando Unsloth e LoRA pode ser dividido nas seguintes etapas:

- **Configuração do Ambiente**: Primeiramente, é necessário configurar o ambiente com as bibliotecas relevantes, incluindo transformers, trl (Training Reinforcement Learning), e unsloth. Isso garante que todos os componentes estejam prontos para a execução.
- **Carregamento do Modelo e Tokenizer**: Carregar o modelo quantizado LLaMA e seu tokenizer associado é o primeiro passo prático. Essa etapa é crucial para garantir que o modelo possa interpretar os dados de entrada corretamente.
- **Definição dos Parâmetros de Treinamento**: É essencial definir parâmetros como taxa de aprendizado, número de épocas e tamanho do lote. Esses parâmetros impactam diretamente a eficácia do fine-tuning. Com o uso de LoRA, parâmetros como o rank r, lora_alpha e lora_dropout devem ser otimizados para obter os melhores resultados.

  - **Rank r**: Define a dimensionalidade do espaço de baixa-rank em que as adaptações são realizadas. Um valor maior para r permite uma representação mais rica e pode melhorar a capacidade do modelo de capturar nuances nos dados, mas também aumenta o número de parâmetros a serem ajustados. Valores típicos sugeridos incluem 8, 16, 32, 64 e 128.
  - **lora_alpha**: Este parâmetro controla a escala das atualizações realizadas pelo LoRA. Um valor maior de lora_alpha significa que as adaptações feitas às partes específicas do modelo têm um impacto mais significativo. Portanto, ele atua como um fator de ampliação que pode aumentar a influência das modificações feitas durante o fine-tuning.
  - **lora_dropout**: Representa a taxa de dropout aplicada às camadas de LoRA. O dropout é uma técnica regularizadora que ajuda a prevenir o overfitting durante o treinamento, desligando aleatoriamente uma fração dos neurônios durante cada iteração. Definir lora_dropout como um valor diferente de zero pode melhorar a generalização do modelo, enquanto um valor de zero significa que não há dropout aplicado, potencialmente levando a um treinamento mais rápido, mas com risco de overfitting.

- **Execução do Treinamento**: O treinamento é então executado, utilizando técnicas como gradient accumulation para permitir tamanhos de lote maiores sem exceder a memória disponível.
- **Avaliação e Ajustes Finais**: Após o fine-tuning, é fundamental avaliar o desempenho do modelo em um conjunto de dados de validação. Ajustes nos parâmetros podem ser feitos com base nos resultados obtidos, buscando sempre a melhoria da qualidade das respostas geradas.

### Conclusão da Parte 2

A combinação de LLaMA, Unsloth e LoRA oferece uma solução poderosa para o fine-tuning de modelos de linguagem. O uso de quantização reduz significativamente o uso de memória, enquanto LoRA proporciona uma abordagem eficiente para adaptações específicas. Essa metodologia permite que desenvolvedores e pesquisadores ajustem modelos complexos de maneira econômica e eficaz, abrindo portas para inovações em aplicações de processamento de linguagem natural.

[Fine-tuning com Unsloth](https://github.com/rafaelpivetta/tech-challenge-fase3/blob/2bd26635e5e162a61acceaac6371c091c5aeb796/fine-tuning/TinyLlama/Tech3_Finetuning_com_Unsloth.ipynb)


## Parte 3: Fine-Tuning utilizando Open AI

A terceira fase envolveu o fine-tuning de um modelo disponibilizado pela Open AI. O modelo escolhido foi o gpt-4o-mini-2024-07-18, principalmente por estar em um período de custo reduzido para utilização.

### Configuração e Parâmetros do Fine-Tuning:

- **Modelo Base**: O modelo escolhido foi o gpt-4.0-mini-2024-07-18, treinado para gerar respostas com base nos dados da Amazon.

- **Pré-processamento dos Dados**:

  - Limpeza e normalização das entradas, removendo caracteres especiais e formatando o texto para que ficasse no formato esperado pelo modelo GPT-4.0.
  - Cada linha foi convertida em um formato JSON compatível com o modelo, extraindo o título e a descrição do produto. Esse processo garantiu que o modelo tivesse dados padronizados para aprendizado.
  - O arquivo de entrada foi processado e convertido para o formato esperado pelo GPT-4.0, utilizando uma estrutura de mensagens que envolvia três tipos de interações:
    - **"system"**: Dava as instruções gerais para o modelo, como "Act as sales representative".
    - **"user"**: Fornecia o título do produto, representando a pergunta feita pelo usuário.
    - **"assistant"**: A resposta do modelo, gerando descrições detalhadas com base no contexto do título e da consulta fornecida.

  - O arquivo final foi salvo em formato JSON, permitindo que o GPT-4.0 utilizasse esses dados para treinamento e futuras inferências.

- **Criação do job de fine tuning**: Realizou-se o fine tuning com o primeiro set de 20 mil linhas, obtendo-se uma taxa de custo-benefício satisfatória, dada a utilização de cupons promocionais. O set de dados foi dividido em 2.000 entradas por job de fine tuning, para avaliar melhor a performance e possível overfitting. O resultado esperado do treinamento é a obtenção de um modelo que consiga sugerir descrições mais assertivas e de alta relevância para cada título fornecido.

### Conclusão da Parte 3

Ao final deste treinamento, o modelo gpt-4o-mini-2024-07-18 foi capaz de gerar descrições detalhadas e relevantes a partir de títulos de produtos. A utilização dos cupons de desconto permitiu que o fine-tuning fosse realizado de maneira econômica. A modularização dos jobs de treinamento foi fundamental para garantir a análise detalhada do desempenho e ajustes, conforme necessário.

[Fine-tuning com Open AI](https://github.com/rafaelpivetta/tech-challenge-fase3/blob/83dac8e97fc66ee88ac385d8e72fd67666de0b1a/fine-tuning/Open%20AI/open-ai-fine-tuning.ipynb)

## Conclusão Final

Este Tech Challenge destacou a importância da escolha cuidadosa do modelo e das técnicas de otimização para o sucesso do fine-tuning. A utilização do LLaMA com Unsloth e LoRA permitiu otimizar recursos e ainda obter um modelo altamente eficiente. Já o modelo GPT-4.0 mini demonstrou seu valor em fornecer respostas detalhadas e assertivas com um custo-benefício interessante.

Em futuras iterações, explorar a integração desses modelos para obter um sistema de question-answering ainda mais robusto seria o próximo passo, especialmente utilizando os dados da Amazon como base.

## Links Úteis

- **Dataset**: [The AmazonTitles-1.3MM](https://huggingface.co/LuizfvFonseca)
- **Modelo**: [rafaelpivetta/tinyllama-chat-bnb-4bit-g19](https://huggingface.co/rafaelpivetta/tinyllama-chat-bnb-4bit-g19)
- **Modelos Usados**:
  - [LLaMA](https://huggingface.co/unsloth/tinyllama-chat-bnb-4bit)
  - [gpt-4o-mini-2024-07-18](https://platform.openai.com/docs/models/gpt-4o-mini)
  - [Unsloth](https://github.com/unslothai/unsloth)
