# Trilha Python DIO — estudos de Giordani

Repositório de estudos baseado no [material original da Digital Innovation One](https://github.com/digitalinnovationone/trilha-python-dio), com o histórico e os créditos preservados pelo fork.

## Projeto de visão: reconhecimento facial com TensorFlow

[**MTCNN + CNN: detecção e classificação de múltiplas faces**](projetos/reconhecimento-facial-tensorflow/README.md)

Piloto com oito identidades da base Olivetti, detector pré-treinado, alinhamento pelos olhos, classificação em lote e avaliação da rejeição de faces não cadastradas. Inclui notebook Colab, cadastro próprio e evidências com acertos e falhas documentados.

## Projeto de detecção: YOLO e duas classes adicionais

[**YOLOv3-tiny: base rotulada, transferência COCO80 → 82 e treinamento piloto**](projetos/yolo-duas-classes/README.md)

Base sintética própria, conversor LabelMe XML/JSON, transplante de pesos, treino real em CPU e avaliação em imagens reservadas. Inclui notebook Colab e evidências com limites do experimento documentados.

## Projeto de avaliação: métricas de classificação

[**Acurácia, recall, especificidade, precisão e F1 — fórmulas, notebook e resultados**](projetos/metricas-avaliacao-classificacao/README.md)

Cálculos explícitos a partir de uma matriz binária fictícia, com tratamento de denominadores zero, onze testes e relatórios JSON/Markdown. Não depende de bibliotecas de aprendizado de máquina.

## Projeto de imagens: cinza e binarização

[**Redução de dimensionalidade em imagens — código, notebook e resultados**](projetos/reducao-dimensionalidade-imagens/README.md)

Conversão explícita por pixel de RGB para cinza (0–255) e binária (0 ou 255), com limiar configurável, comparação visual, oito testes e evidências de execução local.

## Projeto de Deep Learning: Transfer Learning

[**Gatos e cachorros com MobileNetV2 — notebook, documentação e evidências**](projetos/transfer-learning-gatos-cachorros/README.md)

Aplicação de Transfer Learning com extração de características e fine-tuning, usando TensorFlow/Keras. O projeto está em uma pasta independente, com execução pelo Colab e registro do escopo das validações. Os módulos da trilha e o sistema bancário continuam disponíveis.

## Projeto documentado: sistema bancário em Python

O projeto de console permite depósitos, saques e consulta de extrato. A versão com funções também permite cadastrar usuários, criar contas e listar contas.

| Versão | Código |
| --- | --- |
| Fundamentos: depósitos, saques e extrato | [desafio.py](00%20-%20Fundamentos/desafio.py) |
| Funções, usuários e contas | [desafio.py](01%20-%20Estrutura%20de%20dados/desafio.py) |

## Como executar

Instale Python 3.10 ou superior e baixe ou clone este repositório:

```bash
git clone https://github.com/1giordani/trilha-python-dio.git
cd trilha-python-dio
python "01 - Estrutura de dados/desafio.py"
```

No Windows, se necessário, substitua `python` por `py`.
Essas duas versões de console usam somente a biblioteca padrão.

### Opções do menu

- `d`: depositar um valor positivo.
- `s`: sacar, respeitando saldo, limite de R$ 500 por saque e até três saques por execução.
- `e`: consultar extrato e saldo.
- `nu`: cadastrar usuário.
- `nc`: criar conta para usuário cadastrado.
- `lc`: listar contas.
- `q`: encerrar.

Use ponto para casas decimais, por exemplo `100.50`.

## Melhoria incluída neste fork

Na versão com funções, a função `sacar` incrementava a contagem apenas localmente. O programa principal não recebia o novo contador, permitindo saques além do limite.

A correção retorna `numero_saques` junto com saldo e extrato e atualiza os três valores no programa principal. Também foi acrescentada a proteção `if __name__ == "__main__"` para permitir importar as funções nos testes sem abrir o menu.

## Verificação

```bash
python -m unittest test_sistema_bancario -v
```

Quatro testes executados com sucesso:

1. Bloqueio do quarto saque, preservando saldo, extrato e contador.
2. Rejeição por saldo insuficiente.
3. Rejeição de valor superior ao limite por saque.
4. Rejeição de zero e valor negativo.

Exemplo manual: deposite 1000, saque 100 três vezes e tente sacar mais 100. O quarto saque deve ser recusado e o extrato deve mostrar saldo de R$ 700,00.

## Escopo e limitações

É uma simulação educacional. Os dados ficam na memória e são perdidos ao encerrar. O limite de três saques é por execução; essa versão não controla a mudança de dia. O saldo e o extrato são globais na versão com funções, sem movimentação individual por conta cadastrada. Entradas numéricas devem ser válidas; o código original utiliza `float`.

Os demais módulos do material original continuam disponíveis para estudo. Eles têm requisitos próprios e não estão cobertos pelos quatro testes acima.

## Referência e autoria

- Material e implementação de base: Digital Innovation One e colaboradores do repositório original.
- Organização deste fork: Giordani Martins Silva.
- Documentação, correção e testes preparados com auxílio de IA.

O repositório original contém vários desafios. O texto genérico de entrega fornecido não identifica a atividade específica da plataforma; este README documenta o sistema bancário encontrado nos módulos iniciais.
