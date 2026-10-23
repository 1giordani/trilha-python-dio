# Rotular uma base própria

O enunciado aponta o [LabelMe web do MIT](http://labelme.csail.mit.edu/Release3.0/). Ele trabalha com objetos nomeados e polígonos, com anotação XML. O **Labelme desktop** é outro projeto, com exportação JSON; não são o mesmo software nem o mesmo formato. O conversor deste repositório aceita os dois casos, mas não executa as interfaces nem faz upload de imagens.

## Fluxo com o LabelMe web solicitado
1. Escolha imagens próprias ou licenciadas, sem dados sensíveis. Siga as instruções do serviço para acessar uma coleção que possa anotar; disponibilidade e upload dependem da instalação usada.
2. Contorne cada objeto com um polígono. Nomeie exatamente `marcador_triangulo` ou `marcador_quadrado` para manter o exemplo; se mudar a taxonomia, atualize NEW_CLASSES e a lista de nomes de forma consistente.
3. Exporte/obtenha o XML de cada imagem. O script lê object/name, object/deleted e polygon/pt/x/y e obtém as dimensões do arquivo de imagem.
4. Converta a anotação:

```bash
python annotations.py minha_anotacao.xml minha_imagem.png minha_imagem.txt
```

Um polígono é convertido em sua caixa retangular envolvente; este projeto faz **detecção com bounding boxes**, não segmentação. Objetos com deleted=1 são ignorados. Coordenadas fora da imagem, caixas sem área e rótulos desconhecidos são rejeitados para revisão.

## Alternativa desktop
O [Labelme desktop](https://github.com/wkentaro/labelme) exporta JSON com shapes. Use retângulos ou polígonos e depois:

```bash
python annotations.py minha_anotacao.json minha_imagem.png minha_imagem.txt
```

O suporte JSON é uma facilidade adicional; não representa uso do serviço MIT. Linhas, pontos e círculos não são convertidos silenciosamente.

## Revisão e separação
- O arquivo YOLO contém uma linha por objeto: `class_id x_centro y_centro largura altura`, com coordenadas normalizadas.
- IDs: 0–79 reservados ao COCO; 80 e 81 são as duas classes adicionais. Os 80 IDs são os índices contíguos do coco.names do Darknet, **não** os category_id esparsos originais do COCO.
- A estrutura images/ e labels/ deve ter caminhos correspondentes e o mesmo nome-base do arquivo.
- Separe treino, validação e teste por imagem/cena/objeto de origem, evitando quadros muito semelhantes em grupos diferentes.
- Anote todos os objetos das classes treinadas em cada imagem. Rótulos incompletos podem ensinar a rede a tratar objetos não anotados como fundo.
- Faça uma visualização das caixas antes do treino; teste a unidade e o sentido das coordenadas.

## Estado desta entrega
A base demonstrativa foi gerada por generate_dataset.py: 120 imagens e seus polígonos XML, com conversão real para YOLO. **Não houve sessão de rotulagem manual no LabelMe.** O arquivo dataset-sintetico.zip contém a base reproduzível e rotulada automaticamente. Não é COCO nem uma base de fotografias. O COCO permanece como fonte dos pesos pré-treinados e dos nomes das 80 classes anteriores.
