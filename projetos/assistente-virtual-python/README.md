# Assistente virtual em Python — DIO

Assistente em português que transforma fala em texto, identifica intenções por regras de Processamento de Linguagem Natural e responde com áudio. Abre pesquisas na Wikipédia, o YouTube e uma busca de farmácias no Google Maps. Inclui modo de texto, entrada por arquivo de áudio e simulação de comandos.

Projeto educacional de Giordani Martins Silva, preparado com auxílio de IA a partir dos requisitos do desafio. Implementação própria; as referências da aula estão ao final. Não utiliza treinamento de um novo modelo nem um LLM: o reconhecimento de fala usa um serviço existente e a compreensão dos comandos usa regras explícitas.

## Requisitos atendidos

| Requisito | Implementação |
| --- | --- |
| Texto → áudio | `sintetizar`: gTTS em português, saída MP3; `reproduzir`: pygame |
| Fala → texto | `reconhecer`: SpeechRecognition + Google Speech, idioma `pt-BR` |
| Ações por voz | `interpretar` e `executar`: Wikipédia, YouTube e mapa |
| PLN | Normalização de acentos, identificação de intenção e extração de assunto/local |
| Melhoria | Modo texto, URLs codificadas, testes, tratamento de erros e limite de interações |

## Instalação local

Use Python 3.10 a 3.12 e um navegador. No terminal, entre nesta pasta:

```bash
git clone https://github.com/1giordani/trilha-python-dio.git
cd trilha-python-dio/projetos/assistente-virtual-python
python -m venv .venv
```

Ative o ambiente no Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Ou no Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Para usar **microfone**, instale também:

```bash
python -m pip install -r requirements-microfone.txt
```

PyAudio requer PortAudio. No Ubuntu/Debian, se a compilação falhar, instale `portaudio19-dev` e `python3-dev` com o gerenciador de pacotes; no macOS, instale `portaudio` pelo Homebrew. No Windows, use uma versão de Python com wheel disponível para PyAudio. Autorize o microfone nas configurações do sistema. A reprodução precisa de dispositivo de saída de áudio.

## Começar sem microfone

O seguinte comando funciona só com a biblioteca padrão, sem rede e sem abrir páginas:

```bash
python assistente.py --texto --sem-audio --dry-run
```

Digite um comando por linha e `sair` para encerrar. Para abrir realmente as páginas, retire `--dry-run`. Para ouvir as respostas, retire também `--sem-audio` e instale as dependências.

Exemplo de interação completa pelo teclado, com áudio e navegador:

```bash
python assistente.py --texto
```

## Usar por voz

```bash
python assistente.py
```

O programa escuta uma frase e responde antes de iniciar a próxima interação. Fale um destes comandos:

| Comando | Resultado |
| --- | --- |
| Abrir YouTube | Abre a página inicial |
| Pesquisar no YouTube sobre Python | Abre os resultados da busca |
| Pesquisar na Wikipédia sobre inteligência artificial | Abre a busca na Wikipédia em português |
| Encontrar farmácia próxima | Abre o Maps com busca perto de mim |
| Encontrar farmácias em Recife | Usa a localidade informada |
| Que horas são? | Informa o horário local do computador |
| Sair | Encerra |

Termos de pesquisa devem vir **depois** de YouTube/Wikipédia. Por exemplo, prefira “Wikipédia sobre Python” a “Python na Wikipédia”. Variações fora das regras podem exibir a ajuda. O sistema não compreende conversa livre.

Você pode configurar a localidade sem alterar o código:

```bash
python assistente.py --localizacao "Centro, Curitiba" --turnos 5
```

O mapa usa a localização autorizada no navegador ou a cidade fornecida. O projeto **não calcula a farmácia mais próxima**, não verifica estabelecimentos abertos e não obtém coordenadas por conta própria: apresenta uma pesquisa para o usuário conferir distância e horário.

## Comando único e arquivo de áudio

```bash
python assistente.py --comando "Abrir YouTube" --sem-audio --dry-run
python assistente.py --audio minha_fala.wav --sem-audio --dry-run
```

O modo arquivo aceita WAV PCM, AIFF e FLAC nativo compatíveis com SpeechRecognition, lendo até 15 segundos. MP3 não é entrada direta; converta para WAV antes. Um arquivo de voz pode conter dados pessoais: use somente gravações suas ou autorizadas.

`--dry-run` evita abrir páginas; **não desativa os serviços de voz**. Para testar sem rede, use texto com `--sem-audio`. `--dispositivo N` seleciona o índice do microfone. Consulte `python assistente.py --help`.

## Google Colab

Abra [assistente_colab.ipynb](assistente_colab.ipynb) no Colab. Ele permite testar os comandos, reproduzir TTS e enviar um arquivo de voz. O microfone com PyAudio pertence ao computador onde Python está rodando; o notebook **não tenta acessar diretamente o microfone do seu computador pelo kernel remoto**. No Colab os destinos aparecem como links clicáveis.

## Fluxo e decisões

1. Entrada de texto, microfone ou arquivo.
2. Transcrição da fala, quando aplicável.
3. Normalização e interpretação do comando.
4. Construção de URL com `urlencode`, sem concatenar parâmetros sem escape.
5. Abertura do navegador, ou apenas exibição na simulação.
6. Resposta textual e síntese/reprodução opcional.

A execução termina após dez interações por padrão (configurável de 1 a 100), `sair`, EOF ou Ctrl+C. O microfone espera até cinco segundos pelo início da fala e grava frases de até dez segundos. Três erros consecutivos de captura/reconhecimento encerram o programa com orientação para usar texto. Requisições têm timeout. Erros de áudio preservam a resposta textual. Nenhum comando falado é executado como código ou comando de sistema.

## Privacidade e dependências externas

O reconhecimento Google envia o áudio ao serviço externo; gTTS envia o texto para gerar a voz e depende de internet. Não use para segredos ou conteúdo sensível. O arquivo `saidas/resposta.mp3` é substituído a cada resposta bem-sucedida e fica fora do versionamento. O programa não mantém histórico de gravações nem usa escuta em segundo plano. As URLs de busca são recebidas pelos respectivos sites.

## Validação e limites

```bash
python -m unittest -v
```

**15 testes passaram**, cobrindo intenções, codificação de parâmetros, localização, saída, falha do navegador, recuperação de erro de áudio e término após falhas de microfone. Os testes de hardware/serviço usam substitutos controlados; isso não comprova qualidade de transcrição de fala humana.

Consulte [evidencias/testes.txt](evidencias/testes.txt), [evidencias/comandos.json](evidencias/comandos.json) e [evidencias/integracao.json](evidencias/integracao.json). O modo texto e a geração de URLs foram executados de verdade. Não foi aberto navegador durante a validação. A tentativa real de gTTS falhou por conexão neste ambiente; não há áudio de demonstração bem-sucedido. Microfone, reprodução física e reconhecimento de fala humana ainda precisam ser validados no computador do usuário. Não foram fabricados resultados de voz nem capturas de tela.

Se não reconhecer a fala, fale uma frase curta num ambiente silencioso. Se houver erro de serviço, confira a conexão; para continuar, utilize `--texto --sem-audio`. A API padrão de reconhecimento pode limitar requisições e não oferece garantia de disponibilidade para produção.

## Arquivos

| Arquivo | Função |
| --- | --- |
| `assistente.py` | CLI, PLN, ações, reconhecimento e síntese |
| `requirements.txt` | Dependências de voz e reprodução |
| `requirements-microfone.txt` | Dependência adicional para captura |
| `test_assistente.py` | Testes automatizados |
| `assistente_colab.ipynb` | Demonstração no Colab com arquivo de voz |
| `evidencias/` | Resultados reais e escopo da validação |
| `ENTREGA-DIO.md` | Link e descrição para entrega |

## Referências

- [Exemplo TTS da aula — Diego Bruno/DIO](https://github.com/diegobrunoDIO/Text-to-Speech-DIO).
- [Exemplo Speech-to-text da aula — Diego Bruno/DIO](https://github.com/diegobrunoDIO/Speech-to-text-ML-DIO).
- [SpeechRecognition — documentação oficial](https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst).
- [gTTS — documentação oficial](https://gtts.readthedocs.io/en/latest/module.html).
- [pygame.mixer.music — documentação oficial](https://www.pygame.org/docs/ref/music.html).

Os exemplos foram consultados como referência conceitual; este projeto foi escrito separadamente em português, com componentes testáveis e sem as ações de manipulação de lixeira do exemplo original.
