"""Assistente educacional em português. Execute --help para ver as opções."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import tempfile
import time
import unicodedata
from urllib.parse import urlencode
import webbrowser


def normalizar(texto):
    texto = unicodedata.normalize("NFD", texto.casefold())
    return " ".join("".join(c for c in texto if unicodedata.category(c) != "Mn").split())


@dataclass(frozen=True)
class Acao:
    intencao: str
    resposta: str
    url: str = ""
    encerrar: bool = False


def interpretar(texto, localizacao=""):
    """PLN por regras: normalização, intenção e extração de termos."""
    original = " ".join(texto.split()).strip(" .!?")
    comando = normalizar(original)
    if comando in {"sair", "encerrar", "parar", "tchau", "encerrar assistente"}:
        return Acao("sair", "Até a próxima!", encerrar=True)
    if re.search(r"\b(youtube|you tube)\b", comando):
        # Os termos depois do nome do serviço preservam acentos e capitalização.
        termo = re.split(r"you\s*tube", original, maxsplit=1, flags=re.I)[-1]
        termo = re.sub(r"^\s*(?:sobre\b|por\b|para\b|:)\s*", "", termo, flags=re.I).strip()
        url = "https://www.youtube.com/"
        if termo:
            url += "results?" + urlencode({"search_query": termo})
        return Acao("youtube", "Vou abrir o YouTube" + (f" com a busca por {termo}." if termo else "."), url)
    if re.search(r"\b(wikipedia|wikipédia)\b", original, re.I):
        termo = re.split(r"wikip[eé]dia", original, maxsplit=1, flags=re.I)[-1]
        termo = re.sub(r"^\s*(?:sobre\b|por\b|:)\s*", "", termo, flags=re.I).strip()
        if not termo:
            return Acao("wikipedia", "Diga: pesquisar na Wikipédia sobre inteligência artificial.")
        return Acao("wikipedia", f"Vou pesquisar {termo} na Wikipédia.",
                    "https://pt.wikipedia.org/w/index.php?" + urlencode({"search": termo}))
    if re.search(r"\bfarmacias?\b", comando):
        match = re.search(r"\bem\s+(.+)$", original, re.I)
        lugar = match.group(1) if match else localizacao.strip()
        consulta = "farmácias perto de " + (lugar or "mim")
        return Acao("farmacia", "Vou abrir a busca de farmácias no mapa. "
                    "O navegador pode pedir sua localização. Confira distância e horário.",
                    "https://www.google.com/maps/search/?" + urlencode({"api": 1, "query": consulta}))
    if re.search(r"\b(hora|horas|horario)\b", comando):
        return Acao("hora", "Agora são " + datetime.now().strftime("%H:%M") + ".")
    return Acao("ajuda", "Comandos: abrir YouTube; pesquisar na Wikipédia sobre um assunto; "
                "encontrar farmácia próxima; que horas são; sair.")


def executar(acao, simulacao=False, abrir=webbrowser.open):
    """Somente URLs construídas pelo roteador; nunca executa shell."""
    if not acao.url or simulacao:
        return "simulado" if acao.url else "sem_link"
    try:
        return "enviado_ao_navegador" if abrir(acao.url, new=2) else "navegador_indisponivel"
    except webbrowser.Error:
        return "navegador_indisponivel"


def sintetizar(texto, destino):
    """gTTS utiliza rede e produz MP3, independentemente da extensão."""
    from gtts import gTTS
    destino = Path(destino)
    if destino.suffix.lower() != ".mp3":
        raise ValueError("A saída de voz deve ter extensão .mp3")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3", dir=destino.parent, delete=False) as temp:
        temporario = Path(temp.name)
    try:
        gTTS(text=texto, lang="pt", tld="com.br", timeout=(5, 15)).save(str(temporario))
        temporario.replace(destino)
    finally:
        temporario.unlink(missing_ok=True)
    return destino


def reproduzir(caminho):
    import pygame
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(caminho))
        pygame.mixer.music.play()
        inicio = time.monotonic()
        while pygame.mixer.music.get_busy():
            if time.monotonic() - inicio > 90:
                raise RuntimeError("Tempo máximo de reprodução atingido")
            time.sleep(0.05)
    finally:
        pygame.mixer.quit()


def reconhecer(arquivo=None, dispositivo=None):
    """Arquivo WAV/AIFF/FLAC ou microfone; Google Speech em pt-BR."""
    import speech_recognition as sr
    reconhecedor = sr.Recognizer()
    reconhecedor.operation_timeout = 15
    if arquivo:
        with sr.AudioFile(str(arquivo)) as fonte:
            audio = reconhecedor.record(fonte, duration=15)
    else:
        with sr.Microphone(device_index=dispositivo) as fonte:
            reconhecedor.adjust_for_ambient_noise(fonte, duration=0.5)
            audio = reconhecedor.listen(fonte, timeout=5, phrase_time_limit=10)
    return reconhecedor.recognize_google(audio, language="pt-BR")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--texto", action="store_true", help="Conversar pelo teclado")
    modo.add_argument("--comando", help="Executar um único comando de texto")
    modo.add_argument("--audio", type=Path, help="Reconhecer até 15 s de WAV/AIFF/FLAC")
    parser.add_argument("--sem-audio", action="store_true", help="Responder só em texto")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar URLs sem abrir navegador")
    parser.add_argument("--localizacao", default="", help="Bairro/cidade para busca de farmácias")
    parser.add_argument("--dispositivo", type=int, help="Índice do microfone no PyAudio")
    parser.add_argument("--turnos", type=int, default=10, help="Máximo de interações (1 a 100)")
    parser.add_argument("--saida-voz", type=Path, default=Path("saidas/resposta.mp3"))
    args = parser.parse_args(argv)
    if not 1 <= args.turnos <= 100:
        parser.error("--turnos deve estar entre 1 e 100")
    limite = 1 if args.comando is not None or args.audio else args.turnos
    falhas = 0
    for _ in range(limite):
        try:
            if args.comando is not None:
                texto = args.comando
            elif args.texto:
                texto = input("Você: ")
            else:
                print("Ouvindo... (Ctrl+C encerra)")
                texto = reconhecer(args.audio, args.dispositivo)
            falhas = 0
        except (EOFError, KeyboardInterrupt):
            print("\nAssistente encerrada.")
            return 0
        except Exception as erro:
            # Dependências, silêncio, falta de dispositivo e erro do serviço.
            print(f"Não foi possível reconhecer a fala ({type(erro).__name__}): {erro}", file=sys.stderr)
            falhas += 1
            if limite == 1 or falhas >= 3:
                print("Use --texto --sem-audio para continuar sem microfone e sem rede.", file=sys.stderr)
                return 1
            continue
        print("Entrada:", texto)
        acao = interpretar(texto, args.localizacao)
        estado = executar(acao, args.dry_run)
        print(json.dumps({**asdict(acao), "execucao": estado}, ensure_ascii=False))
        if estado == "navegador_indisponivel":
            print("Abra o link exibido manualmente.")
        if not args.sem_audio:
            try:
                reproduzir(sintetizar(acao.resposta, args.saida_voz))
            except KeyboardInterrupt:
                return 0
            except Exception as erro:
                print(f"Resposta disponível em texto; áudio indisponível: {erro}", file=sys.stderr)
        if acao.encerrar:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
