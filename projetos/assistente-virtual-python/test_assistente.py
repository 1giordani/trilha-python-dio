import contextlib
import io
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from assistente import Acao, executar, interpretar, main, sintetizar


class TestAssistente(unittest.TestCase):
    def test_youtube_abre_home(self):
        self.assertEqual(interpretar("Abrir YouTube").url, "https://www.youtube.com/")

    def test_youtube_codifica_busca(self):
        url = interpretar("Pesquisar no YouTube sobre música & Python").url
        self.assertEqual(parse_qs(urlparse(url).query)["search_query"], ["música & Python"])

    def test_wikipedia_busca(self):
        acao = interpretar("Pesquisar na Wikipédia sobre inteligência artificial")
        self.assertEqual(acao.intencao, "wikipedia")
        self.assertEqual(parse_qs(urlparse(acao.url).query)["search"], ["inteligência artificial"])

    def test_wikipedia_pede_assunto(self):
        self.assertFalse(interpretar("abrir wikipedia").url)

    def test_farmacia_com_local(self):
        url = interpretar("Farmácia próxima", "Centro, Curitiba").url
        self.assertEqual(parse_qs(urlparse(url).query)["query"], ["farmácias perto de Centro, Curitiba"])

    def test_farmacia_local_falado(self):
        url = interpretar("encontrar farmácias em Recife").url
        self.assertIn("Recife", url)

    def test_farmacia_sem_local(self):
        self.assertIn("mim", interpretar("farmácia próxima").url)

    def test_desconhecido_nao_executa_shell(self):
        self.assertEqual(interpretar("rm -rf /; calc.exe").intencao, "ajuda")

    def test_sair(self):
        self.assertTrue(interpretar(" ENCERRAR ASSISTENTE! ").encerrar)

    def test_dry_run_nao_abre(self):
        abrir = Mock()
        self.assertEqual(executar(interpretar("youtube"), True, abrir), "simulado")
        abrir.assert_not_called()

    def test_navegador_falha(self):
        self.assertEqual(executar(interpretar("youtube"), abrir=Mock(return_value=False)), "navegador_indisponivel")

    def test_comando_nao_ouve(self):
        with patch("assistente.reconhecer") as ouvir, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--comando", "sair", "--sem-audio"]), 0)
            ouvir.assert_not_called()

    def test_falhas_de_microfone_encerram(self):
        with patch("assistente.reconhecer", side_effect=OSError("sem microfone")) as ouvir:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--sem-audio"]), 1)
        self.assertEqual(ouvir.call_count, 3)

    def test_texto_encerra_antes_do_limite(self):
        with patch("builtins.input", side_effect=["ajuda", "sair"]) as entrada:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--texto", "--sem-audio", "--dry-run"]), 0)
        self.assertEqual(entrada.call_count, 2)

    def test_erro_de_audio_preserva_resposta_em_texto(self):
        saida = io.StringIO()
        with patch("assistente.sintetizar", side_effect=ConnectionError("rede indisponível")):
            with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--comando", "sair"]), 0)
        self.assertIn("Até a próxima!", saida.getvalue())


if __name__ == "__main__":
    unittest.main()
