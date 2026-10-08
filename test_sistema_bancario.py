import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).parent / "01 - Estrutura de dados" / "desafio.py"
spec = importlib.util.spec_from_file_location("banco", path)
banco = importlib.util.module_from_spec(spec)
spec.loader.exec_module(banco)

class TestSaques(unittest.TestCase):
    def sacar(self, saldo=1000, valor=100, numero=0):
        return banco.sacar(saldo=saldo, valor=valor, extrato="", limite=500,
                           numero_saques=numero, limite_saques=3)

    def test_quarto_saque_bloqueado(self):
        saldo, extrato, numero = 1000, "", 0
        for _ in range(3):
            saldo, extrato, numero = banco.sacar(
                saldo=saldo, valor=100, extrato=extrato, limite=500,
                numero_saques=numero, limite_saques=3)
        self.assertEqual((saldo, numero), (700, 3))
        self.assertEqual(banco.sacar(
            saldo=saldo, valor=100, extrato=extrato, limite=500,
            numero_saques=numero, limite_saques=3), (saldo, extrato, numero))

    def test_saldo_insuficiente(self):
        self.assertEqual(self.sacar(saldo=50), (50, "", 0))

    def test_limite_por_saque(self):
        self.assertEqual(self.sacar(valor=501), (1000, "", 0))

    def test_valores_invalidos(self):
        for valor in (0, -10):
            self.assertEqual(self.sacar(valor=valor), (1000, "", 0))

if __name__ == "__main__":
    unittest.main()
