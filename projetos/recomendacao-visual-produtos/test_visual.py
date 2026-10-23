import unittest
import numpy as np
from PIL import Image
from visual import normalize_rows,top_k,color_histograms,colorize,preprocess_upload


class VisualTests(unittest.TestCase):
    def test_normalizacao(self):
        result=normalize_rows([[3,4],[0,2]])
        np.testing.assert_allclose(np.linalg.norm(result,axis=1),1)

    def test_rejeita_vetor_zero(self):
        with self.assertRaises(ValueError): normalize_rows([[0,0]])

    def test_ordem_cosseno(self):
        indices,scores=top_k([1,0],[[1,0],[0,1],[-1,0]],3)
        self.assertEqual(indices.tolist(),[0,1,2])
        np.testing.assert_allclose(scores,[1,0,-1])

    def test_exclusao_da_propria_consulta(self):
        indices,_=top_k([1,0],[[1,0],[.9,.1],[0,1]],5,exclude_index=0)
        self.assertEqual(indices.tolist(),[1,2])

    def test_empate_deterministico(self):
        indices,_=top_k([1,0],[[1,0],[1,0]],2)
        self.assertEqual(indices.tolist(),[0,1])

    def test_k_invalido(self):
        with self.assertRaises(ValueError): top_k([1,0],[[1,0]],0)

    def test_histograma_distingue_cor(self):
        red=np.full((28,28,3),[255,0,0],dtype=np.uint8)
        blue=np.full((28,28,3),[0,0,255],dtype=np.uint8)
        result=color_histograms([red,blue])
        self.assertEqual(result.shape,(2,24))
        self.assertLess(float(result[0]@result[1]),1)

    def test_sem_conteudo(self):
        with self.assertRaises(ValueError): color_histograms(np.zeros((1,28,28,3),dtype=np.uint8))

    def test_preprocessamento_tamanho(self):
        result=preprocess_upload(Image.new("RGBA",(100,50),(255,0,0,255)))
        self.assertEqual(result.shape,(28,28,3))
        self.assertEqual(result.dtype,np.uint8)

    def test_colorizacao_nao_muda_preto(self):
        self.assertTrue((colorize(np.zeros((28,28),dtype=np.uint8),2)==0).all())


if __name__=="__main__":
    unittest.main(verbosity=2)
