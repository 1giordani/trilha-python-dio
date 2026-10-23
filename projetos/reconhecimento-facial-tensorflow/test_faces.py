import unittest
import numpy as np
from PIL import Image
from faces import clip_box, crop_face, choose_identity, FaceRecognizer


class FacesTests(unittest.TestCase):
    def test_caixa_negativa_limitada(self):
        self.assertEqual(clip_box([-10,-10,20,20],100,100,margin=0),[0,0,10,10])

    def test_caixa_invalida(self):
        for box in [[0,0,0,2],[0,0,-1,2],[200,200,10,10],[float("nan"),0,2,2]]:
            with self.subTest(box=box),self.assertRaises(ValueError):
                clip_box(box,100,100)

    def test_preprocessamento(self):
        array, bounds = crop_face(Image.new("RGB",(100,100),"white"),[10,10,30,40])
        self.assertEqual(array.shape,(64,64,1))
        self.assertEqual(array.dtype,np.float32)
        self.assertTrue((array==255).all())

    def test_alinhamento_identidade_geometrica(self):
        image=Image.fromarray(np.arange(64*64,dtype=np.uint8).reshape(64,64)).convert("RGB")
        array,_=crop_face(image,[0,0,64,64],keypoints={"left_eye":[20,24],"right_eye":[44,24]})
        np.testing.assert_array_equal(array[...,0],np.asarray(image.convert("L")))

    def test_identidade_e_limiar(self):
        self.assertEqual(choose_identity([.8,.2],["a","b"],.8),("a",.8))
        self.assertEqual(choose_identity([.7,.3],["a","b"],.8)[0],"nao_cadastrado")

    def test_probabilidades_invalidas(self):
        for vector in [[.8,.8],[-.1,1.1],[float("nan"),0],[.1]]:
            with self.subTest(vector=vector),self.assertRaises(ValueError):
                choose_identity(vector,["a","b"],.8)

    def test_tres_faces_processadas_em_lote(self):
        class Detector:
            def detect_faces(self,image):
                return [{"box":[x,10,20,20],"confidence":.99} for x in [10,40,70]]
        class Model:
            def __call__(self,x,training=False):
                assert len(x)==3
                class Result:
                    def numpy(self):
                        return np.array([[.9,.1],[.1,.9],[.5,.5]])
                return Result()
        system=FaceRecognizer.__new__(FaceRecognizer)
        system.detector,system.model=Detector(),Model()
        system.metadata={"classes":["a","b"],"threshold":.8,"detector_min_confidence":.9}
        _,results=system.recognize(Image.new("RGB",(100,100)))
        self.assertEqual([r["label"] for r in results],["a","b","nao_cadastrado"])

    def test_sem_faces_nao_chama_classificador(self):
        class Detector:
            def detect_faces(self,image): return []
        system=FaceRecognizer.__new__(FaceRecognizer)
        system.detector=Detector()
        system.metadata={"detector_min_confidence":.9}
        _,results=system.recognize(Image.new("RGB",(100,100)))
        self.assertEqual(results,[])


if __name__=="__main__":
    unittest.main(verbosity=2)
