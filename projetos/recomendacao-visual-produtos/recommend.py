"""Consulta local a um catálogo visual; não usa nomes ou preços no ranking."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from visual import descriptors,top_k,preprocess_upload
from figures import recommendation_grid


def load_system(encoder_path="artefatos/encoder.keras",catalog_path="artefatos/catalogo.npz"):
    import tensorflow as tf
    encoder=tf.keras.models.load_model(encoder_path)
    catalog=np.load(catalog_path,allow_pickle=False)
    return encoder,catalog


def recommend(encoder,catalog,image,k=5,invert=False):
    query=preprocess_upload(image,invert)
    feature=descriptors(encoder,query[None,...])[0]
    duplicate=np.flatnonzero(np.all(catalog["images"]==query,axis=(1,2,3)))
    excluded=int(duplicate[0]) if len(duplicate) else None
    indices,scores=top_k(feature,catalog["features"],k,excluded)
    items=[{"item_id":str(catalog["ids"][i]),"catalog_index":int(i),"cosine_similarity":float(score)}
           for i,score in zip(indices,scores)]
    return query,indices,scores,items


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--encoder",default="artefatos/encoder.keras")
    p.add_argument("--catalog",default="artefatos/catalogo.npz")
    p.add_argument("--k",type=int,default=5)
    p.add_argument("--invert",action="store_true")
    p.add_argument("--output",default="resultados")
    a=p.parse_args()
    if a.k<1: p.error("k deve ser positivo")
    encoder,catalog=load_system(a.encoder,a.catalog)
    with Image.open(a.image) as image:
        query,indices,scores,items=recommend(encoder,catalog,image,a.k,a.invert)
    if not len(items): p.error("Nenhum outro item disponível no catálogo")
    out=Path(a.output); out.mkdir(exist_ok=True,parents=True)
    (out/"recomendacoes.json").write_text(json.dumps(items,ensure_ascii=False,indent=2))
    recommendation_grid([query],catalog["images"],[indices],[scores],out/"recomendacoes.png")
    print(json.dumps(items,ensure_ascii=False,indent=2))
