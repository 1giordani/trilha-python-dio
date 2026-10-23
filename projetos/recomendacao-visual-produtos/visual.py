"""Descritores exclusivamente visuais: embedding CNN + histograma RGB."""
import hashlib
import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES=["camiseta","calca","pulover","vestido","casaco","sandalia","camisa","tenis","bolsa","bota"]
PALETTE=np.array([[255,75,75],[70,150,255],[80,225,125],[245,210,70],[200,90,235],[220,170,125]],dtype=np.float32)


def colorize(gray, color_id):
    """Cores artificiais independentes da classe; não são cores originais da base."""
    return np.rint(np.asarray(gray,dtype=np.float32)[...,None]*PALETTE[color_id]/255).astype("uint8")


def normalize_rows(values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError("Esperada matriz finita de descritores")
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if np.any(norms<=1e-12):
        raise ValueError("Descritor sem conteúdo")
    return values/norms


def color_histograms(images,bins=8):
    result=[]
    for image in np.asarray(images):
        pixels=image.reshape(-1,3).astype(np.float32)
        foreground=pixels[np.max(pixels,axis=1)>15] # exclui fundo preto do laboratório
        if not len(foreground):
            raise ValueError("Imagem sem pixels de produto detectáveis")
        histogram=np.concatenate([np.histogram(foreground[:,channel],bins=bins,range=(0,256))[0]
                                  for channel in range(3)]).astype(np.float32)
        result.append(histogram)
    return normalize_rows(result)


def descriptors(encoder,images,color_weight=.25):
    if not 0<=color_weight<=1:
        raise ValueError("Peso da cor deve estar entre 0 e 1")
    embeddings=normalize_rows(encoder(np.asarray(images,dtype=np.float32),training=False).numpy())
    colors=color_histograms(images)
    return normalize_rows(np.concatenate([np.sqrt(1-color_weight)*embeddings,
                                          np.sqrt(color_weight)*colors],axis=1))


def top_k(query,catalog,k=5,exclude_index=None):
    catalog=normalize_rows(catalog)
    query=normalize_rows(np.asarray(query).reshape(1,-1))[0]
    if query.shape[0]!=catalog.shape[1]:
        raise ValueError("Dimensões de descritor incompatíveis")
    if type(k) is not int or k<1:
        raise ValueError("k deve ser inteiro positivo")
    scores=catalog@query
    indices=np.arange(len(catalog))
    if exclude_index is not None:
        if not 0<=exclude_index<len(catalog):
            raise ValueError("Índice de exclusão inválido")
        indices=indices[indices!=exclude_index]
    ranked=indices[np.argsort(-scores[indices],kind="stable")[:k]]
    return ranked, np.clip(scores[ranked],-1,1)


def image_hash(image):
    return hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()


def preprocess_upload(image,invert=False):
    image=ImageOps.exif_transpose(image).convert("RGBA")
    background=Image.new("RGBA",image.size,(0,0,0,255))
    rgb=Image.alpha_composite(background,image).convert("RGB")
    if invert:
        rgb=ImageOps.invert(rgb)
    rgb=ImageOps.pad(rgb,(28,28),method=Image.Resampling.BILINEAR,color="black")
    return np.asarray(rgb,dtype=np.uint8)
