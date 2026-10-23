"""Cria índice visual a partir de fotos locais de um catálogo próprio."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from visual import preprocess_upload,descriptors,image_hash


def main(folder,encoder_path,output,invert=False):
    import tensorflow as tf
    encoder=tf.keras.models.load_model(encoder_path)
    images,ids,seen=[],[],set()
    root=Path(folder)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in [".jpg",".jpeg",".png",".webp"]:
            continue
        with Image.open(path) as image:
            processed=preprocess_upload(image,invert)
        digest=image_hash(processed)
        if digest in seen: continue
        seen.add(digest); images.append(processed); ids.append(str(path.relative_to(root)))
    if len(images)<2:
        raise ValueError("Inclua pelo menos duas imagens distintas no catálogo")
    images=np.stack(images)
    features=np.concatenate([descriptors(encoder,images[i:i+128]) for i in range(0,len(images),128)])
    target=Path(output); target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,images=images,features=features,ids=np.array(ids))
    print("Catálogo indexado:",len(images),"itens em",target)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--encoder",default="artefatos/encoder.keras")
    p.add_argument("--output",default="artefatos/catalogo-proprio.npz")
    p.add_argument("--invert",action="store_true")
    a=p.parse_args()
    main(a.folder,a.encoder,a.output,a.invert)
