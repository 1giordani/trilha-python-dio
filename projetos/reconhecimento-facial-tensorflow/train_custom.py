"""Cadastro local: uma pasta por participante, com fotos individuais autorizadas."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from mtcnn import MTCNN
from faces import crop_face
from train import build_classifier, as_dataset


def main(data_dir, epochs=40, threshold=.8):
    tf.keras.utils.set_random_seed(42)
    folders = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    if len(folders)<2:
        raise ValueError("Crie pelo menos duas pastas, uma por identidade.")
    detector, hashes, rng = MTCNN(device="CPU:0"), set(), np.random.default_rng(42)
    split = {name:([],[]) for name in ["train","valid","test"]}
    for class_id, folder in enumerate(folders):
        photos = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in [".png",".jpg",".jpeg"]:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError("Foto duplicada no cadastro: " + str(path))
            hashes.add(digest)
            with Image.open(path) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                rgb.thumbnail((1024,1024))
            faces = [f for f in detector.detect_faces(np.asarray(rgb)) if f["confidence"] >= .9]
            if len(faces)!=1:
                raise ValueError(f"Use foto individual com exatamente uma face detectável: {path}")
            crop,_ = crop_face(rgb,faces[0]["box"],keypoints=faces[0].get("keypoints"))
            photos.append(crop)
        if len(photos)<10:
            raise ValueError(f"Use pelo menos 10 fotos individuais válidas em {folder}")
        order = rng.permutation(len(photos))
        a,b = int(.6*len(order)),int(.8*len(order))
        for name,indices in zip(split,[order[:a],order[a:b],order[b:]]):
            for index in indices:
                split[name][0].append(photos[index]); split[name][1].append(class_id)
    arrays = {name:(np.stack(x),np.array(y)) for name,(x,y) in split.items()}
    model = build_classifier(len(folders))
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(as_dataset(*arrays["train"],training=True),validation_data=as_dataset(*arrays["valid"]),
              epochs=epochs,callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True)],verbose=2)
    out = Path("artefatos-personalizados")
    out.mkdir(exist_ok=True)
    model.save(out/"classificador.keras")
    metadata = {"classes":[p.name for p in folders],"threshold":threshold,"detector_min_confidence":.9,"alignment":"eyes",
                "threshold_status":"limiar heurístico fornecido; não calibrado com desconhecidos"}
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    result = model.evaluate(as_dataset(*arrays["test"]),return_dict=True,verbose=0)
    (out/"metricas.json").write_text(json.dumps(result,indent=2))
    print("Modelo salvo em",out,"; fotos e nomes não são enviados para um servidor.")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="dados-pessoais")
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--threshold",type=float,default=.8)
    a=p.parse_args()
    if a.epochs<1 or not 0<=a.threshold<=1:
        p.error("Use epochs positivo e limiar entre 0 e 1")
    main(a.data,a.epochs,a.threshold)
