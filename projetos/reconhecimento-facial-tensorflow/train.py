"""Piloto Olivetti: MTCNN pré-treinada + CNN de classificação treinada do zero."""
import os
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
import json
import platform
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from mtcnn import MTCNN
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import classification_report, confusion_matrix
from faces import crop_face, FaceRecognizer


def make_split(target, seed=42):
    rng = np.random.default_rng(seed)
    split = {key:[] for key in ["train","valid","test","unknown_calibration","unknown_test"]}
    for identity in range(8):
        indices = rng.permutation(np.flatnonzero(target == identity))
        for key, subset in zip(["train","valid","test"], [indices[:6],indices[6:8],indices[8:]]):
            split[key].extend(map(int,subset))
    split["unknown_calibration"] = np.flatnonzero(np.isin(target,[8,9])).tolist()
    split["unknown_test"] = np.flatnonzero(np.isin(target,[10,11])).tolist()
    all_indices = [i for values in split.values() for i in values]
    assert len(all_indices) == len(set(all_indices))
    return split


def build_classifier(classes=8):
    return tf.keras.Sequential([
        tf.keras.Input((64,64,1)), tf.keras.layers.Rescaling(1/255),
        tf.keras.layers.Conv2D(16,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32,3,activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dropout(.25,seed=42), tf.keras.layers.Dense(classes,activation="softmax"),
    ])


def choose_threshold(known_prob, known_labels, unknown_prob):
    scores, predictions = known_prob.max(axis=1), known_prob.argmax(axis=1)
    unknown_scores = unknown_prob.max(axis=1)
    # Critério fixado: média entre conhecidos corretamente aceitos e desconhecidos rejeitados.
    # Calibração usa somente validação e identidades desconhecidas reservadas para isso.
    rows = []
    for threshold in np.linspace(.125,1,176):
        correct_accept = float(np.mean((predictions==known_labels)&(scores>=threshold)))
        rejection = float(np.mean(unknown_scores<threshold))
        rows.append({"threshold":float(threshold),"known_correct_acceptance":correct_accept,
                     "unknown_rejection":rejection,"objective":(correct_accept+rejection)/2})
    best = max(rows,key=lambda row:(row["objective"],row["threshold"]))
    return best, rows


def as_dataset(x,y,training=False):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    if training:
        dataset = dataset.shuffle(len(x),seed=42)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.batch(8).with_options(options).prefetch(1)


def main(epochs=40):
    tf.keras.utils.set_random_seed(42)
    out, evidence = Path("artefatos"), Path("evidencias")
    out.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)
    data = fetch_olivetti_faces(data_home="cache")
    split = make_split(data.target)
    detector = MTCNN(device="CPU:0")
    arrays, labels, used, skipped = {}, {}, {}, []
    for group, indices in split.items():
        x, y, accepted = [], [], []
        for index in indices:
            gray = Image.fromarray(np.clip(data.images[index]*255,0,255).astype("uint8"))
            rgb = Image.new("RGB",(352,352),"#242424")
            photo = gray.resize((256,256),Image.Resampling.BILINEAR).convert("RGB")
            rgb.paste(photo,(48,48))
            faces = detector.detect_faces(np.asarray(rgb))
            faces = [face for face in faces if face["confidence"] >= .9]
            if not faces:
                skipped.append({"index":index,"split":group,"reason":"MTCNN não encontrou face >=0.9"})
                continue
            face = max(faces,key=lambda f:f["confidence"])
            crop,_ = crop_face(rgb,face["box"],keypoints=face.get("keypoints"))
            x.append(crop); y.append(int(data.target[index])); accepted.append(index)
        if not x:
            raise RuntimeError(f"Nenhuma face aproveitável: {group}")
        arrays[group],labels[group],used[group] = np.stack(x),np.array(y),accepted
        print(group, len(x), "faces", flush=True)
    for identity in range(8):
        if np.sum(labels["train"]==identity)<3:
            raise RuntimeError(f"Poucas faces de treino para classe {identity}")
    model = build_classifier()
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    history = model.fit(as_dataset(arrays["train"],labels["train"],True),
                        validation_data=as_dataset(arrays["valid"],labels["valid"]),epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True)],verbose=2)
    val_p = model(arrays["valid"],training=False).numpy()
    calib_p = model(arrays["unknown_calibration"],training=False).numpy()
    best, calibration = choose_threshold(val_p,labels["valid"],calib_p)
    names = [f"pessoa_{i:02d}" for i in range(8)]
    metadata = {"classes":names,"threshold":best["threshold"],"detector_min_confidence":.9,
                "input_size":64,"input_mode":"L","input_range":"0..255; Rescaling no modelo","alignment":"eyes",
                "rejection":"heurística pelo máximo softmax; não garante detectar desconhecidos"}
    model.save(out/"classificador.keras")
    (out/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    # Testes conhecidos e identidades não cadastradas são avaliados somente agora.
    test_p = model(arrays["test"],training=False).numpy()
    unknown_p = model(arrays["unknown_test"],training=False).numpy()
    predicted, max_scores = test_p.argmax(axis=1),test_p.max(axis=1)
    accepted = max_scores >= best["threshold"]
    unknown_accepted = unknown_p.max(axis=1) >= best["threshold"]
    report = {"scope":"piloto de identificação de 8 classes Olivetti, não nomes reais",
              "environment":{"python":platform.python_version(),"tensorflow":tf.__version__,"device":"CPU"},
              "split_indices":split,"used_indices":used,"skipped":skipped,
              "used_counts":{k:len(v) for k,v in used.items()},"epochs_executed":len(history.history["loss"]),
              "selected_epoch":int(np.argmin(history.history["val_loss"])+1),
              "known_closed_set_accuracy":float(np.mean(predicted==labels["test"])),
              "known_correct_acceptance":float(np.mean((predicted==labels["test"])&accepted)),
              "known_acceptance_rate":float(accepted.mean()),
              "unknown_false_acceptance_rate":float(unknown_accepted.mean()),
              "unknown_rejection_rate":float(1-unknown_accepted.mean()),
              "threshold_calibration":best,
              "classification_report":classification_report(labels["test"],predicted,labels=list(range(8)),
                                     target_names=names,output_dict=True,zero_division=0),
              "confusion_matrix":confusion_matrix(labels["test"],predicted,labels=list(range(8))).tolist(),
              "limitations":["poucas fotos por classe", "mesmas identidades conhecidas nos três grupos",
                             "sem teste de vivacidade", "sem garantia de rejeição de identidades novas"]}
    (evidence/"metricas.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (evidence/"historico.json").write_text(json.dumps(history.history,indent=2))
    (evidence/"calibracao.json").write_text(json.dumps(calibration,indent=2))
    from demo import create_demo
    demo = create_demo(data,used,FaceRecognizer(out/"classificador.keras",out/"metadata.json",detector))
    print(json.dumps({k:report[k] for k in ["used_counts","known_closed_set_accuracy","unknown_rejection_rate"]},indent=2))
    print("Faces simultâneas detectadas:",demo["detected_faces"])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=40)
    a = p.parse_args()
    if a.epochs < 1:
        p.error("epochs deve ser positivo")
    main(a.epochs)
