"""Treina CNN multiclasse e avalia busca visual em catálogo disjunto."""
import os
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import tensorflow as tf
from visual import CLASS_NAMES, PALETTE, colorize, descriptors, top_k


def create_groups(data,seed=42):
    rng=np.random.default_rng(seed)
    groups={name:{"images":[],"labels":[],"ids":[],"raw_hashes":[],"color_ids":[]}
            for name in ["train","valid","catalog","queries"]}
    seen=set()
    for source,(images,labels),sizes in [
        ("official_train",data[0],[("train",800),("valid",100)]),
        ("official_test",data[1],[("catalog",100),("queries",50)])]:
        for class_id in range(10):
            candidates=iter(rng.permutation(np.flatnonzero(labels==class_id)).tolist())
            for name,count in sizes:
                for _ in range(count):
                    while True:
                        index=next(candidates)
                        digest=hashlib.sha256(images[index].tobytes()).hexdigest()
                        if digest not in seen:
                            seen.add(digest)
                            break
                    color_id=int(rng.integers(len(PALETTE)))
                    group=groups[name]
                    group["images"].append(colorize(images[index],color_id))
                    group["labels"].append(class_id)
                    group["ids"].append(f"{source}_{index:05d}")
                    group["raw_hashes"].append(digest)
                    group["color_ids"].append(color_id)
    for group in groups.values():
        group["images"]=np.stack(group["images"])
        group["labels"]=np.asarray(group["labels"])
    return groups


def build_model():
    inputs=tf.keras.Input((28,28,3))
    x=tf.keras.layers.Rescaling(1/255)(inputs)
    x=tf.keras.layers.Conv2D(16,3,activation="relu")(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(32,3,activation="relu")(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Flatten()(x)
    embedding=tf.keras.layers.Dense(64,activation="relu",name="embedding")(x)
    outputs=tf.keras.layers.Dense(10,activation="softmax")(embedding)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model


def dataset(group,training=False):
    ds=tf.data.Dataset.from_tensor_slices((group["images"],group["labels"]))
    if training:
        ds=ds.shuffle(len(group["labels"]),seed=42)
    options=tf.data.Options()
    options.threading.private_threadpool_size=2
    return ds.batch(64).with_options(options).prefetch(1)


def embed_in_batches(encoder,images):
    return np.concatenate([descriptors(encoder,images[i:i+128]) for i in range(0,len(images),128)])


def main(epochs=5):
    tf.keras.utils.set_random_seed(42)
    out,evidence=Path("artefatos"),Path("evidencias")
    out.mkdir(exist_ok=True); evidence.mkdir(exist_ok=True)
    groups=create_groups(tf.keras.datasets.fashion_mnist.load_data())
    model=build_model()
    history=model.fit(dataset(groups["train"],True),validation_data=dataset(groups["valid"]),epochs=epochs,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)],verbose=2)
    encoder=tf.keras.Model(model.input,model.get_layer("embedding").output)
    catalog=groups["catalog"]; queries=groups["queries"]
    catalog_features=embed_in_batches(encoder,catalog["images"])
    query_features=embed_in_batches(encoder,queries["images"])
    ranks,scores=[],[]
    for feature in query_features:
        indices,similarity=top_k(feature,catalog_features,5)
        ranks.append(indices.tolist()); scores.append(similarity.tolist())
    ranks=np.asarray(ranks)
    relevant=catalog["labels"][ranks]==queries["labels"][:,None]
    rng=np.random.default_rng(42)
    random_ranks=np.stack([rng.choice(len(catalog_features),5,replace=False) for _ in query_features])
    baseline=catalog["labels"][random_ranks]==queries["labels"][:,None]
    predicted=model(queries["images"].astype("float32"),training=False).numpy().argmax(axis=1)
    report={"scope":"Fashion-MNIST colorizado artificialmente; consulta e catálogo separados",
            "environment":{"python":platform.python_version(),"tensorflow":tf.__version__},
            "counts":{k:len(v["labels"]) for k,v in groups.items()},
            "classes":CLASS_NAMES,"class_count":10,"embedding_dimensions":64,"color_histogram_dimensions":24,
            "descriptor_dimensions":88,"color_weight":.25,"epochs_executed":len(history.history["loss"]),
            "selected_epoch":int(np.argmin(history.history["val_loss"])+1),
            "classifier_accuracy_on_queries":float((predicted==queries["labels"]).mean()),
            "precision_at_5_class_proxy":float(relevant.mean()),
            "same_color_at_5_synthetic_proxy":float((np.asarray(catalog["color_ids"])[ranks]==np.asarray(queries["color_ids"])[:,None]).mean()),
            "hit_rate_at_5_class_proxy":float(relevant.any(axis=1).mean()),
            "random_precision_at_5_class_proxy":float(baseline.mean()),
            "precision_at_5_by_class":{CLASS_NAMES[c]:float(relevant[queries["labels"]==c].mean()) for c in range(10)},
            "identical_raw_images_across_groups":False,
            "ranking_uses_labels":False,"ranking_uses_text_or_price":False,
            "metric_limitation":"relevância medida por categoria; não é avaliação humana de semelhança visual"}
    model.save(out/"classificador.keras"); encoder.save(out/"encoder.keras")
    np.savez_compressed(out/"catalogo.npz",images=catalog["images"],features=catalog_features,ids=np.array(catalog["ids"]))
    np.savez_compressed(out/"consultas.npz",images=queries["images"],ids=np.array(queries["ids"]))
    (out/"metadata.json").write_text(json.dumps({"classes":CLASS_NAMES,"color_weight":.25,"size":28,
        "domain":"produtos Fashion-MNIST com cores artificiais e fundo preto"},ensure_ascii=False,indent=2))
    (evidence/"metricas.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (evidence/"historico.json").write_text(json.dumps(history.history,indent=2))
    manifest={name:{k:v for k,v in group.items() if k not in ["images","labels"]} for name,group in groups.items()}
    (evidence/"manifesto.json").write_text(json.dumps(manifest,indent=2))
    (evidence/"ranking-consultas.json").write_text(json.dumps({"query_ids":queries["ids"],"catalog_ids":catalog["ids"],
        "indices":ranks.tolist(),"similarities":scores},indent=2))
    from figures import recommendation_grid
    # Primeira consulta de cada classe, sem seleção pelo acerto do ranking.
    first=[int(np.flatnonzero(queries["labels"]==c)[0]) for c in range(10)]
    recommendation_grid(queries["images"][first],catalog["images"],ranks[first],np.asarray(scores)[first],
                        evidence/"recomendacoes.png",[CLASS_NAMES[c] for c in range(10)])
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=5)
    a=p.parse_args()
    if a.epochs<1: p.error("epochs deve ser positivo")
    main(a.epochs)
