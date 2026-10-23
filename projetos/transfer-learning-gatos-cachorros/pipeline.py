# %%
"""Pipeline de Transfer Learning. As funções também compõem o notebook Colab."""
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


@dataclass
class Config:
    seed: int = 42
    image_size: int = 160
    batch_size: int = 32
    head_epochs: int = 5
    fine_epochs: int = 3
    fine_layers: int = 30
    head_lr: float = 0.001
    fine_lr: float = 0.00001
    train_limit: int = 0  # 0: todo o treino; >0: ensaio reduzido, registrado no relatório
    evaluation_limit: int = 0  # 0: validação/teste completos; >0: somente ensaio de integração
    output_dir: str = "resultados"


def setup(config):
    tf.keras.utils.set_random_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

# %%
def load_data(config):
    # O catálogo possui apenas 'train'; reservamos faixas disjuntas ANTES do shuffle.
    raw, info = tfds.load(
        "cats_vs_dogs:4.0.1",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True, with_info=True, shuffle_files=False,
    )
    names = list(info.features["label"].names)
    assert len(names) == 2
    original_counts = [int(tf.data.experimental.cardinality(ds).numpy()) for ds in raw]
    if config.train_limit > 0:
        raw[0] = raw[0].take(config.train_limit)
    if config.evaluation_limit > 0:
        raw[1] = raw[1].take(config.evaluation_limit)
        raw[2] = raw[2].take(config.evaluation_limit)
    used_counts = [int(tf.data.experimental.cardinality(ds).numpy()) for ds in raw]

    def resize(image, label):
        image = tf.image.resize(image, (config.image_size, config.image_size))
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    def prepare(ds, training):
        # Buffer antes do resize evita armazenar milhares de tensores float32 grandes.
        if training:
            ds = ds.shuffle(512, seed=config.seed, reshuffle_each_iteration=True)
        return ds.map(resize, num_parallel_calls=tf.data.AUTOTUNE).batch(
            config.batch_size
        ).prefetch(tf.data.AUTOTUNE)

    datasets = [prepare(ds, i == 0) for i, ds in enumerate(raw)]
    metadata = {
        "dataset": info.full_name,
        "classes": names,
        "splits": ["train[:80%]", "train[80%:90%]", "train[90%:]"],
        "original_counts": dict(zip(["treino", "validacao", "teste"], original_counts)),
        "used_counts": dict(zip(["treino", "validacao", "teste"], used_counts)),
        "split_stratified": False,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return datasets, metadata


def show_samples(dataset, class_names, output_dir):
    images, labels = next(iter(dataset))
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(np.clip(images[i].numpy(), 0, 255).astype("uint8"))
            ax.set_title(class_names[int(labels[i])])
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "amostras.png", dpi=140)
    plt.show()
    plt.close(fig)

# %%
def build_model(config, weights="imagenet"):
    # weights=None é reservado ao teste sintético de integração, não ao experimento.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(config.image_size, config.image_size, 3),
        include_top=False, weights=weights,
    )
    base.trainable = False
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=config.seed),
        tf.keras.layers.RandomRotation(0.08, seed=config.seed + 1),
    ], name="aumento_de_dados")
    inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
    x = augment(inputs)  # Camadas aleatórias ficam inativas na avaliação/inferência.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    # BatchNormalization permanece em modo de inferência também no fine-tuning.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2, seed=config.seed + 2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classe_1")(x)
    return tf.keras.Model(inputs, outputs), base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )


def enable_fine_tuning(base, last_layers):
    if last_layers <= 0:
        raise ValueError("fine_layers deve ser maior que zero")
    base.trainable = True
    boundary = max(0, len(base.layers) - last_layers)
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= boundary and not isinstance(layer, tf.keras.layers.BatchNormalization)


def callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

# %%
def train_stages(model, base, train, validation, config):
    out = Path(config.output_dir)
    head_path = out / "extracao.weights.h5"
    fine_path = out / "fine_tuning.weights.h5"
    compile_model(model, config.head_lr)
    frozen = model.fit(train, validation_data=validation, epochs=config.head_epochs,
                       callbacks=callbacks(head_path))
    model.load_weights(head_path)
    frozen_val = model.evaluate(validation, return_dict=True, verbose=0)

    enable_fine_tuning(base, config.fine_layers)
    # Recompilar é obrigatório depois de alterar as camadas treináveis.
    compile_model(model, config.fine_lr)
    fine = model.fit(train, validation_data=validation, epochs=config.fine_epochs,
                     callbacks=callbacks(fine_path))
    model.load_weights(fine_path)
    fine_val = model.evaluate(validation, return_dict=True, verbose=0)
    selected = "fine_tuning" if fine_val["loss"] < frozen_val["loss"] else "extracao"
    if selected == "extracao":
        model.load_weights(head_path)
    comparison = {"extracao": frozen_val, "fine_tuning": fine_val, "selecionado": selected,
                  "criterio": "menor loss na validação; teste não usado na seleção"}
    histories = {"extracao": frozen.history, "fine_tuning": fine.history}
    return histories, comparison


def plot_history(histories, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    offset = 0
    for stage, history in histories.items():
        epochs = range(offset + 1, offset + len(history["loss"]) + 1)
        for ax, key in zip(axes, ["accuracy", "loss"]):
            ax.plot(epochs, history[key], marker="o", label=f"treino / {stage}")
            ax.plot(epochs, history[f"val_{key}"], marker="o", label=f"validação / {stage}")
            ax.set_xlabel("Época")
            ax.set_title(key)
            ax.legend(fontsize=8)
        offset += len(history["loss"])
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "curvas.png", dpi=150)
    plt.show()
    plt.close(fig)

# %%
def evaluate_and_export(model, test, metadata, histories, comparison, config):
    # Teste acessado só após a seleção. Rótulo e previsão são coletados juntos.
    metrics = model.evaluate(test, return_dict=True, verbose=0)
    truth, probabilities = [], []
    for images, labels in test:
        truth.extend(labels.numpy().astype(int).tolist())
        probabilities.extend(model(images, training=False).numpy().reshape(-1).tolist())
    truth = np.asarray(truth)
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    report = classification_report(truth, predicted, labels=[0, 1],
                                   target_names=metadata["classes"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    result = {
        "status": "treinamento_e_avaliacao_executados",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "data": metadata,
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__,
                        "tensorflow_datasets": tfds.__version__,
                        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]},
        "validation_comparison": comparison, "test_metrics": metrics,
        "test_class_distribution": np.bincount(truth, minlength=2).tolist(),
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "decision_threshold": 0.5,
    }
    out = Path(config.output_dir)
    (out / "metricas.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "historico.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    np.savetxt(out / "predicoes.csv", np.column_stack([truth, probabilities, predicted]),
               delimiter=",", header="real,probabilidade_classe_1,predito", comments="", fmt=["%d", "%.8f", "%d"])
    model.save(out / "modelo.keras")
    (out / "classes.json").write_text(json.dumps(metadata["classes"]), encoding="utf-8")
    ConfusionMatrixDisplay(matrix, display_labels=metadata["classes"]).plot(cmap="Blues", colorbar=False)
    plt.title("Teste reservado — modelo selecionado pela validação")
    plt.tight_layout()
    plt.savefig(out / "matriz_confusao.png", dpi=150)
    plt.show()
    plt.close()
    print(json.dumps(metrics, indent=2))
    return result


def classify_image(model, path, class_names, image_size):
    image = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    array = tf.keras.utils.img_to_array(image)[None, ...]
    probability = float(model(array, training=False).numpy()[0, 0])
    label = int(probability >= 0.5)
    return {"classe": class_names[label], "probabilidade_classe_1": probability,
            "aviso": "Escore do modelo, sem garantia de calibração ou validade fora dessas duas classes."}
