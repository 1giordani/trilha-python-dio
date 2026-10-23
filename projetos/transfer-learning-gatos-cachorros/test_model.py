"""Integração sintética: não mede acurácia em gatos/cachorros."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from pipeline import Config, build_model, compile_model, enable_fine_tuning


class ModelIntegration(unittest.TestCase):
    def test_training_modes_and_export(self):
        tf.keras.utils.set_random_seed(42)
        config = Config(image_size=32, batch_size=2)
        model, base = build_model(config, weights=None)
        self.assertEqual(len(base.trainable_weights), 0)
        images = np.random.default_rng(42).uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        labels = np.array([0, 1], dtype="float32")
        compile_model(model, config.head_lr)
        self.assertTrue(np.isfinite(model.train_on_batch(images, labels)).all())
        enable_fine_tuning(base, config.fine_layers)
        self.assertGreater(len(base.trainable_weights), 0)
        self.assertTrue(all(not layer.trainable for layer in base.layers
                            if isinstance(layer, tf.keras.layers.BatchNormalization)))
        compile_model(model, config.fine_lr)
        self.assertTrue(np.isfinite(model.train_on_batch(images, labels)).all())
        expected = model(images, training=False).numpy()
        self.assertEqual(expected.shape, (2, 1))
        self.assertTrue(((expected >= 0) & (expected <= 1)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(expected, restored(images, training=False).numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
