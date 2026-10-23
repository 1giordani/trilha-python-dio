"""Detector MTCNN e classificador TensorFlow para identidades cadastradas."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def clip_box(box, width, height, margin=0.05):
    x, y, w, h = map(float, box)
    if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in [x,y,w,h]):
        raise ValueError("Caixa inválida")
    dx, dy = w*margin, h*margin
    result = [max(0, int(math.floor(x-dx))), max(0, int(math.floor(y-dy))),
              min(width, int(math.ceil(x+w+dx))), min(height, int(math.ceil(y+h+dy)))]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("Caixa fora da imagem")
    return result


def crop_face(rgb, box, size=64, keypoints=None):
    bounds = clip_box(box, rgb.width, rgb.height)
    if keypoints and "left_eye" in keypoints and "right_eye" in keypoints:
        left,right=sorted([keypoints["left_eye"],keypoints["right_eye"]],key=lambda p:p[0])
        lx,ly=map(float,left); rx,ry=map(float,right)
        if not all(math.isfinite(v) for v in [lx,ly,rx,ry]) or math.hypot(rx-lx,ry-ly)<1:
            raise ValueError("Landmarks dos olhos inválidos")
        # Transformação inversa: olhos da origem -> posições canônicas (20,24)/(44,24) em 64px.
        target_x,target_y,eye_gap=.3125*size,.375*size,.375*size
        a,d=(rx-lx)/eye_gap,(ry-ly)/eye_gap
        b,e=-d,a
        c,f=lx-a*target_x-b*target_y,ly-d*target_x-e*target_y
        crop=rgb.convert("L").transform((size,size),Image.Transform.AFFINE,(a,b,c,d,e,f),
                                       resample=Image.Resampling.BILINEAR,fillcolor=36)
    else:
        crop = rgb.crop(bounds).convert("L").resize((size,size), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32)[...,None], bounds


def choose_identity(probabilities, names, threshold):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(names) or not np.isfinite(values).all():
        raise ValueError("Vetor de probabilidades inválido")
    if not 0 <= threshold <= 1 or np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(),1,atol=1e-4):
        raise ValueError("Probabilidades ou limiar inválidos")
    index = int(np.argmax(values))
    score = float(values[index])
    return (names[index] if score >= threshold else "nao_cadastrado"), score


class FaceRecognizer:
    def __init__(self, model_path, metadata_path, detector=None):
        import tensorflow as tf
        from mtcnn import MTCNN
        self.model = tf.keras.models.load_model(model_path)
        self.metadata = json.loads(Path(metadata_path).read_text())
        self.detector = detector or MTCNN(device="CPU:0")

    def recognize(self, image):
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        detected = self.detector.detect_faces(np.asarray(rgb))
        crops, boxes, confidences = [], [], []
        for face in detected:
            if float(face["confidence"]) < self.metadata["detector_min_confidence"]:
                continue
            try:
                crop, bounds = crop_face(rgb, face["box"],keypoints=face.get("keypoints")
                                         if self.metadata.get("alignment")=="eyes" else None)
            except ValueError:
                continue
            crops.append(crop)
            boxes.append(bounds)
            confidences.append(float(face["confidence"]))
        if not crops:
            return rgb, []
        # Uma previsão em lote para todas as faces detectadas, sem limitar à primeira.
        probabilities = self.model(np.stack(crops), training=False).numpy()
        results = []
        draw = ImageDraw.Draw(rgb)
        for bounds, confidence, p in zip(boxes, confidences, probabilities):
            label, score = choose_identity(p, self.metadata["classes"], self.metadata["threshold"])
            color = "orange" if label == "nao_cadastrado" else "lime"
            draw.rectangle(bounds, outline=color, width=3)
            draw.text((bounds[0],max(0,bounds[1]-17)), f"{label} {score:.2f}",
                      font=ImageFont.load_default(size=14), fill=color)
            results.append({"box_xyxy":bounds,"label":label,"classifier_score":score,
                            "detection_confidence":confidence})
        return rgb, results
