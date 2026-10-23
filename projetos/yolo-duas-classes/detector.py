"""Inferência via libdarknet.so do commit pjreddie fixado no README."""
import ctypes as c
from pathlib import Path


class Box(c.Structure):
    _fields_ = [(k, c.c_float) for k in ["x", "y", "w", "h"]]


class Detection(c.Structure):
    _fields_ = [("bbox", Box), ("classes", c.c_int), ("prob", c.POINTER(c.c_float)),
                ("mask", c.POINTER(c.c_float)), ("objectness", c.c_float), ("sort_class", c.c_int)]


class Image(c.Structure):
    _fields_ = [("w", c.c_int), ("h", c.c_int), ("channels", c.c_int), ("data", c.POINTER(c.c_float))]


class DarknetDetector:
    def __init__(self, darknet, cfg, weights):
        for path in [Path(darknet)/"libdarknet.so", Path(cfg), Path(weights)]:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.lib = lib = c.CDLL(str((Path(darknet)/"libdarknet.so").resolve()))
        lib.load_network.argtypes, lib.load_network.restype = [c.c_char_p, c.c_char_p, c.c_int], c.c_void_p
        lib.load_image_color.argtypes, lib.load_image_color.restype = [c.c_char_p, c.c_int, c.c_int], Image
        lib.network_predict_image.argtypes = [c.c_void_p, Image]
        lib.network_predict_image.restype = c.POINTER(c.c_float)
        lib.get_network_boxes.argtypes = [c.c_void_p, c.c_int, c.c_int, c.c_float, c.c_float,
                                          c.POINTER(c.c_int), c.c_int, c.POINTER(c.c_int)]
        lib.get_network_boxes.restype = c.POINTER(Detection)
        lib.do_nms_sort.argtypes = [c.POINTER(Detection), c.c_int, c.c_int, c.c_float]
        lib.free_detections.argtypes = [c.POINTER(Detection), c.c_int]
        lib.free_image.argtypes = [Image]
        self.net = lib.load_network(str(cfg).encode(), str(weights).encode(), 0)
        if not self.net:
            raise RuntimeError("Falha ao carregar a rede")

    def predict(self, image_path, threshold=0.01, class_ids=(80, 81)):
        image = self.lib.load_image_color(str(image_path).encode(), 0, 0)
        self.lib.network_predict_image(self.net, image)
        count = c.c_int(0)
        detections = self.lib.get_network_boxes(self.net, image.w, image.h, threshold, 0.5,
                                                None, 0, c.byref(count))
        self.lib.do_nms_sort(detections, count.value, 82, 0.45)
        result = []
        try:
            for index in range(count.value):
                for class_id in class_ids:
                    d = detections[index]
                    if d.classes != 82:
                        raise ValueError("O adaptador espera 82 classes")
                    score = float(d.prob[class_id])
                    if score >= threshold:
                        b = d.bbox
                        result.append({"class_id": class_id, "score": score,
                                       "box": [b.x-b.w/2, b.y-b.h/2, b.x+b.w/2, b.y+b.h/2]})
        finally:
            self.lib.free_detections(detections, count.value)
            self.lib.free_image(image)
        return sorted(result, key=lambda d: -d["score"])
