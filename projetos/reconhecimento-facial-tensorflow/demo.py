"""Montagem ilustrativa fixa: primeira foto de validação de cada classe e um não cadastrado."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_demo(data, used, system):
    # Todas as classes conhecidas, sem escolher fotos pelo resultado das previsões.
    indices = [next(i for i in used["valid"] if data.target[i]==subject) for subject in range(8)]
    indices.append(used["unknown_calibration"][0])
    montage = Image.new("RGB",(1056,1056),"#242424")
    for tile,index in enumerate(indices):
        photo = Image.fromarray(np.clip(data.images[index]*255,0,255).astype("uint8"))
        montage.paste(photo.resize((256,256),Image.Resampling.BILINEAR).convert("RGB"),
                      (48+352*(tile%3),48+352*(tile//3)))
    Path("artefatos").mkdir(exist_ok=True)
    montage.save("artefatos/cena_demo.png")
    recognized, detections = system.recognize(montage)
    matched = []
    for detection in detections:
        x1,y1,x2,y2=detection["box_xyxy"]
        tile=int(((y1+y2)/2)//352)*3+int(((x1+x2)/2)//352)
        expected_id=int(data.target[indices[tile]]) if 0<=tile<len(indices) else None
        expected_label=f"pessoa_{expected_id:02d}" if expected_id is not None and expected_id<8 else "nao_cadastrado"
        matched.append(dict(detection,tile=tile,expected_label=expected_label,
                            matches_expected=detection["label"]==expected_label))
    preview=Image.new("RGB",(1056,1096),"#242424")
    preview.paste(recognized,(0,0))
    ImageDraw.Draw(preview).text((10,1067),
        "Olivetti / AT&T Laboratories Cambridge | Montagem ilustrativa | Rotulos sao previsoes do modelo",
        fill="white",font=ImageFont.load_default(size=15))
    preview.save("evidencias/multiplas_faces.png")
    report={"type":"montagem ilustrativa, não foto de grupo natural",
            "source_split":"validação e calibração; não teste independente",
            "selection":"primeira foto de validação de cada uma das 8 classes, mais primeiro desconhecido da calibração",
            "source_indices":indices,"expected_subject_ids":[int(data.target[i]) for i in indices],
            "expected_faces":len(indices),"detected_faces":len(detections),"detections":matched}
    Path("evidencias/multiplas_faces.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    from sklearn.datasets import fetch_olivetti_faces
    from faces import FaceRecognizer
    data=fetch_olivetti_faces(data_home="cache")
    metrics=json.loads(Path("evidencias/metricas.json").read_text())
    system=FaceRecognizer("artefatos/classificador.keras","artefatos/metadata.json")
    report=create_demo(data,metrics["used_indices"],system)
    print("Faces detectadas:",report["detected_faces"])
    print("Rótulos:",[(d["label"],d["matches_expected"]) for d in report["detections"]])
