"""Comparação visual entre a consulta e os produtos recomendados."""
from PIL import Image, ImageDraw, ImageFont


def recommendation_grid(queries,catalog,ranks,scores,path,labels=None):
    tile,pad,header=112,12,44
    columns=1+len(ranks[0]); rows=len(queries)
    canvas=Image.new("RGB",(columns*(tile+pad)+pad,rows*(tile+36)+header+30),"#20242b")
    draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(size=13)
    draw.text((pad,12),"Consulta | Recomendacoes por aparencia (score cosseno)",fill="white",font=font)
    for row,query in enumerate(queries):
        images=[query]+[catalog[i] for i in ranks[row]]
        for col,image in enumerate(images):
            x,y=pad+col*(tile+pad),header+row*(tile+36)
            canvas.paste(Image.fromarray(image).resize((tile,tile),Image.Resampling.NEAREST),(x,y))
            text=(labels[row] if labels else "consulta") if col==0 else f"#{col}  {scores[row][col-1]:.3f}"
            draw.text((x,y+tile+5),text,fill="#ffd36a" if col==0 else "white",font=font)
    draw.text((pad,canvas.height-22),"Fashion-MNIST / Zalando Research | Cores artificiais | Exemplo de laboratorio",fill="white",font=font)
    canvas.save(path)
