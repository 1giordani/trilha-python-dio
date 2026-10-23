"""Interface local de demonstração: streamlit run app.py."""
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
from recommend import load_system,recommend

st.set_page_config(page_title="Busca visual de produtos",layout="wide")
st.title("Produtos parecidos pela imagem")
st.write("Escolha uma imagem ou envie um arquivo para buscar itens visualmente próximos no catálogo.")
st.caption("Piloto Fashion-MNIST: 10 categorias, 28×28 pixels e cores artificiais. Não é uma busca automática na internet.")
catalog_path=st.sidebar.text_input("Índice do catálogo",value="artefatos/catalogo.npz")


@st.cache_resource
def system(path):
    return load_system(catalog_path=path)


if not Path("artefatos/encoder.keras").is_file() or not Path(catalog_path).is_file():
    st.info("Execute python train.py ou extraia o pacote treinado na pasta do projeto para criar artefatos/.")
    st.stop()
encoder,catalog=system(catalog_path)
upload=st.file_uploader("Imagem de consulta",type=["png","jpg","jpeg","webp"])
invert=st.checkbox("Inverter cores da consulta (opcional)",value=False)
sample_index=st.slider("Ou escolha um item do catálogo",min_value=0,max_value=len(catalog["images"])-1,value=0)
image=Image.open(upload) if upload else Image.fromarray(catalog["images"][sample_index])
try:
    query,indices,scores,items=recommend(encoder,catalog,image,5,invert)
except ValueError as error:
    st.warning(str(error)); st.stop()
cols=st.columns(6)
with cols[0]:
    st.subheader("Consulta")
    st.image(query,width=140)
for column,item,index,score in zip(cols[1:],items,indices,scores):
    with column:
        st.image(catalog["images"][index],width=140)
        st.write(item["item_id"])
        st.caption(f"Similaridade cosseno: {float(score):.3f}")
st.caption("O ranking usa embedding CNN e histograma de cor; identificadores servem apenas para localizar os itens.")
st.info("O modelo não foi validado para objetos fora de moda nem para fotos arbitrárias de lojas. Similaridade não é probabilidade de compra ou garantia de equivalência.")
