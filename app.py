import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st_stats # Renombrado para evitar conflicto con streamlit
import math as mt
import sys
import io # Importante para capturar el output de df.info()
from datetime import datetime
import networkx as nx
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import time
import os
import logging
from collections import Counter
import warnings

# Ignorar todos los warnings como en tu notebook
warnings.filterwarnings("ignore")

# --- Funciones Auxiliares de tu Notebook ---
# (Se mantienen las funciones que usas en el procesamiento)

def ClasificadorAcceso(dato):
    """
    Esta función verifica si un dato es una cadena de texto y si contiene la frase 'Open Access'.
    """
    if isinstance(dato, str):
        if 'Open Access' in dato:
            return True
        else:
            return False
    else:
        return False

def ContarAutores(dato):
    """
    Esta función cuenta el número de elementos en una lista.
    """
    if isinstance(dato, list):
        return len(dato)
    else:
        return 0

# --- Función de Carga de Datos ---
# Usamos un decorador de Streamlit para cachear los datos y que la app sea rápida.
# La app buscará el archivo "scopusffandhkorwtorhf.csv" en la misma carpeta.
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos '{file_path}'.")
        st.error("Por favor, asegúrate de subir este archivo a tu repositorio de GitHub junto con 'app.py'.")
        return None

# --- Comienzo de la Aplicación Streamlit ---

st.set_page_config(layout="wide")
st.title("📊 Análisis Interactivo de Publicaciones de Scopus")
st.write("Esta aplicación web muestra los resultados del análisis del cuaderno `BS04.ipynb`.")

# Define el nombre del archivo de datos
DATA_FILE = "scopusffandhkorwtorhf.csv"

# Carga los datos
dfScopus_raw = load_data(DATA_FILE)

# Solo si el archivo se cargó correctamente, continúa con el análisis
if dfScopus_raw is not None:
    # Hacemos una copia para no alterar el caché
    dfScopus = dfScopus_raw.copy()

    st.header("1. Vista Previa de los Datos Crudos")
    st.dataframe(dfScopus.head())

    # --- Procesamiento y Limpieza de Datos ---
    st.header("2. Procesamiento y Limpieza")
    st.write(f"Tamaño original de los datos: `{dfScopus.shape}`")

    # 1. Eliminar columnas
    eliminar = [
        'Author(s) ID', 'Volume', 'Issue', 'Art. No.', 'Page start',
        'Page end', 'Page count', 'DOI', 'Link', 'Source', 'EID'
    ]
    dfScopus = dfScopus.drop(columns=eliminar)
    st.write(f"Tamaño después de eliminar columnas: `{dfScopus.shape}`")

    # 2. Renombrar columnas
    newcols = {
        'Authors' : 'AUTORES', 'Author full names' : 'AUTORESCOMPLETOS',
        'Title' : 'TITULO', 'Year' : 'ANIO', 'Source title' : 'FUENTE',
        'Cited by' : 'CITACIONES', 'Abstract' : 'RESUMEN',
        'Author Keywords' : 'PCLAVEA', 'Index Keywords' : 'PCLAVEI',
        'Document Type' : 'TIPO', 'Publication Stage' : 'ESTADO',
        'Open Access' : 'ACCESO'
    }
    dfScopus.rename(columns=newcols, inplace=True)
    st.subheader("Datos con columnas renombradas (head)")
    st.dataframe(dfScopus.head(1))

    # 3. Información del DataFrame
    st.subheader("Información del DataFrame (df.info())")
    # Capturamos el output de df.info() para mostrarlo en Streamlit
    buffer = io.StringIO()
    dfScopus.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # 4. Ingeniería de Características
    with st.spinner("Realizando ingeniería de características..."):
        dfScopus['LISTAUTORES'] = dfScopus['AUTORES'].str.split('; ')
        dfScopus['ANIO'] = pd.to_numeric(dfScopus['ANIO'], errors='coerce')
        dfScopus['KEYWORDS'] = dfScopus['PCLAVEA'].fillna('') + '; ' + dfScopus['PCLAVEI'].fillna('')
        dfScopus['ALLKEYWORDS'] = dfScopus['KEYWORDS'].str.split('; ')
        dfScopus['OPENACCESS'] = dfScopus['ACCESO'].apply(ClasificadorAcceso)
        dfScopus['CANTIDADAUTORES'] = dfScopus['LISTAUTORES'].apply(ContarAutores)
    st.success("Ingeniería de características completada.")


    # --- Análisis Interactivo de Autores ---
    st.header("3. Análisis de Autores")

    # Contar la frecuencia de cada autor
    autores = dfScopus['LISTAUTORES'].explode()
    cuentauores = Counter(autores)
    st.write(f'El total de registros de autores analizados es **{len(autores)}**.')

    # --- ¡Aquí está la magia de Streamlit! ---
    # Reemplazamos el parámetro estático de Colab por un slider interactivo
    st.markdown("---")
    CantidadAutores = st.slider(
        "👇 Selecciona el número de autores a visualizar:",
        min_value=5,
        max_value=50,
        value=10,  # Valor por defecto
        step=5
    )

    # El gráfico se actualizará automáticamente cuando muevas el slider
    st.subheader(f"Top {CantidadAutores} Autores más Frecuentes")

    # Seleccionamos la cantidad de autores a visualizar
    top_autores = cuentauores.most_common(CantidadAutores)

    # Convertir a un DataFrame para facilitar la visualización
    top_autores_df = pd.DataFrame(top_autores, columns=['Author', 'Count'])

    # Crear la figura y el gráfico de barras horizontales
    # Es mejor práctica en Streamlit crear explícitamente la figura (fig) y los ejes (ax)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_autores_df['Author'], top_autores_df['Count'], color='skyblue')

    # Agregar etiquetas con los valores al final de cada barra
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{width}', ha='left', va='center')

    # Configurar etiquetas y título
    ax.set_xlabel('Número de Publicaciones')
    ax.set_ylabel('Autores')
    ax.set_title(f'Top {CantidadAutores} Autores más Frecuentes')
    ax.invert_yaxis() # Invertir el eje Y para que se muestre de mayor a menor
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Quitar el borde derecho y superior para un look más limpio
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Usar st.pyplot() para mostrar la figura de matplotlib
    st.pyplot(fig)

    st.markdown("---")
    st.header("4. Explorador de Datos Completo")
    st.write("Usa los filtros para explorar el dataset procesado.")
    st.dataframe(dfScopus)

else:
    st.warning("La aplicación no puede continuar porque el archivo de datos no se ha cargado.")
