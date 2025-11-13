# =====================================================================================================================
# REGION: Librerías
# =====================================================================================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st_stats # Renombrado para evitar conflicto
import math as mt
import sys
import io # Para capturar el output de df.info()
from datetime import datetime
from collections import Counter
import warnings
from wordcloud import WordCloud

# Ignorar todos los warnings como en tu notebook
warnings.filterwarnings("ignore")

# =====================================================================================================================
# REGION: Funciones Auxiliares del Notebook
# =====================================================================================================================

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

# =====================================================================================================================
# REGION: Carga de Datos (Cacheada por Streamlit)
# =====================================================================================================================

# Usamos un decorador de Streamlit para cachear los datos y que la app sea rápida.
# Esta función cargará el ARCHIVO DE EJEMPLO
@st.cache_data
def load_sample_data(file_path):
    """
    Carga los datos de EJEMPLO desde un archivo CSV.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos de ejemplo '{file_path}'.")
        st.error("Por favor, asegúrate de subir este archivo a tu repositorio de GitHub junto con 'app.py'.")
        return None

# =====================================================================================================================
# REGION: Procesamiento y Limpieza de Datos (Cacheado)
# =====================================================================================================================
@st.cache_data
def process_data(df_raw):
    """
    Aplica toda la limpieza y transformación al DataFrame.
    """
    # Verificamos que df_raw no sea None
    if df_raw is None:
        return None, 0, 0, 0
        
    dfScopus = df_raw.copy()

    # 1. Eliminar columnas
    eliminar = [
        'Author(s) ID', 'Volume', 'Issue', 'Art. No.', 'Page start',
        'Page end', 'Page count', 'DOI', 'Link', 'Source', 'EID'
    ]
    # Comprobar si las columnas existen antes de borrarlas
    eliminar_existentes = [col for col in eliminar if col in dfScopus.columns]
    dfScopus = dfScopus.drop(columns=eliminar_existentes)

    # 2. Renombrar columnas
    newcols = {
        'Authors' : 'AUTORES', 'Author full names' : 'AUTORESCOMPLETOS',
        'Title' : 'TITULO', 'Year' : 'ANIO', 'Source title' : 'FUENTE',
        'Cited by' : 'CITACIONES', 'Abstract' : 'RESUMEN',
        'Author Keywords' : 'PCLAVEA', 'Index Keywords' : 'PCLAVEI',
        'Document Type' : 'TIPO', 'Publication Stage' : 'ESTADO',
        'Open Access' : 'ACCESO'
    }
    # Renombrar solo las columnas que existen
    columnas_a_renombrar = {k: v for k, v in newcols.items() if k in dfScopus.columns}
    dfScopus.rename(columns=columnas_a_renombrar, inplace=True)

    # 3. Ingeniería de Características
    # Asegurarnos de que las columnas base existan antes de crear nuevas
    
    if 'AUTORES' in dfScopus.columns:
        dfScopus['LISTAUTORES'] = dfScopus['AUTORES'].str.split('; ')
        dfScopus['CANTIDADAUTORES'] = dfScopus['LISTAUTORES'].apply(ContarAutores)
    else:
        st.warning("Columna 'Authors' (AUTORES) no encontrada. Algunas gráficas de autores no funcionarán.")
        dfScopus['LISTAUTORES'] = [[]] * len(dfScopus)
        dfScopus['CANTIDADAUTORES'] = 0

    if 'AUTORESCOMPLETOS' in dfScopus.columns:
        dfScopus['LISTAUTORESCOMPLETOS'] = dfScopus['AUTORESCOMPLETOS'].str.split('; ')
    else:
        st.warning("Columna 'Author full names' (AUTORESCOMPLETOS) no encontrada.")
        dfScopus['LISTAUTORESCOMPLETOS'] = [[]] * len(dfScopus)

    if 'ANIO' in dfScopus.columns:
        dfScopus['ANIO'] = pd.to_numeric(dfScopus['ANIO'], errors='coerce')
    else:
        st.warning("Columna 'Year' (ANIO) no encontrada. Gráficas de año no funcionarán.")
        dfScopus['ANIO'] = np.nan

    # Combinar PCLAVEA y PCLAVEI
    if 'PCLAVEA' not in dfScopus.columns: dfScopus['PCLAVEA'] = ''
    if 'PCLAVEI' not in dfScopus.columns: dfScopus['PCLAVEI'] = ''
    
    dfScopus['KEYWORDS'] = dfScopus['PCLAVEA'].fillna('') + '; ' + dfScopus['PCLAVEI'].fillna('')
    dfScopus['ALLKEYWORDS'] = dfScopus['KEYWORDS'].str.split('; ')
    
    if 'ACCESO' in dfScopus.columns:
        dfScopus['OPENACCESS'] = dfScopus['ACCESO'].apply(ClasificadorAcceso)
    else:
        st.warning("Columna 'Open Access' (ACCESO) no encontrada.")
        dfScopus['OPENACCESS'] = False

    # Columna de Citaciones por año
    current_year = datetime.now().year
    if 'CITACIONES' in dfScopus.columns and 'ANIO' in dfScopus.columns:
        dfScopus['Citaciones por año'] = dfScopus['CITACIONES'] / (current_year + 1 - dfScopus['ANIO'])
        dfScopus['Citado'] = np.where(dfScopus['CITACIONES'] > 0, 'Si', 'No')
    else:
        st.warning("Columna 'Cited by' (CITACIONES) no encontrada. Análisis de citaciones no funcionará.")
        dfScopus['CITACIONES'] = 0
        dfScopus['Citaciones por año'] = 0
        dfScopus['Citado'] = 'No'

    # Columnas de Afiliaciones y País
    # --- ¡CORRECCIÓN! Usar 'df_raw' (el argumento) en lugar de la variable global ---
    if 'Affiliations' in df_raw.columns:
        dfScopus['Afilaciones'] = df_raw['Affiliations'].str.split('; ')
        dfScopus['pais'] = dfScopus['Afilaciones'].str[-1]
        dfScopus['Pais'] = dfScopus['pais'].str.split().str[-1]
        dfScopus['Pais'] = dfScopus['Pais'].replace('States', 'USA')
        dfScopus['Pais'] = dfScopus['Pais'].replace('Kingdom', 'United Kingdom')
    else:
        st.warning("Columna 'Affiliations' no encontrada. Gráficas de país e institución no funcionarán.")
        dfScopus['Afilaciones'] = [[]] * len(dfScopus)
        dfScopus['Pais'] = [None] * len(dfScopus)

    # Columna de Caracteres en Título
    if 'TITULO' in dfScopus.columns:
        dfScopus['CARACTERESTITULO'] = dfScopus['TITULO'].str.len()
    else:
        st.warning("Columna 'Title' (TITULO) no encontrada.")
        dfScopus['TITULO'] = ''
        dfScopus['CARACTERESTITULO'] = 0


    # 4. Limpieza de Keywords (reemplaza el bucle lento de tqdm)
    dfScopus['ALLKEYWORDS'] = dfScopus['ALLKEYWORDS'].apply(lambda keys: [k.strip() for k in keys if k and k.strip()])
    longitudactual = dfScopus.shape[0]
    dfScopus = dfScopus[dfScopus['ALLKEYWORDS'].map(len) > 0].copy()
    longitudnueva = dfScopus.shape[0]
    contadorborrado = longitudactual - longitudnueva
    
    return dfScopus, longitudactual, longitudnueva, contadorborrado

# =====================================================================================================================
# REGION: Configuración de la Página Streamlit
# =====================================================================================================================

st.set_page_config(page_title="Análisis de Scopus", layout="wide")
st.title("📊 Análisis Interactivo de Publicaciones de Scopus")
st.write("Esta aplicación analiza un conjunto de datos de Scopus. Puedes usar los datos de ejemplo o subir tu propio archivo CSV.")

# Define el nombre del archivo de datos de EJEMPLO
DATA_FILE = "scopusffandhkorwtorhf.csv"

# --- NUEVA SECCIÓN: Selección de Fuente de Datos ---
st.sidebar.header("Fuente de Datos")
data_source = st.sidebar.radio(
    "Elige una fuente de datos:",
    ("Usar datos de ejemplo", "Subir mi propio archivo CSV")
)

dfScopus_raw = None
uploaded_file = None

if data_source == "Usar datos de ejemplo":
    # Carga los datos de ejemplo
    dfScopus_raw = load_sample_data(DATA_FILE)
    
else:
    # --- AQUÍ COMIENZAN LOS CAMBIOS SOLICITADOS ---
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Rúbrica para la Base de Datos")
    st.sidebar.info(
        "**Importante:** Para que el análisis funcione, asegúrate de que tu archivo CSV "
        "cumpla con los requisitos mostrados en la siguiente imagen:"
    )
    
    # Mostramos la imagen indicando que son los requisitos/rúbrica
    # Asegúrate de que 'image_292efe.png' esté en la misma carpeta que tu script
    try:
        st.sidebar.image("image_292efe.png", caption="Figura 1. Campos requeridos en Scopus", use_container_width=True)
    except:
        st.sidebar.warning("No se pudo cargar la imagen de la rúbrica (image_292efe.png).")

    st.sidebar.markdown("---")
    
    # Muestra el widget para subir archivos
    uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV de Scopus:", type=["csv"])
    
    # --- FIN DE LOS CAMBIOS ---
    
    if uploaded_file is not None:
        try:
            # Lee el archivo CSV subido
            dfScopus_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("¡Archivo cargado exitosamente!")
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            st.warning("Asegúrate de que el archivo CSV tenga el formato correcto de Scopus.")


# Solo si el archivo se cargó correctamente (de ejemplo o subido), continúa con el análisis
if dfScopus_raw is not None:

    # --- Procesamiento de Datos ---
    with st.spinner("Procesando y limpiando los datos..."):
        dfScopus, longitudactual, longitudnueva, contadorborrado = process_data(dfScopus_raw)

    # --- Panel Lateral de Controles ---
    st.sidebar.header("Filtros Interactivos")

    CantidadAutores = st.sidebar.slider(
        "Top Autores Frecuentes:",
        min_value=5, max_value=50, value=10, step=5
    )
    
    CantidadPalabrasClave = st.sidebar.slider(
        "Top Palabras Clave Frecuentes:",
        min_value=5, max_value=50, value=20, step=5
    )

    CantidadFuentes = st.sidebar.slider(
        "Top Fuentes Comunes:",
        min_value=5, max_value=50, value=10, step=5
    )
    
    search_string = st.sidebar.text_input(
        "Buscar Autor (ej. Lauder G.):",
        "Lauder G."
    )

    # --- Resumen del Procesamiento ---
    with st.expander("Ver Resumen del Procesamiento de Datos"):
        st.subheader("Información del DataFrame (df.info())")
        buffer = io.StringIO()
        dfScopus.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("Limpieza de Palabras Clave")
        st.metric("Registros antes de limpiar 'ALLKEYWORDS'", longitudactual)
        st.metric("Registros después de limpiar 'ALLKEYWORDS'", longitudnueva)
        st.metric("Registros eliminados (sin palabras clave)", contadorborrado, delta_color="inverse")


    # --- Pestañas de Visualización ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Análisis de Autores", 
        "Análisis de Publicaciones", 
        "Análisis de Palabras Clave", 
        "Fuentes y Afiliación", 
        "Análisis de Citaciones",
        "Búsqueda y Rankings"
    ])

    # ==================================================================
    # PESTAÑA 1: ANÁLISIS DE AUTORES
    # ==================================================================
    with tab1:
        st.header("Análisis de Autores")
        
        # --- Gráfico 1: Top Autores (AUTORES) ---
        st.subheader(f"Top {CantidadAutores} Autores (Formato Corto)")
        
        try:
            autores = dfScopus['LISTAUTORES'].explode()
            if not autores.empty:
                cuentauores = Counter(autores)
                top_autores = cuentauores.most_common(CantidadAutores)
                top_autores_df = pd.DataFrame(top_autores, columns=['Author', 'Count'])

                # Crear la figura y los ejes
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                
                # Dibujar en los ejes (ax1)
                bars1 = ax1.barh(top_autores_df['Author'], top_autores_df['Count'], color='skyblue')
                ax1.set_xlabel('Número de Publicaciones')
                ax1.set_ylabel('Autores')
                ax1.set_title(f'Top {CantidadAutores} Autores más Frecuentes')
                ax1.invert_yaxis()
                ax1.grid(axis='x', linestyle='--', alpha=0.7)
                for bar in bars1:
                    width = bar.get_width()
                    ax1.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center')
                
                st.pyplot(fig1)
            else:
                st.warning("No hay datos de 'LISTAUTORES' para mostrar.")

        except Exception as e:
            st.error(f"Error al generar Gráfico 1 (Top Autores): {e}")


        # --- Gráfico 2: Top Autores (COMPLETOS) ---
        st.subheader(f"Top {CantidadAutores} Autores (Nombre Completo)")
        
        try:
            autorescompletos = dfScopus['LISTAUTORESCOMPLETOS'].explode()
            if not autorescompletos.empty:
                cuentaautorescompletos = Counter(autorescompletos)
                top_autores_completos = cuentaautorescompletos.most_common(CantidadAutores)
                top_autores_completos_df = pd.DataFrame(top_autores_completos, columns=['Author', 'Count'])

                # Crear la figura y los ejes
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # Dibujar en los ejes (ax2)
                bars2 = ax2.barh(top_autores_completos_df['Author'], top_autores_completos_df['Count'], color='lightgreen')
                ax2.set_xlabel('Número de Publicaciones')
                ax2.set_ylabel('Autores')
                ax2.set_title(f'Top {CantidadAutores} Autores (Nombres Completos)')
                ax2.invert_yaxis()
                ax2.grid(axis='x', linestyle='--', alpha=0.7)
                for bar in bars2:
                    width = bar.get_width()
                    ax2.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center')
                
                st.pyplot(fig2)
            else:
                st.warning("No hay datos de 'LISTAUTORESCOMPLETOS' para mostrar.")
                
        except Exception as e:
            st.error(f"Error al generar Gráfico 2 (Autores Completos): {e}")


        # --- Gráfico 3: Número de autores por publicacion ---
        st.subheader("Número de autores por publicación")
        
        try:
            # Asegurarse que 'CANTIDADAUTORES' existe
            if 'CANTIDADAUTORES' in dfScopus.columns:
                df3filtrado = dfScopus[dfScopus['CANTIDADAUTORES'] >= 1]
                if not df3filtrado.empty:
                    conteo_autores = df3filtrado['CANTIDADAUTORES'].value_counts().sort_index()
                    
                    # Crear la figura y los ejes
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    
                    # Dibujar en los ejes (ax3)
                    ax3.bar(conteo_autores.index, conteo_autores.values)
                    ax3.set_title('Número de autores por publicacion')
                    ax3.set_xlabel('Número de autores')
                    ax3.set_ylabel('Número de publicaciones')
                    ax3.grid(True)
                    
                    st.pyplot(fig3)
                else:
                    st.warning("No hay publicaciones con 1 o más autores.")
            else:
                st.warning("Columna 'CANTIDADAUTORES' no encontrada.")

        except Exception as e:
            st.error(f"Error al generar Gráfico 3 (Autores por Publicación): {e}")

    # ==================================================================
    # PESTAÑA 2: ANÁLISIS DE PUBLICACIONES
    # ==================================================================
    with tab2:
        st.header("Análisis de Publicaciones")

        # --- Gráfico 4: Publicaciones por año ---
        st.subheader("Distribución de publicaciones por año")
        try:
            pubanio = dfScopus.copy()
            pubporanio = pubanio['ANIO'].value_counts().sort_index()
            
            if not pubporanio.empty:
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                bars4 = pubporanio.plot(kind='bar', color='peru', ax=ax4)
                for bar in bars4.containers[0]:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
                ax4.set_xlabel('Año')
                ax4.set_ylabel('Número de publicaciones')
                
                st.pyplot(fig4)
            else:
                st.warning("No hay datos de 'ANIO' para mostrar.")

        except Exception as e:
            st.error(f"Error al generar Gráfico 4 (Publicaciones por Año): {e}")


        # --- Gráfico 5: Publicaciones acumuladas ---
        st.subheader("Publicaciones acumuladas por año")
        try:
            if not pubporanio.empty:
                df_acum = pubporanio.reset_index()
                df_acum.columns = ['ANIO', 'count']
                df_acum = df_acum.sort_values(by='ANIO')
                df_acum['Acumulado'] = df_acum['count'].cumsum()

                fig5, ax5 = plt.subplots(figsize=(12, 6))
                ax5.bar(df_acum['ANIO'], df_acum['count'], width=0.8, label='Publicaciones por año', color='lightblue')
                ax5.set_ylabel('Publicaciones por año')
                ax5.legend(loc='upper left')
                
                ax5b = ax5.twinx() # Eje Y secundario
                ax5b.plot(df_acum['ANIO'], df_acum['Acumulado'], label='Publicaciones Acumuladas', color='red', marker='o')
                ax5b.set_ylabel('Publicaciones acumuladas')
                ax5b.legend(loc='upper right')
                
                st.pyplot(fig5)
            else:
                st.warning("No hay datos de 'ANIO' para mostrar.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 5 (Publicaciones Acumuladas): {e}")


        # --- Gráfico 6: Tipo de documentos ---
        st.subheader("Distribución de Tipos de Documentos")
        try:
            if 'TIPO' in dfScopus.columns:
                document_type_counts = dfScopus['TIPO'].value_counts()
                
                if not document_type_counts.empty:
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    bars6 = document_type_counts.plot(kind='bar', color='lightblue', ax=ax6)
                    for bar in bars6.containers[0]:
                        height = bar.get_height()
                        ax6.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
                    ax6.set_xlabel('Tipo de Documento')
                    ax6.set_ylabel('Número de Publicaciones')
                    ax6.set_title('Distribución de Tipos de Documentos')
                    plt.xticks(rotation=45, ha='right')
                    
                    st.pyplot(fig6)
                else:
                    st.warning("No hay datos de 'TIPO' (Tipo de Documento) para mostrar.")
            else:
                st.warning("Columna 'TIPO' (Tipo de Documento) no encontrada.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 6 (Tipos de Documento): {e}")

        # --- Gráfico 7: Caracteres en Título ---
        st.subheader("Distribución de Caracteres en el Título")
        try:
            if 'CARACTERESTITULO' in dfScopus.columns and dfScopus['CARACTERESTITULO'].dropna().any():
                fig7, ax7 = plt.subplots(figsize=(10, 4))
                ax7.boxplot(dfScopus['CARACTERESTITULO'].dropna(), vert=False, flierprops=dict(markerfacecolor='r', marker='o'), showfliers=False)
                ax7.set_title('Distribución de Caracteres en el titulo')
                ax7.set_xlabel('Caracteres')
                ax7.grid(True)
                
                st.pyplot(fig7)
            else:
                st.warning("No hay datos de 'CARACTERESTITULO' para mostrar.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 7 (Caracteres en Título): {e}")


    # ==================================================================
    # PESTAÑA 3: ANÁLISIS DE PALABRAS CLAVE
    # ==================================================================
    with tab3:
        st.header("Análisis de Palabras Clave")
        
        try:
            # Contar keywords
            keywords_exploded = dfScopus['ALLKEYWORDS'].explode()
            keyword_counts = Counter(keywords_exploded)
            if '' in keyword_counts: del keyword_counts['']
            
            if keyword_counts:
                # --- Gráfico 8: Top Palabras Clave ---
                st.subheader(f"Top {CantidadPalabrasClave} Palabras clave más frecuentes")
                top_keywords = keyword_counts.most_common(CantidadPalabrasClave)
                top_keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])

                fig8, ax8 = plt.subplots(figsize=(10, 8))
                bars8 = ax8.barh(top_keywords_df['Keyword'], top_keywords_df['Count'], color='salmon')
                for bar in bars8:
                    width = bar.get_width()
                    ax8.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center')
                ax8.set_xlabel('Frecuencia')
                ax8.set_ylabel('Palabras clave')
                ax8.set_title(f'Top {CantidadPalabrasClave} Palabras clave más frecuentes')
                ax8.invert_yaxis()
                
                st.pyplot(fig8)
            else:
                st.warning("No se encontraron palabras clave.")

        except Exception as e:
            st.error(f"Error al generar Gráfico 8 (Top Keywords): {e}")


        # --- Gráfico 9: Nube de Palabras ---
        st.subheader("Nube de Palabras para Keywords")
        try:
            if keyword_counts:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_counts)
                fig9, ax9 = plt.subplots(figsize=(10, 6))
                ax9.imshow(wordcloud, interpolation='bilinear')
                ax9.axis('off')
                ax9.set_title('Nube de Palabras para Keywords')
                
                st.pyplot(fig9)
            else:
                st.warning("No hay suficientes palabras clave para generar una nube de palabras.")
        except Exception as e:
            st.error(f"Error al generar la nube de palabras: {e}")

        # --- Gráfico 10: Frecuencia de palabras en el titulo ---
        st.subheader("Frecuencia de palabras en el título (longitud > 3, frecuencia > 40)")
        try:
            if 'TITULO' in dfScopus.columns and dfScopus['TITULO'].dropna().any():
                palabras_titulo = dfScopus['TITULO'].dropna().str.lower().str.cat(sep=';').split(' ')
                cuenta_palabras_titulo = Counter(palabras_titulo)
                top_palabras_titulo = cuenta_palabras_titulo.most_common()
                palabras_titulo_df = pd.DataFrame(top_palabras_titulo, columns=['Palabra', 'Numero'])
                palabras_titulo_df['Longitud'] = palabras_titulo_df['Palabra'].str.len()
                palabras_titulo_largas_df = palabras_titulo_df[(palabras_titulo_df['Longitud'] > 3) & (palabras_titulo_df['Numero'] > 40)]
                # Eliminar 'for' 'and' 'the' 'with'
                palabras_comunes_a_quitar = ['from', 'with', 'research', 'analysis', 'using', 'based', 'model', 'control', 'between', 'study']
                palabras_titulo_largas_df = palabras_titulo_largas_df[~palabras_titulo_largas_df['Palabra'].isin(palabras_comunes_a_quitar)]

                if not palabras_titulo_largas_df.empty:
                    fig10, ax10 = plt.subplots(figsize=(10, 8))
                    bars10 = ax10.barh(palabras_titulo_largas_df['Palabra'], palabras_titulo_largas_df['Numero'], color='red')
                    ax10.set_xlabel('Frecuencia')
                    ax10.set_ylabel('Palabra en Título')
                    ax10.set_title('Palabras más frecuentes en Títulos')
                    ax10.invert_yaxis()
                    
                    st.pyplot(fig10)
                else:
                    st.warning("No se encontraron palabras frecuentes en títulos con los filtros aplicados.")
            else:
                st.warning("No hay datos de 'TITULO' para analizar.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 10 (Palabras en Título): {e}")


    # ==================================================================
    # PESTAÑA 4: FUENTES Y AFILIACIÓN
    # ==================================================================
    with tab4:
        st.header("Análisis de Fuentes y Afiliación")

        # --- Gráfico 11: Fuentes de publicación ---
        st.subheader(f"Top {CantidadFuentes} Fuentes de publicación más comunes")
        try:
            if 'FUENTE' in dfScopus.columns:
                source_counts = dfScopus['FUENTE'].value_counts().head(CantidadFuentes)
                
                if not source_counts.empty:
                    fig11, ax11 = plt.subplots(figsize=(10, 6))
                    bars11 = source_counts.plot(kind='bar', ax=ax11)
                    for bar in bars11.containers[0]:
                        height = bar.get_height()
                        ax11.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
                    shortened_labels = [' '.join(str(label).split()[:5]) for label in source_counts.index]
                    ax11.set_title(f'Top {CantidadFuentes} Fuentes de publicación más comunes')
                    ax11.set_xlabel('Fuente de publicación')
                    ax11.set_ylabel('Número de publicaciones')
                    ax11.set_xticklabels(shortened_labels, rotation=45, ha='right')
                    ax11.grid(True)
                    
                    st.pyplot(fig11)
                else:
                    st.warning("No hay datos de 'FUENTE' para mostrar.")
            else:
                st.warning("Columna 'FUENTE' (Source title) no encontrada.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 11 (Fuentes): {e}")

        
        # --- Gráfico 12: Instituciones ---
        st.subheader("Instituciones más Frecuentes")
        if 'Afilaciones' in dfScopus.columns:
            try:
                instituciones = dfScopus['Afilaciones'].explode()
                cuenta_instituciones = Counter(instituciones)
                top_instituciones = cuenta_instituciones.most_common(11)
                
                top_instituciones_df = pd.DataFrame(top_instituciones, columns=['Institución', 'Numero'])
                if not top_instituciones_df.empty:
                    # Intentar eliminar la primera fila si es un valor nulo/vacío
                    if pd.isna(top_instituciones_df.iloc[0, 0]) or top_instituciones_df.iloc[0, 0].strip() == '':
                            top_instituciones_df = top_instituciones_df.drop([0])
                    
                    fig12, ax12 = plt.subplots(figsize=(10, 6))
                    bars12 = ax12.barh(top_instituciones_df['Institución'], top_instituciones_df['Numero'], color='green')
                    for bar in bars12:
                        width = bar.get_width()
                        ax12.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center')
                    ax12.set_xlabel('Número de publicaciones')
                    ax12.set_ylabel('Institución')
                    ax12.set_title(f'Top 10 Instituciones más Frecuentes')
                    ax12.invert_yaxis()
                    
                    st.pyplot(fig12)
                else:
                    st.warning("No se encontraron datos de instituciones.")
            except Exception as e:
                st.error(f"Error al generar Gráfico 12 (Instituciones): {e}")

            # --- Gráfico 13: País ---
            st.subheader("País de publicación más frecuentes")
            try:
                pais = dfScopus['Pais'].explode().dropna()
                if not pais.empty:
                    cuenta_pais = Counter(pais)
                    top_pais = cuenta_pais.most_common(10)
                    df_pais = pd.DataFrame(top_pais, columns=['Pais', 'Número de publicaciones'])
                    
                    fig13, ax13 = plt.subplots(figsize=(10, 6))
                    bars13 = ax13.barh(df_pais['Pais'], df_pais['Número de publicaciones'], color='red')
                    for bar in bars13:
                        width = bar.get_width()
                        ax13.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center')
                    ax13.set_xlabel('Número de publicaciones')
                    ax13.set_ylabel('País')
                    ax13.set_title(f'Top 10 Países más frecuentes')
                    ax13.invert_yaxis()
                    
                    st.pyplot(fig13)
                else:
                    st.warning("No se pudieron extraer datos de 'Pais'.")
            except Exception as e:
                st.error(f"Error al generar Gráfico 13 (País): {e}")
        
        else:
            st.warning("La columna 'Affiliations' no se encontró, se omiten los gráficos de Institución y País.")


    # ==================================================================
    # PESTAÑA 5: ANÁLISIS DE CITACIONES
    # ==================================================================
    with tab5:
        st.header("Análisis de Citaciones")

        col1, col2 = st.columns(2)
        
        # --- Gráfico 14: Acceso Abierto ---
        with col1:
            st.subheader("Publicaciones de acceso abierto")
            try:
                open_access_counts = dfScopus['OPENACCESS'].value_counts()
                mylabels = ["No", "Si"]
                
                fig14, ax14 = plt.subplots(figsize=(6, 6))
                if not open_access_counts.empty:
                    ax14.pie(open_access_counts, autopct='%1.1f%%', colors=['skyblue', 'red'], startangle=90, labels=mylabels)
                    ax14.set_title('(a) Publicaciones de acceso abierto')
                    ax14.legend()
                
                st.pyplot(fig14)
            except Exception as e:
                st.error(f"Error al generar Gráfico 14 (Acceso Abierto): {e}")


        # --- Gráfico 15: Publicaciones con citaciones ---
        with col2:
            st.subheader("Publicaciones con citaciones")
            try:
                conteo_publicaciones_citadas = dfScopus['Citado'].value_counts()
                
                fig15, ax15 = plt.subplots(figsize=(6, 6))
                if not conteo_publicaciones_citadas.empty:
                    labels = conteo_publicaciones_citadas.keys()
                    ax15.pie(conteo_publicaciones_citadas, autopct='%1.1f%%', labels=labels)
                    ax15.set_title('(b) Publicaciones con citaciones')
                    ax15.legend()
                
                st.pyplot(fig15)
            except Exception as e:
                st.error(f"Error al generar Gráfico 15 (Con Citaciones): {e}")

        st.divider()

        # --- Gráfico 16: Distribución de Citaciones (Log) ---
        st.subheader("Distribución de Citaciones (Escala Logarítmica)")
        try:
            if 'CITACIONES' in dfScopus.columns:
                citation_counts = dfScopus['CITACIONES'].value_counts().sort_index()
                if not citation_counts.empty:
                    fig16, ax16 = plt.subplots(figsize=(10, 6))
                    ax16.bar(citation_counts.index, citation_counts.values)
                    ax16.set_title('Distribución de Citaciones')
                    ax16.set_xlabel('Número de Citaciones')
                    ax16.set_ylabel('Frecuencia')
                    ax16.grid(True)
                    ax16.set_xscale('log') # Escala logarítmica
                    
                    st.pyplot(fig16)
                else:
                    st.warning("No hay datos de 'CITACIONES' para mostrar.")
            else:
                st.warning("Columna 'CITACIONES' no encontrada.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 16 (Dist. Citaciones): {e}")

        # --- Gráfico 17: Boxplot Citaciones (Todas vs Review) ---
        st.subheader("Boxplot de Citaciones (Todas vs. Revisión)")
        try:
            if 'TIPO' in dfScopus.columns and 'CITACIONES' in dfScopus.columns:
                revision = dfScopus[dfScopus['TIPO'] == 'Review']
                
                fig17, ax17 = plt.subplots(figsize=(10, 6))
                datos = [dfScopus['CITACIONES'].dropna(), revision['CITACIONES'].dropna()]
                mylabels = ["Todas las publicaciones", "Publicaciones de revisión"]
                bp = ax17.boxplot(datos, labels=mylabels, showfliers=False) # showfliers=False para imitar el notebook
                
                # Extraer medianas y etiquetarlas
                medians = bp['medians']
                for i, median in enumerate(medians):
                    # Asegurarse de que hay datos en la mediana
                    if len(median.get_ydata()) > 0:
                        value = median.get_ydata()[0]
                        ax17.text(i + 1, value, f'{value:.2f}', ha='center', va='bottom', color='black')
                
                ax17.set_ylabel('Número de Citaciones')
                ax17.grid(True)
                
                st.pyplot(fig17)
            else:
                st.warning("Columnas 'TIPO' o 'CITACIONES' no encontradas.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 17 (Boxplot Citaciones): {e}")


        # --- Gráfico 18: Boxplot Citaciones por Año ---
        st.subheader("Boxplot de Citaciones por Año")
        try:
            if 'Citaciones por año' in dfScopus.columns and dfScopus['Citaciones por año'].dropna().any():
                fig18, ax18 = plt.subplots(figsize=(10, 6))
                datos_cpa = [dfScopus['Citaciones por año'].dropna()]
                mylabels_cpa = ["Todas las publicaciones"]
                bp_cpa = ax18.boxplot(datos_cpa, labels=mylabels_cpa, showfliers=False)
                
                medians_cpa = bp_cpa['medians']
                if medians_cpa:
                    for i, median in enumerate(medians_cpa):
                        if len(median.get_ydata()) > 0:
                            value = median.get_ydata()[0]
                            ax18.text(i + 1, value, f'{value:.2f}', ha='center', va='bottom', color='black')
                
                ax18.set_ylabel('Número de Citaciones por Año')
                ax18.grid(True)
                
                st.pyplot(fig18)
            else:
                st.warning("No hay datos de 'Citaciones por año' para mostrar.")
        except Exception as e:
            st.error(f"Error al generar Gráfico 18 (Boxplot Cit./Año): {e}")


    # ==================================================================
    # PESTAÑA 6: BÚSQUEDA Y RANKINGS
    # ==================================================================
    with tab6:
        st.header("Búsqueda y Rankings de Publicaciones")

        # --- Búsqueda por Autor ---
        st.subheader(f"Resultados de búsqueda para: '{search_string}'")
        if search_string:
            try:
                if 'AUTORES' in dfScopus.columns:
                    results = dfScopus[dfScopus['AUTORES'].str.contains(search_string, na=False)]
                    columnas_a_extraer = ['TITULO', 'AUTORES', 'CITACIONES']
                    # Asegurarse que las columnas existen
                    columnas_existentes = [col for col in columnas_a_extraer if col in results.columns]
                    results1 = results[columnas_existentes]
                    
                    if 'CITACIONES' in results1.columns:
                        results2 = results1.sort_values(by='CITACIONES', ascending=False)
                    else:
                        results2 = results1
                        
                    st.dataframe(results2.head(30))
                else:
                    st.warning("Columna 'AUTORES' no encontrada para realizar la búsqueda.")
            except Exception as e:
                st.error(f"Error al buscar autor: {e}")
        else:
            st.info("Escribe un nombre de autor en el panel lateral izquierdo para buscar.")

        st.divider()
        
        col_rank1, col_rank2, col_rank3 = st.columns(3)

        # --- Ranking 1: Mayor Impacto (General) ---
        with col_rank1:
            st.subheader("Top 10 Mayor Impacto (General)")
            try:
                if 'CITACIONES' in dfScopus.columns:
                    impacto_cols = ['TITULO', 'AUTORES', 'CITACIONES']
                    impacto_cols_ex = [col for col in impacto_cols if col in dfScopus.columns]
                    impacto = dfScopus.sort_values(by='CITACIONES', ascending=False)[impacto_cols_ex]
                    st.dataframe(impacto.head(10))
                else:
                    st.warning("Columna 'CITACIONES' no disponible.")
            except Exception as e:
                st.error(f"Error en Ranking 1: {e}")

        # --- Ranking 2: Mayor Impacto (Review) ---
        with col_rank2:
            st.subheader("Top 10 Mayor Impacto (Review)")
            try:
                if 'TIPO' in dfScopus.columns and 'CITACIONES' in dfScopus.columns:
                    revision_rank = dfScopus[dfScopus['TIPO'] == 'Review']
                    impacto_cols = ['TITULO', 'AUTORES', 'CITACIONES']
                    impacto_cols_ex = [col for col in impacto_cols if col in revision_rank.columns]
                    impactorevision = revision_rank.sort_values(by='CITACIONES', ascending=False)[impacto_cols_ex]
                    st.dataframe(impactorevision.head(10))
                else:
                    st.warning("Columnas 'TIPO' o 'CITACIONES' no disponibles.")
            except Exception as e:
                st.error(f"Error en Ranking 2: {e}")

        # --- Ranking 3: Mayor Impacto (Citaciones por Año) ---
        with col_rank3:
            st.subheader("Top 10 Mayor Impacto (Cit./Año)")
            try:
                if 'Citaciones por año' in dfScopus.columns:
                    impacto_cols = ['TITULO', 'AUTORES', 'Citaciones por año']
                    impacto_cols_ex = [col for col in impacto_cols if col in dfScopus.columns]
                    impacto_cpa = dfScopus.sort_values(by='Citaciones por año', ascending=False)[impacto_cols_ex]
                    st.dataframe(impacto_cpa.head(10))
                else:
                    st.warning("Columna 'Citaciones por año' no disponible.")
            except Exception as e:
                st.error(f"Error en Ranking 3: {e}")

    # --- Expander para ver datos completos ---
    with st.expander("Ver DataFrame Procesado Completo"):
        st.dataframe(dfScopus)

else:
    st.info("Por favor, selecciona 'Usar datos de ejemplo' o sube tu propio archivo CSV para comenzar el análisis.")
