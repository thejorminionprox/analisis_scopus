import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st_stats
import math as mt
import sys
import io
from datetime import datetime
from collections import Counter
import warnings
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

def ClasificadorAcceso(dato):
    if isinstance(dato, str):
        if 'Open Access' in dato:
            return True
        else:
            return False
    else:
        return False

def ContarAutores(dato):
    if isinstance(dato, list):
        return len(dato)
    else:
        return 0

@st.cache_data
def load_sample_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'.")
        return None

@st.cache_data
def process_data(df_raw):
    if df_raw is None:
        return None, 0, 0, 0
        
    dfScopus = df_raw.copy()

    eliminar = [
        'Author(s) ID', 'Volume', 'Issue', 'Art. No.', 'Page start',
        'Page end', 'Page count', 'DOI', 'Link', 'Source', 'EID'
    ]
    eliminar_existentes = [col for col in eliminar if col in dfScopus.columns]
    dfScopus = dfScopus.drop(columns=eliminar_existentes)

    newcols = {
        'Authors' : 'AUTORES', 'Author full names' : 'AUTORESCOMPLETOS',
        'Title' : 'TITULO', 'Year' : 'ANIO', 'Source title' : 'FUENTE',
        'Cited by' : 'CITACIONES', 'Abstract' : 'RESUMEN',
        'Author Keywords' : 'PCLAVEA', 'Index Keywords' : 'PCLAVEI',
        'Document Type' : 'TIPO', 'Publication Stage' : 'ESTADO',
        'Open Access' : 'ACCESO'
    }
    columnas_a_renombrar = {k: v for k, v in newcols.items() if k in dfScopus.columns}
    dfScopus.rename(columns=columnas_a_renombrar, inplace=True)

    if 'AUTORES' in dfScopus.columns:
        dfScopus['LISTAUTORES'] = dfScopus['AUTORES'].str.split('; ')
        dfScopus['CANTIDADAUTORES'] = dfScopus['LISTAUTORES'].apply(ContarAutores)
    else:
        st.warning("Columna 'Authors' no encontrada.")
        dfScopus['LISTAUTORES'] = [[]] * len(dfScopus)
        dfScopus['CANTIDADAUTORES'] = 0

    if 'AUTORESCOMPLETOS' in dfScopus.columns:
        dfScopus['LISTAUTORESCOMPLETOS'] = dfScopus['AUTORESCOMPLETOS'].str.split('; ')
    else:
        dfScopus['LISTAUTORESCOMPLETOS'] = [[]] * len(dfScopus)

    if 'ANIO' in dfScopus.columns:
        dfScopus['ANIO'] = pd.to_numeric(dfScopus['ANIO'], errors='coerce')
    else:
        st.warning("Columna 'Year' no encontrada.")
        dfScopus['ANIO'] = np.nan

    if 'PCLAVEA' not in dfScopus.columns: dfScopus['PCLAVEA'] = ''
    if 'PCLAVEI' not in dfScopus.columns: dfScopus['PCLAVEI'] = ''
    
    dfScopus['KEYWORDS'] = dfScopus['PCLAVEA'].fillna('') + '; ' + dfScopus['PCLAVEI'].fillna('')
    dfScopus['ALLKEYWORDS'] = dfScopus['KEYWORDS'].str.split('; ')
    
    if 'ACCESO' in dfScopus.columns:
        dfScopus['OPENACCESS'] = dfScopus['ACCESO'].apply(ClasificadorAcceso)
    else:
        dfScopus['OPENACCESS'] = False

    current_year = datetime.now().year
    if 'CITACIONES' in dfScopus.columns and 'ANIO' in dfScopus.columns:
        dfScopus['Citaciones por año'] = dfScopus['CITACIONES'] / (current_year + 1 - dfScopus['ANIO'])
        dfScopus['Citado'] = np.where(dfScopus['CITACIONES'] > 0, 'Si', 'No')
    else:
        dfScopus['CITACIONES'] = 0
        dfScopus['Citaciones por año'] = 0
        dfScopus['Citado'] = 'No'

    if 'Affiliations' in df_raw.columns:
        dfScopus['Afilaciones'] = df_raw['Affiliations'].str.split('; ')
        dfScopus['pais'] = dfScopus['Afilaciones'].str[-1]
        dfScopus['Pais'] = dfScopus['pais'].str.split().str[-1]
        dfScopus['Pais'] = dfScopus['Pais'].replace('States', 'USA')
        dfScopus['Pais'] = dfScopus['Pais'].replace('Kingdom', 'United Kingdom')
    else:
        st.warning("Columna 'Affiliations' no encontrada.")
        dfScopus['Afilaciones'] = [[]] * len(dfScopus)
        dfScopus['Pais'] = [None] * len(dfScopus)

    if 'TITULO' in dfScopus.columns:
        dfScopus['CARACTERESTITULO'] = dfScopus['TITULO'].str.len()
    else:
        dfScopus['TITULO'] = ''
        dfScopus['CARACTERESTITULO'] = 0

    dfScopus['ALLKEYWORDS'] = dfScopus['ALLKEYWORDS'].apply(lambda keys: [k.strip() for k in keys if k and k.strip()])
    longitudactual = dfScopus.shape[0]
    dfScopus = dfScopus[dfScopus['ALLKEYWORDS'].map(len) > 0].copy()
    longitudnueva = dfScopus.shape[0]
    contadorborrado = longitudactual - longitudnueva
    
    return dfScopus, longitudactual, longitudnueva, contadorborrado

st.set_page_config(page_title="Análisis de Scopus", layout="wide")
st.title("📊 Análisis Interactivo de Publicaciones de Scopus")
st.write("Esta aplicación analiza un conjunto de datos de Scopus.")

DATA_FILE = "scopusffandhkorwtorhf.csv"

st.sidebar.header("Fuente de Datos")
data_source = st.sidebar.radio(
    "Elige una fuente de datos:",
    ("Usar datos de ejemplo", "Subir mi propio archivo CSV")
)

dfScopus_raw = None
uploaded_file = None

if data_source == "Usar datos de ejemplo":
    dfScopus_raw = load_sample_data(DATA_FILE)
    
else:
    # --- CAMBIO: Imagen y descripción en el CENTRO de la página ---
    st.info(
        "ℹ️ **Instrucciones:** Para utilizar tu propia base de datos, el archivo CSV debe tener una estructura específica. "
        "La siguiente imagen muestra los campos (columnas) requeridos para que el análisis funcione correctamente."
    )
    
    try:
        # Se muestra la imagen en el contenedor principal
        st.image("image_292efe.png", caption="Rúbrica: Campos necesarios en la base de datos de Scopus", use_container_width=True)
    except:
        st.warning("No se pudo cargar la imagen de referencia.")
    
    st.markdown("---")
    # --------------------------------------------------------------

    uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV de Scopus:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            dfScopus_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("¡Archivo cargado exitosamente!")
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")

if dfScopus_raw is not None:
    with st.spinner("Procesando datos..."):
        dfScopus, longitudactual, longitudnueva, contadorborrado = process_data(dfScopus_raw)

    st.sidebar.header("Filtros Interactivos")

    CantidadAutores = st.sidebar.slider("Top Autores Frecuentes:", 5, 50, 10, 5)
    CantidadPalabrasClave = st.sidebar.slider("Top Palabras Clave Frecuentes:", 5, 50, 20, 5)
    CantidadFuentes = st.sidebar.slider("Top Fuentes Comunes:", 5, 50, 10, 5)
    search_string = st.sidebar.text_input("Buscar Autor (ej. Lauder G.):", "Lauder G.")

    with st.expander("Ver Resumen del Procesamiento"):
        st.subheader("Información del DataFrame")
        buffer = io.StringIO()
        dfScopus.info(buf=buffer)
        st.text(buffer.getvalue())
        st.subheader("Limpieza de Palabras Clave")
        st.metric("Registros antes", longitudactual)
        st.metric("Registros después", longitudnueva)
        st.metric("Eliminados", contadorborrado, delta_color="inverse")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Análisis de Autores", 
        "Análisis de Publicaciones", 
        "Análisis de Palabras Clave", 
        "Fuentes y Afiliación", 
        "Análisis de Citaciones",
        "Búsqueda y Rankings"
    ])

    with tab1:
        st.header("Análisis de Autores")
        st.subheader(f"Top {CantidadAutores} Autores (Formato Corto)")
        try:
            autores = dfScopus['LISTAUTORES'].explode()
            if not autores.empty:
                cuentauores = Counter(autores)
                top_autores_df = pd.DataFrame(cuentauores.most_common(CantidadAutores), columns=['Author', 'Count'])
                
                # --- MODIFICACIÓN ---
                # Altura dinámica: 0.4 pulgadas por barra, con un mínimo de 6
                altura_dinamica_1 = max(6, len(top_autores_df) * 0.4)
                fig1, ax1 = plt.subplots(figsize=(10, altura_dinamica_1))
                # --- FIN MODIFICACIÓN ---
                
                bars1 = ax1.barh(top_autores_df['Author'], top_autores_df['Count'], color='skyblue')
                ax1.set_title(f'Top {CantidadAutores} Autores más Frecuentes')
                ax1.invert_yaxis()
                for bar in bars1:
                    ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                st.pyplot(fig1)
            else:
                st.warning("No hay datos.")
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(f"Top {CantidadAutores} Autores (Nombre Completo)")
        try:
            autorescompletos = dfScopus['LISTAUTORESCOMPLETOS'].explode()
            if not autorescompletos.empty:
                cuentaautorescompletos = Counter(autorescompletos)
                top_autores_completos_df = pd.DataFrame(cuentaautorescompletos.most_common(CantidadAutores), columns=['Author', 'Count'])
                
                # --- MODIFICACIÓN ---
                altura_dinamica_2 = max(6, len(top_autores_completos_df) * 0.4)
                fig2, ax2 = plt.subplots(figsize=(10, altura_dinamica_2))
                # --- FIN MODIFICACIÓN ---
                
                bars2 = ax2.barh(top_autores_completos_df['Author'], top_autores_completos_df['Count'], color='lightgreen')
                ax2.set_title(f'Top {CantidadAutores} Autores (Nombres Completos)')
                ax2.invert_yaxis()
                for bar in bars2:
                    ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Número de autores por publicación")
        try:
            if 'CANTIDADAUTORES' in dfScopus.columns:
                df3filtrado = dfScopus[dfScopus['CANTIDADAUTORES'] >= 1]
                if not df3filtrado.empty:
                    conteo_autores = df3filtrado['CANTIDADAUTORES'].value_counts().sort_index()
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.bar(conteo_autores.index, conteo_autores.values)
                    ax3.set_title('Número de autores por publicación')
                    ax3.set_xlabel('Número de autores')
                    ax3.set_ylabel('Publicaciones')
                    st.pyplot(fig3)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab2:
        st.header("Análisis de Publicaciones")
        st.subheader("Distribución de publicaciones por año")
        try:
            pubporanio = dfScopus['ANIO'].value_counts().sort_index()
            if not pubporanio.empty:
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                bars4 = pubporanio.plot(kind='bar', color='peru', ax=ax4)
                for bar in bars4.containers[0]:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
                
                # --- MODIFICACIÓN ---
                plt.xticks(rotation=45, ha='right') # Rotar etiquetas
                plt.tight_layout() # Ajustar para que no se corten
                # --- FIN MODIFICACIÓN ---
                
                st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Publicaciones acumuladas por año")
        try:
            if not pubporanio.empty:
                df_acum = pubporanio.reset_index()
                df_acum.columns = ['ANIO', 'count']
                df_acum = df_acum.sort_values(by='ANIO')
                df_acum['Acumulado'] = df_acum['count'].cumsum()
                fig5, ax5 = plt.subplots(figsize=(12, 6))
                ax5.bar(df_acum['ANIO'], df_acum['count'], color='lightblue', label='Por año')
                ax5b = ax5.twinx()
                ax5b.plot(df_acum['ANIO'], df_acum['Acumulado'], color='red', marker='o', label='Acumulado')
                ax5.legend(loc='upper left')
                ax5b.legend(loc='upper right')
                st.pyplot(fig5)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Distribución de Tipos de Documentos")
        try:
            if 'TIPO' in dfScopus.columns:
                document_type_counts = dfScopus['TIPO'].value_counts()
                if not document_type_counts.empty:
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    bars6 = document_type_counts.plot(kind='bar', color='lightblue', ax=ax6)
                    for bar in bars6.containers[0]:
                        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
                    plt.xticks(rotation=45, ha='right')
                    
                    # --- MODIFICACIÓN ---
                    plt.tight_layout() # Ajustar para que no se corten
                    # --- FIN MODIFICACIÓN ---
                    
                    st.pyplot(fig6)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Distribución de Caracteres en el Título")
        try:
            if 'CARACTERESTITULO' in dfScopus.columns:
                fig7, ax7 = plt.subplots(figsize=(10, 4))
                ax7.boxplot(dfScopus['CARACTERESTITULO'].dropna(), vert=False, showfliers=False)
                st.pyplot(fig7)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab3:
        st.header("Análisis de Palabras Clave")
        try:
            keywords_exploded = dfScopus['ALLKEYWORDS'].explode()
            keyword_counts = Counter(keywords_exploded)
            if '' in keyword_counts: del keyword_counts['']
            
            if keyword_counts:
                st.subheader(f"Top {CantidadPalabrasClave} Palabras clave")
                top_keywords_df = pd.DataFrame(keyword_counts.most_common(CantidadPalabrasClave), columns=['Keyword', 'Count'])
                
                # --- MODIFICACIÓN ---
                altura_dinamica_8 = max(8, len(top_keywords_df) * 0.4)
                fig8, ax8 = plt.subplots(figsize=(10, altura_dinamica_8))
                # --- FIN MODIFICACIÓN ---
                
                bars8 = ax8.barh(top_keywords_df['Keyword'], top_keywords_df['Count'], color='salmon')
                ax8.invert_yaxis()
                for bar in bars8:
                    ax8.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                st.pyplot(fig8)

                st.subheader("Nube de Palabras")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_counts)
                fig9, ax9 = plt.subplots(figsize=(10, 6))
                ax9.imshow(wordcloud, interpolation='bilinear')
                ax9.axis('off')
                st.pyplot(fig9)
            else:
                st.warning("No hay palabras clave.")
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Frecuencia de palabras en el título")
        try:
            if 'TITULO' in dfScopus.columns:
                palabras_titulo = dfScopus['TITULO'].dropna().str.lower().str.cat(sep=';').split(' ')
                cuenta_palabras_titulo = Counter(palabras_titulo)
                palabras_df = pd.DataFrame(cuenta_palabras_titulo.most_common(), columns=['Palabra', 'Numero'])
                palabras_df = palabras_df[(palabras_df['Palabra'].str.len() > 3) & (palabras_df['Numero'] > 40)]
                stop_words = ['from', 'with', 'research', 'analysis', 'using', 'based', 'model', 'control', 'between', 'study']
                palabras_df = palabras_df[~palabras_df['Palabra'].isin(stop_words)]

                if not palabras_df.empty:
                    
                    # --- MODIFICACIÓN ---
                    # Esta es la gráfica de la imagen. Hacemos la altura dinámica.
                    # 0.35 pulgadas por palabra, con un mínimo de 8.
                    altura_dinamica_10 = max(8, len(palabras_df) * 0.35)
                    fig10, ax10 = plt.subplots(figsize=(10, altura_dinamica_10))
                    # --- FIN MODIFICACIÓN ---
                    
                    bars10 = ax10.barh(palabras_df['Palabra'], palabras_df['Numero'], color='red')
                    ax10.invert_yaxis()
                    st.pyplot(fig10)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab4:
        st.header("Análisis de Fuentes y Afiliación")
        st.subheader(f"Top {CantidadFuentes} Fuentes de publicación")
        try:
            if 'FUENTE' in dfScopus.columns:
                source_counts = dfScopus['FUENTE'].value_counts().head(CantidadFuentes)
                if not source_counts.empty:
                    fig11, ax11 = plt.subplots(figsize=(10, 6))
                    bars11 = source_counts.plot(kind='bar', ax=ax11)
                    for bar in bars11.containers[0]:
                        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
                    labels = [' '.join(str(l).split()[:5]) for l in source_counts.index]
                    ax11.set_xticklabels(labels, rotation=45, ha='right')
                    
                    # --- MODIFICACIÓN ---
                    plt.tight_layout() # Ajustar para que no se corten
                    # --- FIN MODIFICACIÓN ---
                    
                    st.pyplot(fig11)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Instituciones más Frecuentes")
        if 'Afilaciones' in dfScopus.columns:
            try:
                instituciones = dfScopus['Afilaciones'].explode()
                top_inst = pd.DataFrame(Counter(instituciones).most_common(11), columns=['Institución', 'Numero'])
                if not top_inst.empty:
                    if pd.isna(top_inst.iloc[0, 0]) or top_inst.iloc[0, 0].strip() == '': top_inst = top_inst.drop([0])
                    
                    # --- MODIFICACIÓN ---
                    altura_dinamica_12 = max(6, len(top_inst) * 0.4)
                    fig12, ax12 = plt.subplots(figsize=(10, altura_dinamica_12))
                    # --- FIN MODIFICACIÓN ---
                    
                    bars12 = ax12.barh(top_inst['Institución'], top_inst['Numero'], color='green')
                    ax12.invert_yaxis()
                    for bar in bars12:
                        ax12.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                    st.pyplot(fig12)
            except Exception as e:
                st.error(f"Error: {e}")

            st.subheader("Países más frecuentes")
            try:
                pais = dfScopus['Pais'].explode().dropna()
                if not pais.empty:
                    top_pais = pd.DataFrame(Counter(pais).most_common(10), columns=['Pais', 'Numero'])
                    
                    # --- MODIFICACIÓN ---
                    altura_dinamica_13 = max(6, len(top_pais) * 0.4)
                    fig13, ax13 = plt.subplots(figsize=(10, altura_dinamica_13))
                    # --- FIN MODIFICACIÓN ---
                    
                    bars13 = ax13.barh(top_pais['Pais'], top_pais['Numero'], color='red')
                    ax13.invert_yaxis()
                    for bar in bars13:
                        ax13.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                    st.pyplot(fig13)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Sin datos de afiliación.")

    with tab5:
        st.header("Análisis de Citaciones")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Acceso Abierto")
            try:
                oa_counts = dfScopus['OPENACCESS'].value_counts()
                fig14, ax14 = plt.subplots(figsize=(6, 6))
                if not oa_counts.empty:
                    ax14.pie(oa_counts, autopct='%1.1f%%', colors=['skyblue', 'red'], startangle=90, labels=["No", "Si"])
                    st.pyplot(fig14)
            except Exception as e:
                st.error(f"Error: {e}")

        with col2:
            st.subheader("Publicaciones con citaciones")
            try:
                cit_counts = dfScopus['Citado'].value_counts()
                fig15, ax15 = plt.subplots(figsize=(6, 6))
                if not cit_counts.empty:
                    ax15.pie(cit_counts, autopct='%1.1f%%', labels=cit_counts.keys())
                    st.pyplot(fig15)
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader("Distribución de Citaciones (Log)")
        try:
            if 'CITACIONES' in dfScopus.columns:
                cits = dfScopus['CITACIONES'].value_counts().sort_index()
                fig16, ax16 = plt.subplots(figsize=(10, 6))
                ax16.bar(cits.index, cits.values)
                ax16.set_xscale('log')
                st.pyplot(fig16)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Boxplot Citaciones (Todas vs Review)")
        try:
            if 'TIPO' in dfScopus.columns and 'CITACIONES' in dfScopus.columns:
                rev = dfScopus[dfScopus['TIPO'] == 'Review']
                fig17, ax17 = plt.subplots(figsize=(10, 6))
                datos = [dfScopus['CITACIONES'].dropna(), rev['CITACIONES'].dropna()]
                bp = ax17.boxplot(datos, labels=["Todas", "Review"], showfliers=False)
                for i, m in enumerate(bp['medians']):
                    if len(m.get_ydata()) > 0: ax17.text(i+1, m.get_ydata()[0], f'{m.get_ydata()[0]:.2f}', ha='center', va='bottom')
                st.pyplot(fig17)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Boxplot Citaciones por Año")
        try:
            if 'Citaciones por año' in dfScopus.columns:
                fig18, ax18 = plt.subplots(figsize=(10, 6))
                datos = [dfScopus['Citaciones por año'].dropna()]
                bp = ax18.boxplot(datos, labels=["Todas"], showfliers=False)
                for i, m in enumerate(bp['medians']):
                    if len(m.get_ydata()) > 0: ax18.text(i+1, m.get_ydata()[0], f'{m.get_ydata()[0]:.2f}', ha='center', va='bottom')
                st.pyplot(fig18)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab6:
        st.header("Búsqueda y Rankings")
        st.subheader(f"Búsqueda: '{search_string}'")
        if search_string and 'AUTORES' in dfScopus.columns:
            res = dfScopus[dfScopus['AUTORES'].str.contains(search_string, na=False)]
            cols = [c for c in ['TITULO', 'AUTORES', 'CITACIONES'] if c in res.columns]
            st.dataframe(res[cols].sort_values(by='CITACIONES', ascending=False).head(30) if 'CITACIONES' in cols else res[cols].head(30))

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Top Impacto (General)")
            if 'CITACIONES' in dfScopus.columns:
                st.dataframe(dfScopus.sort_values(by='CITACIONES', ascending=False)[['TITULO', 'CITACIONES']].head(10))
        with c2:
            st.subheader("Top Impacto (Review)")
            if 'TIPO' in dfScopus.columns and 'CITACIONES' in dfScopus.columns:
                st.dataframe(dfScopus[dfScopus['TIPO'] == 'Review'].sort_values(by='CITACIONES', ascending=False)[['TITULO', 'CITACIONES']].head(10))
        with c3:
            st.subheader("Top Impacto (Cit./Año)")
            if 'Citaciones por año' in dfScopus.columns:
                st.dataframe(dfScopus.sort_values(by='Citaciones por año', ascending=False)[['TITULO', 'Citaciones por año']].head(10))

    with st.expander("Ver DataFrame Completo"):
        st.dataframe(dfScopus)
else:
    if data_source == "Subir mi propio archivo CSV" and uploaded_file is None:
        # Mensaje vacío porque ya mostramos la imagen y las instrucciones arriba
        pass
    elif data_source == "Usar datos de ejemplo":
        st.info("Cargando datos de ejemplo...")
    else:
        st.info("Sube un archivo para comenzar.")
