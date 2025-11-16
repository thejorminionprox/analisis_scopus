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
from PIL import Image 

warnings.filterwarnings("ignore")

translations = {
    'es': {
        'page_title': "Análisis de Scopus",
        'main_title': "Análisis Interactivo de Publicaciones de Scopus",
        'app_description': "Esta aplicación analiza un conjunto de datos de Scopus.",
        'sidebar_header_source': "Fuente de Datos",
        'radio_source_prompt': "Elige una fuente de datos:",
        'radio_source_option_1': "Usar datos de ejemplo",
        'radio_source_option_2': "Subir mi propio archivo CSV",
        'upload_prompt': "Carga tu archivo CSV de Scopus:",
        'upload_success': "¡Archivo cargado exitosamente!",
        'upload_error': "Error al leer el archivo CSV: {e}",
        'info_instructions': "Para utilizar tu propia base de datos, el archivo CSV debe tener una estructura específica que contenga campos como Autores, Título, Año, Citaciones, Palabras clave, Tipo de Documento y Afiliaciones.",
        'image_caption': "Campos necesarios en la base de datos de Scopus",
        'warning_image_load': "No se pudo cargar la imagen de referencia 'image_ff8c45.jpg'.",
        'processing_data': "Procesando datos...",
        'sidebar_header_filters': "Filtros Interactivos",
        'slider_top_authors': "Top Autores Frecuentes:",
        'slider_top_keywords': "Top Palabras Clave Frecuentes:",
        'slider_top_sources': "Top Fuentes Comunes:",
        'text_search_author': "Buscar Autor (ej. Lauder G.):",
        'expander_processing_summary': "Ver Resumen del Procesamiento",
        'info_header': "Información del DataFrame",
        'keywords_cleaning_header': "Limpieza de Palabras Clave",
        'metric_before': "Registros antes",
        'metric_after': "Registros después",
        'metric_removed': "Eliminados",
        'tab_names': [
            "Análisis de Autores", 
            "Análisis de Publicaciones", 
            "Análisis de Palabras Clave", 
            "Fuentes y Afiliación", 
            "Análisis de Citaciones",
            "Búsqueda y Rankings"
        ],
        'tab1_header': "Análisis de Autores",
        'tab1_subheader_top_authors_short': "Top {count} Autores (Formato Corto)",
        'tab1_plot_title_top_authors_short': "Top {count} Autores más Frecuentes",
        'tab1_subheader_top_authors_full': "Top {count} Autores (Nombre Completo)",
        'tab1_plot_title_top_authors_full': "Top {count} Autores (Nombres Completos)",
        'tab1_subheader_authors_per_pub': "Número de autores por publicación",
        'tab1_plot_title_authors_per_pub': "Número de autores por publicación",
        'tab1_plot_xlabel_authors_per_pub': "Número de autores",
        'tab1_plot_ylabel_authors_per_pub': "Publicaciones",
        'warning_no_data': "No hay datos.",
        'tab2_header': "Análisis de Publicaciones",
        'tab2_subheader_pub_by_year': "Distribución de publicaciones por año",
        'tab2_subheader_pub_by_year_cumulative': "Publicaciones acumuladas por año",
        'tab2_plot_label_yearly': "Por año",
        'tab2_plot_label_cumulative': "Acumulado",
        'tab2_subheader_doc_types': "Distribución de Tipos de Documentos",
        'tab2_subheader_title_chars': "Distribución de Caracteres en el Título",
        'tab3_header': "Análisis de Palabras Clave",
        'tab3_subheader_top_keywords': "Top {count} Palabras clave",
        'tab3_subheader_word_cloud': "Nube de Palabras",
        'tab3_warning_no_keywords': "No hay palabras clave.",
        'tab3_subheader_title_words': "Frecuencia de palabras en el título",
        'tab4_header': "Análisis de Fuentes y Afiliación",
        'tab4_subheader_top_sources': "Top {count} Fuentes de publicación",
        'tab4_subheader_top_institutions': "Instituciones más Frecuentes",
        'tab4_subheader_top_countries': "Países más frecuentes",
        'tab4_warning_no_affiliation_data': "Sin datos de afiliación.",
        'tab5_header': "Análisis de Citaciones",
        'tab5_subheader_open_access': "Acceso Abierto",
        'tab5_pie_label_no': "No",
        'tab5_pie_label_yes': "Si",
        'tab5_subheader_cited_pubs': "Publicaciones con citaciones",
        'tab5_subheader_citation_dist': "Distribución de Citaciones (Log)",
        'tab5_subheader_boxplot_citations': "Boxplot Citaciones (Todas vs Review)",
        'tab5_boxplot_label_all': "Todas",
        'tab5_boxplot_label_review': "Review",
        'tab5_subheader_boxplot_citations_per_year': "Boxplot Citaciones por Año",
        'tab6_header': "Búsqueda y Rankings",
        'tab6_subheader_search_results': "Búsqueda: '{query}'",
        'tab6_subheader_top_impact_general': "Top Impacto (General)",
        'tab6_subheader_top_impact_review': "Top Impacto (Review)",
        'tab6_subheader_top_impact_cites_per_year': "Top Impacto (Cit./Año)",
        'expander_full_dataframe': "Ver DataFrame Completo",
        'info_loading_sample': "Cargando datos de ejemplo...",
        'info_upload_prompt': "Sube un archivo para comenzar.",
        'error_filenotfound': "Error: No se encontró el archivo '{file}'.",
        'warning_authors_not_found': "Columna 'Authors' no encontrada.",
        'warning_year_not_found': "Columna 'Year' no encontrada.",
        'warning_affiliations_not_found': "Columna 'Affiliations' no encontrada.",
        'cited_yes': 'Si',
        'cited_no': 'No',
        'download_button_label': "Descargar DataFrame (CSV)",
        'download_file_name': "analisis_scopus.csv"
    },
    'en': {
        'page_title': "Scopus Analysis",
        'main_title': "Interactive Scopus Publication Analysis",
        'app_description': "This application analyzes a Scopus dataset.",
        'sidebar_header_source': "Data Source",
        'radio_source_prompt': "Choose a data source:",
        'radio_source_option_1': "Use sample data",
        'radio_source_option_2': "Upload my own CSV file",
        'upload_prompt': "Upload your Scopus CSV file:",
        'upload_success': "File uploaded successfully!",
        'upload_error': "Error reading CSV file: {e}",
        'info_instructions': "To use your own database, the CSV file must have a specific structure containing fields like Authors, Title, Year, Citations, Keywords, Document Type, and Affiliations.",
        'image_caption': "Required fields in the Scopus database",
        'warning_image_load': "Could not load reference image 'image_ff8c45.jpg'.",
        'processing_data': "Processing data...",
        'sidebar_header_filters': "Interactive Filters",
        'slider_top_authors': "Top Frequent Authors:",
        'slider_top_keywords': "Top Frequent Keywords:",
        'slider_top_sources': "Top Common Sources:",
        'text_search_author': "Search Author (e.g., Lauder G.):",
        'expander_processing_summary': "View Processing Summary",
        'info_header': "DataFrame Information",
        'keywords_cleaning_header': "Keyword Cleaning",
        'metric_before': "Records before",
        'metric_after': "Records after",
        'metric_removed': "Removed",
        'tab_names': [
            "Author Analysis", 
            "Publication Analysis", 
            "Keyword Analysis", 
            "Sources & Affiliation", 
            "Citation Analysis",
            "Search & Rankings"
        ],
        'tab1_header': "Author Analysis",
        'tab1_subheader_top_authors_short': "Top {count} Authors (Short Format)",
        'tab1_plot_title_top_authors_short': "Top {count} Most Frequent Authors",
        'tab1_subheader_top_authors_full': "Top {count} Authors (Full Name)",
        'tab1_plot_title_top_authors_full': "Top {count} Authors (Full Names)",
        'tab1_subheader_authors_per_pub': "Number of authors per publication",
        'tab1_plot_title_authors_per_pub': "Number of authors per publication",
        'tab1_plot_xlabel_authors_per_pub': "Number of authors",
        'tab1_plot_ylabel_authors_per_pub': "Publications",
        'warning_no_data': "No data available.",
        'tab2_header': "Publication Analysis",
        'tab2_subheader_pub_by_year': "Distribution of publications by year",
        'tab2_subheader_pub_by_year_cumulative': "Cumulative publications by year",
        'tab2_plot_label_yearly': "By year",
        'tab2_plot_label_cumulative': "Cumulative",
        'tab2_subheader_doc_types': "Distribution of Document Types",
        'tab2_subheader_title_chars': "Distribution of Characters in Title",
        'tab3_header': "Keyword Analysis",
        'tab3_subheader_top_keywords': "Top {count} Keywords",
        'tab3_subheader_word_cloud': "Word Cloud",
        'tab3_warning_no_keywords': "No keywords available.",
        'tab3_subheader_title_words': "Frequency of words in title",
        'tab4_header': "Source and Affiliation Analysis",
        'tab4_subheader_top_sources': "Top {count} Publication Sources",
        'tab4_subheader_top_institutions': "Most Frequent Institutions",
        'tab4_subheader_top_countries': "Most Frequent Countries",
        'tab4_warning_no_affiliation_data': "No affiliation data.",
        'tab5_header': "Citation Analysis",
        'tab5_subheader_open_access': "Open Access",
        'tab5_pie_label_no': "No",
        'tab5_pie_label_yes': "Yes",
        'tab5_subheader_cited_pubs': "Publications with citations",
        'tab5_subheader_citation_dist': "Distribution of Citations (Log)",
        'tab5_subheader_boxplot_citations': "Boxplot Citations (All vs Review)",
        'tab5_boxplot_label_all': "All",
        'tab5_boxplot_label_review': "Review",
        'tab5_subheader_boxplot_citations_per_year': "Boxplot Citations per Year",
        'tab6_header': "Search and Rankings",
        'tab6_subheader_search_results': "Search: '{query}'",
        'tab6_subheader_top_impact_general': "Top Impact (General)",
        'tab6_subheader_top_impact_review': "Top Impact (Review)",
        'tab6_subheader_top_impact_cites_per_year': "Top Impact (Cites/Year)",
        'expander_full_dataframe': "View Full DataFrame",
        'info_loading_sample': "Loading sample data...",
        'info_upload_prompt': "Upload a file to start.",
        'error_filenotfound': "Error: File '{file}' not found.",
        'warning_authors_not_found': "'Authors' column not found.",
        'warning_year_not_found': "'Year' column not found.",
        'warning_affiliations_not_found': "'Affiliations' column not found.",
        'cited_yes': 'Yes',
        'cited_no': 'No',
        'download_button_label': "Download DataFrame (CSV)",
        'download_file_name': "scopus_analysis.csv"
    }
}

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
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def load_sample_data(file_path, t_error_string):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(t_error_string.format(file=file_path))
        return None

@st.cache_data
def process_data(df_raw, t_strings):
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
        st.warning(t_strings['warning_authors_not_found'])
        dfScopus['LISTAUTORES'] = [[]] * len(dfScopus)
        dfScopus['CANTIDADAUTORES'] = 0

    if 'AUTORESCOMPLETOS' in dfScopus.columns:
        dfScopus['LISTAUTORESCOMPLETOS'] = dfScopus['AUTORESCOMPLETOS'].str.split('; ')
    else:
        dfScopus['LISTAUTORESCOMPLETOS'] = [[]] * len(dfScopus)

    if 'ANIO' in dfScopus.columns:
        dfScopus['ANIO'] = pd.to_numeric(dfScopus['ANIO'], errors='coerce')
    else:
        st.warning(t_strings['warning_year_not_found'])
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
        dfScopus['Citado'] = np.where(dfScopus['CITACIONES'] > 0, t_strings['cited_yes'], t_strings['cited_no'])
    else:
        dfScopus['CITACIONES'] = 0
        dfScopus['Citaciones por año'] = 0
        dfScopus['Citado'] = t_strings['cited_no']

    if 'Affiliations' in df_raw.columns:
        dfScopus['Afilaciones'] = df_raw['Affiliations'].str.split('; ')
        dfScopus['pais'] = dfScopus['Afilaciones'].str[-1]
        dfScopus['Pais'] = dfScopus['pais'].str.split().str[-1]
        dfScopus['Pais'] = dfScopus['Pais'].replace('States', 'USA')
        dfScopus['Pais'] = dfScopus['Pais'].replace('Kingdom', 'United Kingdom')
    else:
        st.warning(t_strings['warning_affiliations_not_found'])
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

selected_lang_label = st.sidebar.selectbox(
    "Idioma / Language", 
    ("Español", "English")
)
lang_map = {"Español": "es", "English": "en"}
lang_key = lang_map[selected_lang_label]

t = translations[lang_key]

st.set_page_config(page_title=t['page_title'], layout="wide")
st.title(t['main_title'])
st.write(t['app_description'])

DATA_FILE = "scopusffandhkorwtorhf.csv"

st.sidebar.header(t['sidebar_header_source'])
data_source = st.sidebar.radio(
    t['radio_source_prompt'],
    (t['radio_source_option_1'], t['radio_source_option_2'])
)

dfScopus_raw = None
uploaded_file = None

if data_source == t['radio_source_option_1']:
    dfScopus_raw = load_sample_data(DATA_FILE, t['error_filenotfound'])
    
else:
    st.info(t['info_instructions'])
    
    try:
        # CORRECCIÓN: Se usa el nombre de archivo 'image_ff8c45.jpg'
        img = Image.open("image_ff8c45.jpg") 
        width, height = img.size
        pixeles_a_cortar = 50 
        crop_coords = (0, 0, width, height - pixeles_a_cortar)
        cropped_img = img.crop(crop_coords)
        st.image(cropped_img, caption=t['image_caption'], use_container_width=True)

    except FileNotFoundError:
        # CORRECCIÓN: Se actualiza el mensaje de error
        st.warning(t['warning_image_load']) 
    except Exception as e:
        st.warning(f"No se pudo cargar o recortar la imagen 'image_ff8c45.jpg': {e}")
    
    st.markdown("---")

    uploaded_file = st.sidebar.file_uploader(t['upload_prompt'], type=["csv"])
    
    if uploaded_file is not None:
        try:
            dfScopus_raw = pd.read_csv(uploaded_file)
            st.sidebar.success(t['upload_success'])
        except Exception as e:
            st.error(t['upload_error'].format(e=e))

if dfScopus_raw is not None:
    
    t_process_strings = {
        'warning_authors_not_found': t['warning_authors_not_found'],
        'warning_year_not_found': t['warning_year_not_found'],
        'warning_affiliations_not_found': t['warning_affiliations_not_found'],
        'cited_yes': t['cited_yes'],
        'cited_no': t['cited_no']
    }

    with st.spinner(t['processing_data']):
        dfScopus, longitudactual, longitudnueva, contadorborrado = process_data(dfScopus_raw, t_process_strings)

    st.sidebar.header(t['sidebar_header_filters'])

    CantidadAutores = st.sidebar.slider(t['slider_top_authors'], 5, 50, 10, 5)
    CantidadPalabrasClave = st.sidebar.slider(t['slider_top_keywords'], 5, 50, 20, 5)
    CantidadFuentes = st.sidebar.slider(t['slider_top_sources'], 5, 50, 10, 5)
    search_string = st.sidebar.text_input(t['text_search_author'], "Lauder G.")

    with st.expander(t['expander_processing_summary']):
        st.subheader(t['info_header'])
        buffer = io.StringIO()
        dfScopus.info(buf=buffer)
        st.text(buffer.getvalue())
        st.subheader(t['keywords_cleaning_header'])
        st.metric(t['metric_before'], longitudactual)
        st.metric(t['metric_after'], longitudnueva)
        st.metric(t['metric_removed'], contadorborrado, delta_color="inverse")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(t['tab_names'])

    with tab1:
        st.header(t['tab1_header'])
        st.subheader(t['tab1_subheader_top_authors_short'].format(count=CantidadAutores))
        try:
            autores = dfScopus['LISTAUTORES'].explode()
            if not autores.empty:
                cuentauores = Counter(autores)
                top_autores_df = pd.DataFrame(cuentauores.most_common(CantidadAutores), columns=['Author', 'Count'])
                
                altura_dinamica_1 = max(6, len(top_autores_df) * 0.4)
                fig1, ax1 = plt.subplots(figsize=(10, altura_dinamica_1))
                
                bars1 = ax1.barh(top_autores_df['Author'], top_autores_df['Count'], color='skyblue')
                ax1.set_title(t['tab1_plot_title_top_authors_short'].format(count=CantidadAutores))
                ax1.invert_yaxis()
                for bar in bars1:
                    ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                st.pyplot(fig1)
            else:
                st.warning(t['warning_no_data'])
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab1_subheader_top_authors_full'].format(count=CantidadAutores))
        try:
            autorescompletos = dfScopus['LISTAUTORESCOMPLETOS'].explode()
            if not autorescompletos.empty:
                cuentaautorescompletos = Counter(autorescompletos)
                top_autores_completos_df = pd.DataFrame(cuentaautorescompletos.most_common(CantidadAutores), columns=['Author', 'Count'])
                
                altura_dinamica_2 = max(6, len(top_autores_completos_df) * 0.4)
                fig2, ax2 = plt.subplots(figsize=(10, altura_dinamica_2))
                
                bars2 = ax2.barh(top_autores_completos_df['Author'], top_autores_completos_df['Count'], color='lightgreen')
                ax2.set_title(t['tab1_plot_title_top_authors_full'].format(count=CantidadAutores))
                ax2.invert_yaxis()
                for bar in bars2:
                    ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab1_subheader_authors_per_pub'])
        try:
            if 'CANTIDADAUTORES' in dfScopus.columns:
                df3filtrado = dfScopus[dfScopus['CANTIDADAUTORES'] >= 1]
                if not df3filtrado.empty:
                    conteo_autores = df3filtrado['CANTIDADAUTORES'].value_counts().sort_index()
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    
                    bars3 = ax3.bar(conteo_autores.index, conteo_autores.values) 
                    
                    ax3.set_title(t['tab1_plot_title_authors_per_pub'])
                    ax3.set_xlabel(t['tab1_plot_xlabel_authors_per_pub'])
                    ax3.set_ylabel(t['tab1_plot_ylabel_authors_per_pub'])
                    
                    for bar in bars3:
                        height = bar.get_height()
                        ax3.text(
                            bar.get_x() + bar.get_width() / 2, 
                            height,                          
                            f'{height}',                     
                            ha='center',
                            va='bottom'
                        )
                        
                    st.pyplot(fig3)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab2:
        st.header(t['tab2_header'])
        st.subheader(t['tab2_subheader_pub_by_year'])
        try:
            pubporanio = dfScopus['ANIO'].value_counts().sort_index()
            if not pubporanio.empty:
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                bars4 = pubporanio.plot(kind='bar', color='peru', ax=ax4)
                for bar in bars4.containers[0]:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab2_subheader_pub_by_year_cumulative'])
        try:
            if not pubporanio.empty:
                df_acum = pubporanio.reset_index()
                df_acum.columns = ['ANIO', 'count']
                df_acum = df_acum.sort_values(by='ANIO')
                df_acum['Acumulado'] = df_acum['count'].cumsum()
                fig5, ax5 = plt.subplots(figsize=(12, 6))
                ax5.bar(df_acum['ANIO'], df_acum['count'], color='lightblue', label=t['tab2_plot_label_yearly'])
                ax5b = ax5.twinx()
                ax5b.plot(df_acum['ANIO'], df_acum['Acumulado'], color='red', marker='o', label=t['tab2_plot_label_cumulative'])
                ax5.legend(loc='upper left')
                ax5b.legend(loc='upper right')
                st.pyplot(fig5)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab2_subheader_doc_types'])
        try:
            if 'TIPO' in dfScopus.columns:
                document_type_counts = dfScopus['TIPO'].value_counts()
                if not document_type_counts.empty:
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    bars6 = document_type_counts.plot(kind='bar', ax=ax6)
                    for bar in bars6.containers[0]:
                        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    
                    st.pyplot(fig6)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab2_subheader_title_chars'])
        try:
            if 'CARACTERESTITULO' in dfScopus.columns:
                fig7, ax7 = plt.subplots(figsize=(10, 4))
                ax7.boxplot(dfScopus['CARACTERESTITULO'].dropna(), vert=False, showfliers=False)
                st.pyplot(fig7)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab3:
        st.header(t['tab3_header'])
        try:
            keywords_exploded = dfScopus['ALLKEYWORDS'].explode()
            keyword_counts = Counter(keywords_exploded)
            if '' in keyword_counts: del keyword_counts['']
            
            if keyword_counts:
                st.subheader(t['tab3_subheader_top_keywords'].format(count=CantidadPalabrasClave))
                top_keywords_df = pd.DataFrame(keyword_counts.most_common(CantidadPalabrasClave), columns=['Keyword', 'Count'])
                
                altura_dinamica_8 = max(8, len(top_keywords_df) * 0.4)
                fig8, ax8 = plt.subplots(figsize=(10, altura_dinamica_8))
                
                bars8 = ax8.barh(top_keywords_df['Keyword'], top_keywords_df['Count'], color='salmon')
                ax8.invert_yaxis()
                for bar in bars8:
                    ax8.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                st.pyplot(fig8)

                st.subheader(t['tab3_subheader_word_cloud'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_counts)
                fig9, ax9 = plt.subplots(figsize=(10, 6))
                ax9.imshow(wordcloud, interpolation='bilinear')
                ax9.axis('off')
                st.pyplot(fig9)
            else:
                st.warning(t['tab3_warning_no_keywords'])
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab3_subheader_title_words'])
        try:
            if 'TITULO' in dfScopus.columns:
                palabras_titulo = dfScopus['TITULO'].dropna().str.lower().str.cat(sep=';').split(' ')
                cuenta_palabras_titulo = Counter(palabras_titulo)
                
                # --- CORRECCIÓN DE SYNTAX ERROR ---
                palabras_df = pd.DataFrame(cuenta_palabras_titulo.most_common(), columns=['Palabra', 'Numero'])
                palabras_df = palabras_df[(palabras_df['Palabra'].str.len() > 3) & (palabras_df['Numero'] > 40)]
                stop_words = ['from', 'with', 'research', 'analysis', 'using', 'based', 'model', 'control', 'between', 'study']
                palabras_df = palabras_df[~palabras_df['Palabra'].isin(stop_words)]

                if not palabras_df.empty:
                    
                    altura_dinamica_10 = max(8, len(palabras_df) * 0.35)
                    fig10, ax10 = plt.subplots(figsize=(10, altura_dinamica_10))
                    
                    bars10 = ax10.barh(palabras_df['Palabra'], palabras_df['Numero'], color='red')
                    # --- FIN DE CORRECCIÓN ---

                    ax10.invert_yaxis()
                    st.pyplot(fig10)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab4:
        st.header(t['tab4_header'])
        st.subheader(t['tab4_subheader_top_sources'].format(count=CantidadFuentes))
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
                    
                    plt.tight_layout()
                    
                    st.pyplot(fig11)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab4_subheader_top_institutions'])
        if 'Afilaciones' in dfScopus.columns:
            try:
                instituciones = dfScopus['Afilaciones'].explode()
                top_inst = pd.DataFrame(Counter(instituciones).most_common(11), columns=['Institución', 'Numero'])
                if not top_inst.empty:
                    if pd.isna(top_inst.iloc[0, 0]) or top_inst.iloc[0, 0].strip() == '': top_inst = top_inst.drop([0])
                    
                    altura_dinamica_12 = max(6, len(top_inst) * 0.4)
                    fig12, ax12 = plt.subplots(figsize=(10, altura_dinamica_12))
                    
                    bars12 = ax12.barh(top_inst['Institución'], top_inst['Numero'], color='green')
                    ax12.invert_yaxis()
                    for bar in bars12:
                        ax12.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                    st.pyplot(fig12)
            except Exception as e:
                st.error(f"Error: {e}")

            st.subheader(t['tab4_subheader_top_countries'])
            try:
                pais = dfScopus['Pais'].explode().dropna()
                if not pais.empty:
                    top_pais = pd.DataFrame(Counter(pais).most_common(10), columns=['Pais', 'Numero'])
                    
                    altura_dinamica_13 = max(6, len(top_pais) * 0.4)
                    fig13, ax13 = plt.subplots(figsize=(10, altura_dinamica_13))
                    
                    bars13 = ax13.barh(top_pais['Pais'], top_pais['Numero'], color='red')
                    ax13.invert_yaxis()
                    for bar in bars13:
                        ax13.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()}', ha='left', va='center')
                    st.pyplot(fig13)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning(t['tab4_warning_no_affiliation_data'])

    with tab5:
        st.header(t['tab5_header'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t['tab5_subheader_open_access'])
            try:
                oa_counts = dfScopus['OPENACCESS'].value_counts()
                fig14, ax14 = plt.subplots(figsize=(6, 6))
                if not oa_counts.empty:
                    ax14.pie(oa_counts, autopct='%1.1f%%', colors=['skyblue', 'red'], startangle=90, labels=[t['tab5_pie_label_no'], t['tab5_pie_label_yes']])
                    st.pyplot(fig14)
            except Exception as e:
                st.error(f"Error: {e}")

        with col2:
            st.subheader(t['tab5_subheader_cited_pubs'])
            try:
                cit_counts = dfScopus['Citado'].value_counts()
                fig15, ax15 = plt.subplots(figsize=(6, 6))
                if not cit_counts.empty:
                    ax15.pie(cit_counts, autopct='%1.1f%%', labels=cit_counts.keys())
                    st.pyplot(fig15)
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader(t['tab5_subheader_citation_dist'])
        try:
            if 'CITACIONES' in dfScopus.columns:
                cits = dfScopus['CITACIONES'].value_counts().sort_index()
                fig16, ax16 = plt.subplots(figsize=(10, 6))
                ax16.bar(cits.index, cits.values)
                ax16.set_xscale('log')
                st.pyplot(fig16)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab5_subheader_boxplot_citations'])
        try:
            if 'TIPO' in dfScopus.columns and 'CITACIONES' in dfScopus.columns:
                rev = dfScopus[dfScopus['TIPO'] == 'Review']
                fig17, ax17 = plt.subplots(figsize=(10, 6))
                datos = [dfScopus['CITACIONES'].dropna(), rev['CITACIONES'].dropna()]
                bp = ax17.boxplot(datos, labels=[t['tab5_boxplot_label_all'], t['tab5_boxplot_label_review']], showfliers=False)
                for i, m in enumerate(bp['medians']):
                    if len(m.get_ydata()) > 0: ax17.text(i+1, m.get_ydata()[0], f'{m.get_ydata()[0]:.2f}', ha='center', va='bottom')
                st.pyplot(fig17)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader(t['tab5_subheader_boxplot_citations_per_year'])
        try:
            if 'Citaciones por año' in dfScopus.columns:
                fig18, ax18 = plt.subplots(figsize=(10, 6))
                datos = [dfScopus['Citaciones por año'].dropna()]
                bp = ax18.boxplot(datos, labels=[t['tab5_boxplot_label_all']], showfliers=False)
                for i, m in enumerate(bp['medians']):
                    if len(m.get_ydata()) > 0: ax18.text(i+1, m.get_ydata()[0], f'{m.get_ydata()[0]:.2f}', ha='center', va='bottom')
                st.pyplot(fig18)
        except Exception as e:
            st.error(f"Error: {e}")

    with tab6:
        st.header(t['tab6_header'])
        st.subheader(t['tab6_subheader_search_results'].format(query=search_string))
        if search_string and 'AUTORES' in dfScopus.columns:
            res = dfScopus[dfScopus['AUTORES'].str.contains(search_string, na=False)]
            cols = [c for c in ['TITULO', 'AUTORES', 'CITACIONES'] if c in res.columns]
            st.dataframe(res[cols].sort_values(by='CITACIONES', ascending=False).head(30) if 'CITACIONES' in cols else res[cols].head(30))

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader(t['tab6_subheader_top_impact_general'])
            if 'CITACIONES' in dfScopus.columns:
                st.dataframe(dfScopus.sort_values(by='CITACIONES', ascending=False)[['TITULO', 'CITACIONES']].head(10))
        with c2:
            st.subheader(t['tab6_subheader_top_impact_review'])
            if 'TIPO' in dfScopus.columns and 'CITACIONES' in dfScopus.columns:
                st.dataframe(dfScopus[dfScopus['TIPO'] == 'Review'].sort_values(by='CITACIONES', ascending=False)[['TITULO', 'CITACIONES']].head(10))
        with c3:
            st.subheader(t['tab6_subheader_top_impact_cites_per_year'])
            if 'Citaciones por año' in dfScopus.columns:
                st.dataframe(dfScopus.sort_values(by='Citaciones por año', ascending=False)[['TITULO', 'Citaciones por año']].head(10))

    st.divider()

    csv_data = convert_df_to_csv(dfScopus)
    
    st.download_button(
       label=t['download_button_label'],
       data=csv_data,
       file_name=t['download_file_name'],
       mime='text/csv',
    )

    with st.expander(t['expander_full_dataframe']):
        st.dataframe(dfScopus)
else:
    if data_source == t['radio_source_option_2'] and uploaded_file is None:
        pass
    elif data_source == t['radio_source_option_1']:
        if dfScopus_raw is None:
            pass
        else:
            st.info(t['info_loading_sample'])
    else:
        st.info(t['info_upload_prompt'])
