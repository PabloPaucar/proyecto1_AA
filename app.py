# -*- coding: utf-8 -*-
"""
UPScholar - Sistema de Búsqueda y Recomendación Inteligente
Proyecto Primer Bimestre
"""


import streamlit as st
import pandas as pd
from backend import metodo_tfidf, metodo_llm


# ======================================================================
# CONFIGURACIÓN DE PÁGINA
# ======================================================================
st.set_page_config(
    page_title="UPScholar",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================
# ESTILOS CSS PERSONALIZADOS
# ======================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .article-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .similarity-score {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .time-badge {
        display: inline-block;
        background-color: #E3F2FD;
        color: #1976D2;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .search-badge {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)


# ======================================================================
# INICIALIZACIÓN DEL SISTEMA (SOLO UNA VEZ)
# ======================================================================
@st.cache_resource
def inicializar_sistemas():
    """Inicializa ambos métodos de búsqueda (solo primera vez)"""
    status = {}
    
    # Método TF-IDF
    with st.spinner("Inicializando método TF-IDF..."):
        msg_tfidf, success_tfidf = metodo_tfidf.inicializar_tfidf('data/ICMLA_2014_2015_2016_2017.csv')
        status['tfidf'] = (msg_tfidf, success_tfidf)
    
    # Método LLM
    with st.spinner("Inicializando método LLM (puede tardar la primera vez)..."):
        msg_llm, success_llm = metodo_llm.inicializar_llm('data/ICMLA_2014_2015_2016_2017.csv')
        status['llm'] = (msg_llm, success_llm)
    
    return status


# Inicializar sistemas
status = inicializar_sistemas()


# Inicializar session_state
if 'ultima_busqueda_tfidf' not in st.session_state:
    st.session_state.ultima_busqueda_tfidf = None
if 'ultima_busqueda_llm' not in st.session_state:
    st.session_state.ultima_busqueda_llm = None
if 'desde_sugerencia_tfidf' not in st.session_state:
    st.session_state.desde_sugerencia_tfidf = False
if 'desde_sugerencia_llm' not in st.session_state:
    st.session_state.desde_sugerencia_llm = False


# ======================================================================
# HEADER PRINCIPAL
# ======================================================================
st.markdown('<h1 class="main-header">UPScholar</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de Búsqueda y Recomendación Inteligente de Documentos Científicos</p>', unsafe_allow_html=True)


# ======================================================================
# SIDEBAR - INFORMACIÓN Y ESTADÍSTICAS
# ======================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/search.png", width=80)
    st.title("Información del Sistema")
    
    # Estado de inicialización
    st.subheader("Estado de los Métodos")
    if status['tfidf'][1]:
        st.success("TF-IDF: Listo")
    else:
        st.error("TF-IDF: Error")
    
    if status['llm'][1]:
        st.success("LLM: Listo")
    else:
        st.error("LLM: Error")
    
    st.divider()
    
    # Estadísticas del dataset
    doc = metodo_tfidf.get_dataset()
    if doc is not None:
        st.subheader("Estadísticas")
        st.metric("Total de Artículos", len(doc))
        st.metric("Años de Conferencias", "2014-2017")
        st.metric("Conferencia", "ICMLA")
    
    st.divider()
    
    # Información de métodos
    st.subheader("Métodos Implementados")
    st.markdown("""
    **Método 1: TF-IDF + Jaccard**
    - 60% Abstracts (TF-IDF)
    - 25% Keywords (Jaccard)
    - 15% Títulos (Jaccard)
    
    **Método 2: LLM Embeddings**
    - Modelo: MPNet
    - Solo Abstracts
    - 768 dimensiones
    """)


# ======================================================================
# TABS PRINCIPALES
# ======================================================================
tab1, tab2, tab3 = st.tabs(["Método TF-IDF", "Método LLM", "Comparación"])


# ======================================================================
# TAB 1: MÉTODO TF-IDF
# ======================================================================
with tab1:
    st.header("Búsqueda con Matriz Ponderada (TF-IDF + Jaccard)")
    st.markdown("Combina similitud TF-IDF en abstracts con Jaccard en keywords y títulos")
    
    # Formulario de búsqueda
    with st.form(key="form_tfidf"):
        query_tfidf = st.text_input(
            "Ingrese su consulta:",
            placeholder="Ejemplo: machine learning algorithms"
        )
        
        btn_tfidf = st.form_submit_button("Buscar", use_container_width=False)
    
    # Capturar query desde formulario
    if btn_tfidf and query_tfidf:
        st.session_state.ultima_busqueda_tfidf = query_tfidf
        st.session_state.desde_sugerencia_tfidf = False
    
    # EJECUTAR BÚSQUEDA (desde formulario O desde session_state)
    if st.session_state.ultima_busqueda_tfidf:
        query_final = st.session_state.ultima_busqueda_tfidf
        
        # Mostrar qué se está buscando (especialmente si viene de sugerencia)
        if st.session_state.desde_sugerencia_tfidf:
            st.markdown(
                f'<div class="search-badge">🔍 Buscando: <strong>{query_final}</strong></div>',
                unsafe_allow_html=True
            )
            st.session_state.desde_sugerencia_tfidf = False
        
        with st.spinner("Procesando búsqueda con TF-IDF..."):
            try:
                idx_top10, similitudes, tiene_resultados, tiempo_ms, sugerencias = metodo_tfidf.buscar_tfidf(query_final)
                
                # ============================================
                # NUEVO: Verificar si la similitud máxima es 0
                # ============================================
                mejor_similitud = similitudes[idx_top10[0]] if len(idx_top10) > 0 else 0.0
                
                if mejor_similitud == 0.0:
                    # NO HAY RESULTADOS RELEVANTES
                    st.error("❌ No se encontraron resultados relevantes para esta consulta.")
                    
                    # Mostrar sugerencias si existen
                    if sugerencias:
                        st.info("💡 **¿Quisiste decir?** Haz clic en una sugerencia para buscarla:")
                        
                        for i, sug in enumerate(sugerencias):
                            palabra_incorrecta = sug.split(" → ")[0] if " → " in sug else ""
                            palabra_sugerida = sug.split(" → ")[1] if " → " in sug else sug
                            
                            if st.button(f"{palabra_incorrecta} → {palabra_sugerida}", 
                                       key=f"sug_tfidf_{i}", 
                                       use_container_width=False,
                                       type="secondary"):
                                st.session_state.ultima_busqueda_tfidf = query_final.replace(
                                    palabra_incorrecta, palabra_sugerida
                                )
                                st.session_state.desde_sugerencia_tfidf = True
                                st.rerun()
                    else:
                        st.info("💡 **Intenta con:**\n- Palabras clave más generales\n- Términos en inglés\n- Temas relacionados a Machine Learning")
                    
                    # NO MOSTRAR LOS 10 DOCUMENTOS
                    
                else:
                    # SÍ HAY RESULTADOS (similitud > 0)
                    
                    # Mostrar sugerencias si existen
                    if sugerencias:
                        st.info("💡 **¿Quisiste decir?** Haz clic en una sugerencia para buscarla:")
                        
                        for i, sug in enumerate(sugerencias):
                            palabra_incorrecta = sug.split(" → ")[0] if " → " in sug else ""
                            palabra_sugerida = sug.split(" → ")[1] if " → " in sug else sug
                            
                            if st.button(f"{palabra_incorrecta} → {palabra_sugerida}", 
                                       key=f"sug_tfidf_{i}", 
                                       use_container_width=False,
                                       type="secondary"):
                                st.session_state.ultima_busqueda_tfidf = query_final.replace(
                                    palabra_incorrecta, palabra_sugerida
                                )
                                st.session_state.desde_sugerencia_tfidf = True
                                st.rerun()
                    
                    # Mensaje de éxito/advertencia con tiempo
                    if tiene_resultados:
                        st.markdown(
                            f'<p style="color: #4CAF50; font-size: 1rem; margin-bottom: 0.5rem;">'
                            f'✅ Se encontraron {len(idx_top10)} artículos relevantes '
                            f'<span class="time-badge">⏱️ {tiempo_ms:.2f} ms</span>'
                            f'</p>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Baja calidad pero similitud > 0
                        st.warning("⚠️ Los resultados pueden tener baja relevancia. Considera usar otras palabras clave.")
                        
                        st.markdown(
                            f'<p style="color: #FF9800; font-size: 0.9rem; margin-bottom: 0.5rem;">'
                            f'Mostrando los {len(idx_top10)} documentos más cercanos '
                            f'<span class="time-badge">⏱️ {tiempo_ms:.2f} ms</span>'
                            f'</p>',
                            unsafe_allow_html=True
                        )
                    
                    # MOSTRAR LOS 10 DOCUMENTOS
                    doc = metodo_tfidf.get_dataset()
                    st.subheader("Top 10 Artículos Más Similares")
                    
                    for i, idx in enumerate(idx_top10, 1):
                        with st.expander(
                            f"**#{i}** - {doc['title'].iloc[idx]}", 
                            expanded=False
                        ):
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.markdown(f"**Título:** {doc['title'].iloc[idx]}")
                                st.markdown(f"**Keywords:** {doc['keywords'].iloc[idx]}")
                            
                            with col_b:
                                st.metric("Similitud", f"{similitudes[idx]:.4f}")
                            
                            abstract_text = doc['abstract'].iloc[idx]
                            if len(abstract_text) > 400:
                                st.markdown(f"**Abstract:** {abstract_text[:400]}...")
                            else:
                                st.markdown(f"**Abstract:** {abstract_text}")
                            
                            st.divider()
                            
                            st.markdown("### Artículos Relacionados")
                            recs = metodo_tfidf.recomendar_tfidf(idx, idx_top10)
                            
                            for j, (rec_idx, score) in enumerate(recs, 1):
                                st.markdown(
                                    f"**{j}.** {doc['title'].iloc[rec_idx]} "
                                    f"<span class='similarity-score'>sim: {score:.4f}</span>",
                                    unsafe_allow_html=True
                                )
                
            except Exception as e:
                st.error(f"Error en la búsqueda: {str(e)}")


# ======================================================================
# TAB 2: MÉTODO LLM
# ======================================================================
with tab2:
    st.header("Búsqueda con Embeddings LLM (MPNet)")
    st.markdown("Utiliza embeddings del modelo MPNet para capturar similitud semántica")
    
    # Formulario de búsqueda
    with st.form(key="form_llm"):
        query_llm = st.text_input(
            "Ingrese su consulta:",
            placeholder="Ejemplo: deep neural networks"
        )
        
        btn_llm = st.form_submit_button("Buscar", use_container_width=False)
    
    # Capturar query desde formulario
    if btn_llm and query_llm:
        st.session_state.ultima_busqueda_llm = query_llm
        st.session_state.desde_sugerencia_llm = False
    
    # EJECUTAR BÚSQUEDA (desde formulario O desde session_state)
    if st.session_state.ultima_busqueda_llm:
        query_final_llm = st.session_state.ultima_busqueda_llm
        
        # Mostrar qué se está buscando (especialmente si viene de sugerencia)
        if st.session_state.desde_sugerencia_llm:
            st.markdown(
                f'<div class="search-badge">🔍 Buscando: <strong>{query_final_llm}</strong></div>',
                unsafe_allow_html=True
            )
            st.session_state.desde_sugerencia_llm = False
        
        with st.spinner("Procesando búsqueda con LLM..."):
            try:
                idx_top10, similitudes, tiene_resultados, tiempo_ms, sugerencias = metodo_llm.buscar_llm(query_final_llm)
                
                # ============================================
                # NUEVO: Verificar si la similitud máxima es 0
                # ============================================
                mejor_similitud = similitudes[idx_top10[0]] if len(idx_top10) > 0 else 0.0
                
                if mejor_similitud == 0.0:
                    # NO HAY RESULTADOS RELEVANTES
                    st.error("❌ No se encontraron resultados relevantes para esta consulta.")
                    
                    # Mostrar sugerencias si existen
                    if sugerencias:
                        st.info("💡 **¿Quisiste decir?** Haz clic en una sugerencia para buscarla:")
                        
                        for i, sug in enumerate(sugerencias):
                            palabra_incorrecta = sug.split(" → ")[0] if " → " in sug else ""
                            palabra_sugerida = sug.split(" → ")[1] if " → " in sug else sug
                            
                            if st.button(f"{palabra_incorrecta} → {palabra_sugerida}", 
                                       key=f"sug_llm_{i}", 
                                       use_container_width=False,
                                       type="secondary"):
                                st.session_state.ultima_busqueda_llm = query_final_llm.replace(
                                    palabra_incorrecta, palabra_sugerida
                                )
                                st.session_state.desde_sugerencia_llm = True
                                st.rerun()
                    else:
                        st.info("💡 **Intenta con:**\n- Palabras clave más generales\n- Términos en inglés\n- Temas relacionados a Machine Learning")
                    
                    # NO MOSTRAR LOS 10 DOCUMENTOS
                    
                else:
                    # SÍ HAY RESULTADOS (similitud > 0)
                    
                    # Mostrar sugerencias si existen
                    if sugerencias:
                        st.info("💡 **¿Quisiste decir?** Haz clic en una sugerencia para buscarla:")
                        
                        for i, sug in enumerate(sugerencias):
                            palabra_incorrecta = sug.split(" → ")[0] if " → " in sug else ""
                            palabra_sugerida = sug.split(" → ")[1] if " → " in sug else sug
                            
                            if st.button(f"{palabra_incorrecta} → {palabra_sugerida}", 
                                       key=f"sug_llm_{i}", 
                                       use_container_width=False,
                                       type="secondary"):
                                st.session_state.ultima_busqueda_llm = query_final_llm.replace(
                                    palabra_incorrecta, palabra_sugerida
                                )
                                st.session_state.desde_sugerencia_llm = True
                                st.rerun()
                    
                    # Mensaje de éxito/advertencia con tiempo
                    if tiene_resultados:
                        st.markdown(
                            f'<p style="color: #4CAF50; font-size: 1rem; margin-bottom: 0.5rem;">'
                            f'✅ Se encontraron {len(idx_top10)} artículos relevantes '
                            f'<span class="time-badge">⏱️ {tiempo_ms:.2f} ms</span>'
                            f'</p>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Baja calidad pero similitud > 0
                        st.warning("⚠️ Los resultados pueden tener baja relevancia. Considera usar otras palabras clave.")
                        
                        st.markdown(
                            f'<p style="color: #FF9800; font-size: 0.9rem; margin-bottom: 0.5rem;">'
                            f'Mostrando los {len(idx_top10)} documentos más cercanos '
                            f'<span class="time-badge">⏱️ {tiempo_ms:.2f} ms</span>'
                            f'</p>',
                            unsafe_allow_html=True
                        )
                    
                    # MOSTRAR LOS 10 DOCUMENTOS
                    doc = metodo_llm.get_dataset()
                    st.subheader("Top 10 Artículos Más Similares")
                    
                    for i, idx in enumerate(idx_top10, 1):
                        with st.expander(
                            f"**#{i}** - {doc['title'].iloc[idx]}", 
                            expanded=False
                        ):
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.markdown(f"**Título:** {doc['title'].iloc[idx]}")
                                st.markdown(f"**Keywords:** {doc['keywords'].iloc[idx]}")
                            
                            with col_b:
                                st.metric("Similitud", f"{similitudes[idx]:.4f}")
                            
                            abstract_text = doc['abstract'].iloc[idx]
                            if len(abstract_text) > 400:
                                st.markdown(f"**Abstract:** {abstract_text[:400]}...")
                            else:
                                st.markdown(f"**Abstract:** {abstract_text}")
                            
                            st.divider()
                            
                            st.markdown("### Artículos Relacionados")
                            recs = metodo_llm.recomendar_llm(idx, idx_top10)
                            
                            for j, (rec_idx, score) in enumerate(recs, 1):
                                st.markdown(
                                    f"**{j}.** {doc['title'].iloc[rec_idx]} "
                                    f"<span class='similarity-score'>sim: {score:.4f}</span>",
                                    unsafe_allow_html=True
                                )
                
            except Exception as e:
                st.error(f"Error en la búsqueda: {str(e)}")


# ======================================================================
# TAB 3: COMPARACIÓN
# ======================================================================
with tab3:
    st.header("Comparación de Métodos")
    st.markdown("Compare los resultados de ambos métodos con la misma consulta")
    
    # Usar formulario para permitir Enter
    with st.form(key="form_compare"):
        query_compare = st.text_input(
            "Ingrese su consulta para comparar:",
            placeholder="Ejemplo: classification algorithms"
        )
        
        btn_compare = st.form_submit_button("Buscar con Ambos Métodos", use_container_width=False)
    
    if btn_compare and query_compare:
        col_left, col_right = st.columns(2)
        
        # Columna Izquierda: TF-IDF
        with col_left:
            st.subheader("TF-IDF + Jaccard")
            with st.spinner("Procesando..."):
                try:
                    idx_tfidf, sim_tfidf, tiene_resultados_tfidf, tiempo_tfidf, sugerencias_tfidf = metodo_tfidf.buscar_tfidf(query_compare)
                    
                    # Verificar similitud
                    mejor_sim_tfidf = sim_tfidf[idx_tfidf[0]] if len(idx_tfidf) > 0 else 0.0
                    
                    if mejor_sim_tfidf == 0.0:
                        st.error("❌ Sin resultados")
                        if sugerencias_tfidf:
                            st.caption("**Sugerencias:**")
                            for sug in sugerencias_tfidf[:2]:
                                st.caption(f"• {sug}")
                    else:
                        # Tiempo compacto
                        st.caption(f"⏱️ {tiempo_tfidf:.2f} ms")
                        
                        # Mostrar sugerencias si existen
                        if sugerencias_tfidf:
                            st.caption("**Sugerencias:**")
                            for sug in sugerencias_tfidf[:2]:
                                st.caption(f"• {sug}")
                        
                        if not tiene_resultados_tfidf:
                            st.caption("⚠️ Baja relevancia")
                        
                        doc = metodo_tfidf.get_dataset()
                        
                        for i, idx in enumerate(idx_tfidf[:5], 1):
                            st.markdown(f"**{i}.** {doc['title'].iloc[idx]}")
                            st.caption(f"Similitud: {sim_tfidf[idx]:.4f}")
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Columna Derecha: LLM
        with col_right:
            st.subheader("LLM Embeddings")
            with st.spinner("Procesando..."):
                try:
                    idx_llm, sim_llm, tiene_resultados_llm, tiempo_llm, sugerencias_llm = metodo_llm.buscar_llm(query_compare)
                    
                    # Verificar similitud
                    mejor_sim_llm = sim_llm[idx_llm[0]] if len(idx_llm) > 0 else 0.0
                    
                    if mejor_sim_llm == 0.0:
                        st.error("❌ Sin resultados")
                        if sugerencias_llm:
                            st.caption("**Sugerencias:**")
                            for sug in sugerencias_llm[:2]:
                                st.caption(f"• {sug}")
                    else:
                        # Tiempo compacto
                        st.caption(f"⏱️ {tiempo_llm:.2f} ms")
                        
                        # Mostrar sugerencias si existen
                        if sugerencias_llm:
                            st.caption("**Sugerencias:**")
                            for sug in sugerencias_llm[:2]:
                                st.caption(f"• {sug}")
                        
                        if not tiene_resultados_llm:
                            st.caption("⚠️ Baja relevancia")
                        
                        doc = metodo_llm.get_dataset()
                        
                        for i, idx in enumerate(idx_llm[:5], 1):
                            st.markdown(f"**{i}.** {doc['title'].iloc[idx]}")
                            st.caption(f"Similitud: {sim_llm[idx]:.4f}")
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Análisis de coincidencias (solo si ambos tienen resultados)
        if 'mejor_sim_tfidf' in locals() and 'mejor_sim_llm' in locals():
            if mejor_sim_tfidf > 0.0 and mejor_sim_llm > 0.0:
                st.divider()
                st.subheader("Análisis de Coincidencias")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Tiempo TF-IDF", f"{tiempo_tfidf:.0f} ms")
                with col_m2:
                    st.metric("Tiempo LLM", f"{tiempo_llm:.0f} ms")
                with col_m3:
                    coincidencias = set(idx_tfidf) & set(idx_llm)
                    st.metric("Coincidencias", len(coincidencias))
                
                if coincidencias:
                    st.success(f"Hay {len(coincidencias)} artículos en común")
    
    elif btn_compare:
        st.warning("Por favor ingrese una consulta")


# ======================================================================
# FOOTER
# ======================================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><b>UPScholar</b> - Proyecto Primer Bimestre | Desarrollado con Streamlit</p>
        <p>Dataset: ICMLA 2014-2017 | Métodos: TF-IDF + Jaccard & LLM Embeddings (MPNet)</p>
    </div>
""", unsafe_allow_html=True)
