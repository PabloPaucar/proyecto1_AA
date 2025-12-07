# -*- coding: utf-8 -*-
"""
Módulo de búsqueda con Embeddings LLM (MPNet) para UPScholar
"""

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Directorio para persistencia
PERSISTENCE_DIR = 'upscholar_data_llm'
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

# Variables globales (caché)
doc = None
model = None
embeddings = None
sim_matrix = None

# ======================================================================
# FUNCIONES PÚBLICAS PARA STREAMLIT
# ======================================================================

def cargar_dataset(csv_path='data/ICMLA_2014_2015_2016_2017.csv'):
    """Carga el dataset CSV"""
    global doc
    if doc is None:
        doc = pd.read_csv(csv_path, encoding="latin1")
    return doc


def inicializar_llm(csv_path='data/ICMLA_2014_2015_2016_2017.csv'):
    """
    Inicializa el sistema LLM: carga modelo y embeddings
    Retorna: (mensaje_estado, es_exitoso)
    """
    global doc, model, embeddings, sim_matrix
    
    # 1. Cargar dataset
    doc = cargar_dataset(csv_path)
    abstracts = doc["abstract"].astype(str).tolist()
    
    print(f"Cantidad de documentos cargados: {len(abstracts)}")
    
    # 2. Cargar modelo (siempre necesario para queries)
    print("\nCargando modelo de embeddings MPNet...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # 3. Cargar o generar embeddings
    EMBEDDINGS_FILE = os.path.join(PERSISTENCE_DIR, 'embeddings_llm.npy')
    SIM_MATRIX_FILE = os.path.join(PERSISTENCE_DIR, 'sim_matrix_llm.npy')
    
    if not os.path.exists(EMBEDDINGS_FILE):
        # --- FASE DE CÁLCULO (LENTO) ---
        print("\n" + "="*70)
        print("FASE DE CÁLCULO: Generando embeddings LLM...")
        print("="*70)
        
        # Generar embeddings
        embeddings = model.encode(abstracts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings)
        print(f"Dimensión del embedding generado: {embeddings.shape}")
        
        # Calcular matriz de similitud Doc vs Doc
        print("Calculando matriz de similitud Documento vs Documento...")
        sim_matrix = cosine_similarity(embeddings, embeddings)
        
        # Guardar
        print("Guardando artefactos...")
        np.save(EMBEDDINGS_FILE, embeddings)
        np.save(SIM_MATRIX_FILE, sim_matrix)
        
        return "✅ Embeddings LLM generados y guardados exitosamente", True
        
    else:
        # --- FASE DE CARGA (RÁPIDO) ---
        print("\n" + "="*70)
        print("FASE DE CARGA: Cargando embeddings LLM desde disco...")
        print("="*70)
        
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"Dimensión del embedding cargado: {embeddings.shape}")
        
        sim_matrix = np.load(SIM_MATRIX_FILE)
        
        return "✅ Embeddings LLM cargados exitosamente", True


def buscar_llm(query, threshold_minimo=0.05):
    """
    Busca los Top 10 artículos más similares usando embeddings LLM
    
    Args:
        query (str): Consulta del usuario
        threshold_minimo (float): Similitud mínima para considerar que hay resultados (default: 0.05)
        
    Returns:
        tuple: (idx_top10, similitudes, tiene_resultados)
    """
    global model, embeddings
    
    # Generar embedding de la query
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    
    # Calcular similitud con todos los documentos
    similitudes = cosine_similarity([query_vec], embeddings)[0]
    
    # Top 10 (siempre los 10 mejores)
    idx_top10 = np.argsort(similitudes)[-10:][::-1]
    
    # Verificar si el MEJOR resultado supera el threshold mínimo
    mejor_similitud = similitudes[idx_top10[0]]
    
    if mejor_similitud < threshold_minimo:
        # Todos los resultados son extremadamente malos
        return np.array([]), similitudes, False
    
    # Hay al menos un resultado decente, retornar los 10 mejores
    return idx_top10, similitudes, True



def recomendar_llm(idx_articulo, idx_top10):
    """
    Recomienda 3 artículos similares a uno dado
    
    Args:
        idx_articulo (int): Índice del artículo base
        idx_top10 (array): Índices del Top 10 (para excluir)
        
    Returns:
        list: [(idx, score), ...] - 3 recomendaciones
    """
    global sim_matrix
    
    # Similitud del artículo base con todos
    sims = sim_matrix[idx_articulo]
    
    # Excluir el mismo artículo y los del Top 10
    prohibidos = set(idx_top10) | {idx_articulo}
    candidatos = [(i, sims[i]) for i in range(len(sims)) if i not in prohibidos]
    
    # Top 3 recomendados
    candidatos = sorted(candidatos, key=lambda x: x[1], reverse=True)[:3]
    
    return candidatos


def get_dataset():
    """Retorna el dataset cargado"""
    return doc
