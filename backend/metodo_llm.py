# -*- coding: utf-8 -*-
"""
Módulo de búsqueda con Embeddings LLM (MPNet) para UPScholar
"""

import pandas as pd
import numpy as np
import os
import time
import re
import difflib
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
vocab_corpus = None  # Nuevo: vocabulario para sugerencias

# ======================================================================
# FUNCIONES PARA SUGERENCIAS
# ======================================================================

def extraer_vocabulario_corpus():
    """Extrae todas las palabras únicas del corpus para sugerencias"""
    global doc, vocab_corpus
    
    if vocab_corpus is not None:
        return vocab_corpus
    
    if doc is None:
        return set()
    
    vocab_corpus = set()
    
    # Extraer palabras de abstracts (sin procesar, palabras originales)
    for text in doc['abstract'].fillna(''):
        palabras = re.sub(r'[^a-zA-Z\s]', ' ', str(text)).lower().split()
        vocab_corpus.update(palabras)
    
    for text in doc['keywords'].fillna(''):
        palabras = re.sub(r'[^a-zA-Z\s]', ' ', str(text)).lower().split()
        vocab_corpus.update(palabras)
    
    for text in doc['title'].fillna(''):
        palabras = re.sub(r'[^a-zA-Z\s]', ' ', str(text)).lower().split()
        vocab_corpus.update(palabras)
    
    # Filtrar palabras muy cortas
    vocab_corpus = {p for p in vocab_corpus if len(p) >= 3}
    
    return vocab_corpus

def sugerir_palabras(query, n=5):
    """
    Sugiere palabras similares si la query no tiene resultados
    
    Args:
        query (str): Consulta del usuario
        n (int): Número máximo de sugerencias por palabra
        
    Returns:
        list: Lista de sugerencias en formato "palabra_incorrecta → sugerencia"
    """
    vocab = extraer_vocabulario_corpus()
    
    if not vocab:
        return []
    
    palabras_query = re.sub(r'[^a-zA-Z\s]', ' ', query).lower().split()
    sugerencias = []
    
    for palabra in palabras_query:
        if len(palabra) < 3:  # Ignorar palabras muy cortas
            continue
        
        # Buscar palabras similares en el vocabulario
        matches = difflib.get_close_matches(palabra, vocab, n=n, cutoff=0.6)
        
        # Si la palabra no está en el vocabulario pero hay matches
        if matches and palabra not in vocab:
            sugerencias.append(f"{palabra} → {matches[0]}")
    
    return sugerencias[:5]  # Limitar a 5 sugerencias totales

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
        
        # Extraer vocabulario para sugerencias
        extraer_vocabulario_corpus()
        
        return "✅ Embeddings LLM generados y guardados exitosamente", True
        
    else:
        # --- FASE DE CARGA (RÁPIDO) ---
        print("\n" + "="*70)
        print("FASE DE CARGA: Cargando embeddings LLM desde disco...")
        print("="*70)
        
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"Dimensión del embedding cargado: {embeddings.shape}")
        
        sim_matrix = np.load(SIM_MATRIX_FILE)
        
        # Extraer vocabulario para sugerencias
        extraer_vocabulario_corpus()
        
        return "✅ Embeddings LLM cargados exitosamente", True

def buscar_llm(query, threshold_minimo=0.15):
    """
    Busca los Top 10 artículos más similares usando embeddings LLM
    
    Args:
        query (str): Consulta del usuario
        threshold_minimo (float): Similitud mínima para considerar resultados de calidad (default: 0.15)
        
    Returns:
        tuple: (idx_top10, similitudes, tiene_resultados, tiempo_ms, sugerencias)
    """
    global model, embeddings
    
    inicio = time.time()
    
    # Generar embedding de la query
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    
    # Calcular similitud con todos los documentos
    similitudes = cosine_similarity([query_vec], embeddings)[0]
    
    # SIEMPRE devolver Top 10
    idx_top10 = np.argsort(similitudes)[-10:][::-1]
    
    # Verificar si el MEJOR resultado supera el threshold
    mejor_similitud = similitudes[idx_top10[0]]
    
    tiempo_ms = (time.time() - inicio) * 1000
    
    # Si el mejor resultado es malo, generar sugerencias PERO IGUAL DEVOLVER LOS 10
    if mejor_similitud < threshold_minimo:
        sugerencias = sugerir_palabras(query)
        return idx_top10, similitudes, False, tiempo_ms, sugerencias  # False = baja calidad
    
    # Resultados de buena calidad
    return idx_top10, similitudes, True, tiempo_ms, []


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
