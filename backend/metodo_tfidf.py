# -*- coding: utf-8 -*-
"""
Módulo de búsqueda con TF-IDF y Jaccard para UPScholar
"""

import re
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import pickle
import os
import time
import difflib

# Descargar recursos NLTK
nltk.download('stopwords', quiet=True)

# Directorio para persistencia
PERSISTENCE_DIR = 'upscholar_data_tfidf'
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

# Variables globales (caché)
doc = None
vectors_abs = None
matriz_final = None
idf_abs = None
vocab_abs = None
keywords_tokens = None
titulos_tokens = None
abstracts_tokens = None
vocab_corpus = None  # Nuevo: vocabulario para sugerencias

# ======================================================================
# FUNCIONES DE PROCESAMIENTO NLP Y CÁLCULO
# ======================================================================

def procesar_nlp(textos):
    """Preprocesa textos: limpieza, tokenización, stopwords, stemming"""
    limpio = [re.sub(r'[^a-zA-Z]', ' ', str(t)).lower() for t in textos]
    tokens = [t.split() for t in limpio]
    sw = set(stopwords.words("english"))
    tokens_sw = [[w for w in tok if w not in sw] for tok in tokens]
    ps = PorterStemmer()
    return [[ps.stem(w) for w in tok] for tok in tokens_sw]

def compute_tf_log(tokens):
    """Calcula TF logarítmico"""
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return {t: 1 + math.log10(c) for t, c in counts.items()}

def compute_idf(doc_tokens_list):
    """Calcula IDF para todos los términos"""
    N = len(doc_tokens_list)
    vocab = sorted(set(t for doc in doc_tokens_list for t in doc))
    df = {term: 0 for term in vocab}
    for term in vocab:
        df[term] = sum(1 for doc in doc_tokens_list if term in doc)
    idf = {term: math.log10(N / df[term]) if df[term] > 0 else 0 for term in vocab}
    return idf, vocab

def compute_tfidf_vector(tokens, idf, vocab):
    """Genera vector TF-IDF normalizado"""
    tf = compute_tf_log(tokens)
    vec = np.array([tf.get(term, 0) * idf.get(term, 0) for term in vocab], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def compute_cosine_matrix(doc_tokens_list):
    """Calcula matriz de similitud coseno TF-IDF"""
    idf, vocab = compute_idf(doc_tokens_list)
    vectors = np.array([compute_tfidf_vector(tokens, idf, vocab)
                        for tokens in doc_tokens_list])
    dot = vectors @ vectors.T
    norms = np.linalg.norm(vectors, axis=1)
    denom = np.outer(norms, norms) + 1e-10
    sim = dot / denom
    sim = np.clip(sim, 0, 1)
    return sim, idf, vocab, vectors

def jaccard(tokens):
    """Calcula matriz de similitud Jaccard"""
    n = len(tokens)
    M = np.zeros((n, n))
    for i in range(n):
        s1 = set(tokens[i])
        for j in range(n):
            s2 = set(tokens[j])
            inter = len(s1 & s2)
            union = len(s1 | s2)
            M[i, j] = inter / union if union > 0 else 0
    return M

def jaccard_query(query_tokens, docs_tokens):
    """Calcula similitud Jaccard entre query y documentos"""
    sims = []
    qs = set(query_tokens)
    for d in docs_tokens:
        ds = set(d)
        inter = len(qs & ds)
        union = len(qs | ds)
        sims.append(inter / union if union > 0 else 0)
    return np.array(sims)

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
    
    # Extraer palabras de abstracts, keywords y títulos (sin procesar)
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

def inicializar_tfidf(csv_path='data/ICMLA_2014_2015_2016_2017.csv'):
    """
    Inicializa el sistema TF-IDF: carga o genera artefactos
    Retorna: (mensaje_estado, es_exitoso)
    """
    global doc, vectors_abs, matriz_final, idf_abs, vocab_abs
    global keywords_tokens, titulos_tokens, abstracts_tokens
    
    doc = cargar_dataset(csv_path)
    CHECK_FILE = os.path.join(PERSISTENCE_DIR, 'matriz_final_tfidf.npy')
    
    if not os.path.exists(CHECK_FILE):
        # --- FASE DE CÁLCULO (LENTO) ---
        print("\n" + "="*70)
        print("FASE DE CÁLCULO: Generando artefactos TF-IDF...")
        print("="*70)
        
        # 1. Generación de Tokens
        print("Procesando NLP...")
        abstracts_tokens = procesar_nlp(doc["abstract"])
        keywords_tokens = procesar_nlp(doc["keywords"])
        titulos_tokens = procesar_nlp(doc["title"])
        
        # 2. TF-IDF Coseno (Abstracts)
        print("Calculando matriz TF-IDF Coseno (abstracts)...")
        matriz_abstract, idf_abs, vocab_abs, vectors_abs = compute_cosine_matrix(abstracts_tokens)
        
        # 3. Jaccard (Keywords y Titles)
        print("Calculando Jaccard (keywords)...")
        matriz_keywords = jaccard(keywords_tokens)
        print("Calculando Jaccard (titles)...")
        matriz_titulos = jaccard(titulos_tokens)
        
        # 4. Matriz Final Ponderada
        matriz_final = (
            0.60 * matriz_abstract +
            0.25 * matriz_keywords +
            0.15 * matriz_titulos
        )
        np.fill_diagonal(matriz_final, 1)
        
        # 5. GUARDADO
        print("\nGuardando artefactos...")
        np.save(os.path.join(PERSISTENCE_DIR, 'vectors_abs.npy'), vectors_abs)
        np.save(os.path.join(PERSISTENCE_DIR, 'matriz_final_tfidf.npy'), matriz_final)
        
        with open(os.path.join(PERSISTENCE_DIR, 'idf_abs.pkl'), 'wb') as f:
            pickle.dump(idf_abs, f)
        with open(os.path.join(PERSISTENCE_DIR, 'vocab_abs.pkl'), 'wb') as f:
            pickle.dump(vocab_abs, f)
        with open(os.path.join(PERSISTENCE_DIR, 'keywords_tokens.pkl'), 'wb') as f:
            pickle.dump(keywords_tokens, f)
        with open(os.path.join(PERSISTENCE_DIR, 'titulos_tokens.pkl'), 'wb') as f:
            pickle.dump(titulos_tokens, f)
        
        # Extraer vocabulario para sugerencias
        extraer_vocabulario_corpus()
        
        return "✅ Artefactos TF-IDF generados y guardados exitosamente", True
        
    else:
        # --- FASE DE CARGA (RÁPIDO) ---
        print("\n" + "="*70)
        print("FASE DE CARGA: Cargando artefactos TF-IDF desde disco...")
        print("="*70)
        
        vectors_abs = np.load(os.path.join(PERSISTENCE_DIR, 'vectors_abs.npy'))
        matriz_final = np.load(os.path.join(PERSISTENCE_DIR, 'matriz_final_tfidf.npy'))
        
        with open(os.path.join(PERSISTENCE_DIR, 'idf_abs.pkl'), 'rb') as f:
            idf_abs = pickle.load(f)
        with open(os.path.join(PERSISTENCE_DIR, 'vocab_abs.pkl'), 'rb') as f:
            vocab_abs = pickle.load(f)
        with open(os.path.join(PERSISTENCE_DIR, 'keywords_tokens.pkl'), 'rb') as f:
            keywords_tokens = pickle.load(f)
        with open(os.path.join(PERSISTENCE_DIR, 'titulos_tokens.pkl'), 'rb') as f:
            titulos_tokens = pickle.load(f)
        
        abstracts_tokens = []  # No usado en queries
        
        # Extraer vocabulario para sugerencias
        extraer_vocabulario_corpus()
        
        return "✅ Artefactos TF-IDF cargados exitosamente", True

def buscar_tfidf(query, threshold_minimo=0.03):
    """
    Busca los Top 10 artículos más similares usando TF-IDF
    
    Args:
        query (str): Consulta del usuario
        threshold_minimo (float): Similitud mínima para considerar resultados de calidad
        
    Returns:
        tuple: (idx_top10, sim_query_final, tiene_resultados, tiempo_ms, sugerencias)
    """
    global vectors_abs, idf_abs, vocab_abs, keywords_tokens, titulos_tokens
    
    inicio = time.time()
    
    # Procesar query
    query_abs = procesar_nlp([query])[0]
    query_key = procesar_nlp([query])[0]
    query_tit = procesar_nlp([query])[0]
    
    # TF-IDF para abstracts
    query_vec_abs = compute_tfidf_vector(query_abs, idf_abs, vocab_abs)
    sim_query_abs = (vectors_abs @ query_vec_abs) / (
        np.linalg.norm(vectors_abs, axis=1) * np.linalg.norm(query_vec_abs) + 1e-10
    )
    
    # Jaccard para keywords y titles
    sim_query_key = jaccard_query(query_key, keywords_tokens)
    sim_query_tit = jaccard_query(query_tit, titulos_tokens)
    
    # Similitud final ponderada
    sim_query_final = (
        0.60 * sim_query_abs +
        0.25 * sim_query_key +
        0.15 * sim_query_tit
    )
    
    # SIEMPRE devolver Top 10
    idx_top10 = np.argsort(sim_query_final)[-10:][::-1]
    
    # Verificar similitud del MEJOR resultado
    mejor_similitud = sim_query_final[idx_top10[0]]
    
    tiempo_ms = (time.time() - inicio) * 1000
    
    # ============================================
    # NUEVO: CHEQUEO INDEPENDIENTE DE ORTOGRAFÍA
    # ============================================
    sugerencias_ortografia = sugerir_palabras(query)
    
    # Si hay errores ortográficos, SIEMPRE sugerir
    if sugerencias_ortografia:
        tiene_calidad = mejor_similitud >= threshold_minimo
        return idx_top10, sim_query_final, tiene_calidad, tiempo_ms, sugerencias_ortografia
    
    # Si NO hay errores pero similitud es baja
    if mejor_similitud < threshold_minimo:
        return idx_top10, sim_query_final, False, tiempo_ms, []
    
    # Todo bien
    return idx_top10, sim_query_final, True, tiempo_ms, []


def recomendar_tfidf(idx_articulo, idx_top10):
    """
    Recomienda 3 artículos similares a uno dado
    
    Args:
        idx_articulo (int): Índice del artículo base
        idx_top10 (array): Índices del Top 10 (para excluir)
        
    Returns:
        list: [(idx, score), ...] - 3 recomendaciones
    """
    global matriz_final
    
    sims_doc = matriz_final[idx_articulo]
    prohibidos = set(idx_top10) | {idx_articulo}
    candidatos = [(i, sims_doc[i]) for i in range(len(sims_doc)) if i not in prohibidos]
    candidatos = sorted(candidatos, key=lambda x: x[1], reverse=True)[:3]
    
    return candidatos

def get_dataset():
    """Retorna el dataset cargado"""
    return doc
