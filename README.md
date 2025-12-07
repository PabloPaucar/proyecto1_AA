# UPScholar

Sistema de Búsqueda y Recomendación Inteligente de Documentos Científicos

## Descripción

UPScholar es un motor de búsqueda académico que recupera artículos científicos utilizando dos metodologías: TF-IDF + Jaccard y embeddings LLM (MPNet). El sistema retorna los 10 artículos más similares y recomienda 3 artículos adicionales por cada resultado.

## Características

- Búsqueda con matriz ponderada (TF-IDF 60%, Jaccard Keywords 25%, Jaccard Títulos 15%)
- Búsqueda con embeddings del modelo MPNet (768 dimensiones)
- Sistema de recomendación basado en similitud documento-documento
- Interfaz web interactiva con Streamlit
- Comparación lado a lado de ambos métodos

## Tecnologías

- Python 3.9+
- Streamlit (interfaz web)
- NLTK (procesamiento de lenguaje natural)
- Sentence-Transformers (embeddings LLM)
- Scikit-learn (métricas de similitud)

## Instalación

git clone https://github.com/tu-usuario/upscholar-project.git
cd upscholar-project
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

## Estructura del Proyecto

upscholar_project/

├── backend/

│ ├── metodo_tfidf.py

│ └── metodo_llm.py

├── data/

│ └── ICMLA_2014_2015_2016_2017.csv

├── app.py

└── requirements.txt


## Dataset

ICMLA Conference Papers (2014-2017) - IEEE International Conference on Machine Learning and Applications

## Uso

1. Seleccione método de búsqueda (TF-IDF o LLM)
2. Ingrese consulta académica
3. Explore los 10 resultados más relevantes
4. Revise las 3 recomendaciones por cada artículo

## Deployment

Compatible con Streamlit Community Cloud. La primera ejecución genera embeddings precalculados (5-10 minutos), posteriores ejecuciones cargan desde caché.

## Proyecto Académico

Desarrollado como Proyecto Primer Bimestre  
Universidad Politécnica Salesiana  
Ingeniería en Ciencias de la Computación  
2025
