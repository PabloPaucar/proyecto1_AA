"""
Backend de UPScholar
Módulos de búsqueda y recomendación
"""

from . import metodo_tfidf
from . import metodo_llm

__all__ = ['metodo_tfidf', 'metodo_llm']
