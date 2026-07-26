import os
from dataclasses import dataclass

@dataclass
class Config:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2000
    
    DOCUMENT_MAPPING = {
        "01_Reglamento_del_Estudiante": "Reglamento del Estudiante",
        "02_Politica_de_Reembolso_de_Matriculas": "Política de Reembolso",
        "03_Preguntas_Frecuentes_Cursos_y_Certificados": "Preguntas Frecuentes",
        "04_Guia_de_Uso_de_la_Plataforma": "Guía de Uso de la Plataforma",
        "05_Programa_de_Becas_y_Afiliados": "Programa de Becas y Afiliados",
        "06_Horarios_de_Clases_y_Ramos": "Horarios de Clases y Ramos"
    }
