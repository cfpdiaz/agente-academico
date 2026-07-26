"""Agente RAG para consultas académicas usando Google Gemini."""
import os
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict
from .config import Config
from .document_processor import DocumentProcessor

class AcademicAgent:
    """Agente de IA para consultas académicas usando RAG con Google Gemini."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        genai.configure(api_key=self.config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.config.MODEL_NAME)
        self.vector_store = None
        self.chunks = []
        self.metadata = []
        self.chat_history = []
        self._initialize()
    
    def _initialize(self):
        """Inicializa el procesador de documentos y crea el índice."""
        processor = DocumentProcessor(self.config)
        documents = processor.process_documents("data")
        
        if documents:
            self.chunks = [doc.page_content for doc in documents]
            self.metadata = [doc.metadata for doc in documents]
            self._create_vector_store()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Obtiene el embedding de un texto usando Google Gemini."""
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']
    
    def _create_vector_store(self):
        """Crea el índice FAISS con los embeddings de los chunks."""
        embeddings = []
        for chunk in self.chunks:
            embedding = self._get_embedding(chunk)
            embeddings.append(embedding)
        
        embeddings_np = np.array(embeddings).astype('float32')
        dimension = embeddings_np.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
    
    def _retrieve_relevant(self, query: str, k: int = 5) -> List[tuple]:
        """Recupera los chunks más relevantes para una consulta."""
        query_embedding = self._get_embedding(query)
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        
        scores, indices = self.index.search(query_np, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], self.metadata[idx], float(score)))
        
        return results
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Construye el prompt para el modelo."""
        history = ""
        for msg in self.chat_history[-6:]:  # Últimos 3 intercambios
            role = "Usuario" if msg["role"] == "user" else "Asistente"
            history += f"{role}: {msg['content']}\n"
        
        prompt = f"""Eres un asistente académico inteligente que responde consultas sobre:
- Matrículas y registro
- Horarios académicos
- Programas de beca
- Uso de la plataforma online
- Reglamento del estudiante

INSTRUCCIONES:
1. Responde basándote ÚNICAMENTE en la información de los documentos proporcionados
2. Si no sabes la respuesta, di: "No tengo información sobre eso en los documentos disponibles"
3. Sé claro, conciso y amable
4. Cita la fuente cuando sea posible (ej: "Según el reglamento...")
5. Si la pregunta es ambigua, pide clarificación

Contexto relevante:
{context}

Historial de conversación:
{history}

Pregunta: {query}
Respuesta:"""
        return prompt
    
    def query(self, question: str) -> Dict:
        """Procesa una consulta y devuelve la respuesta."""
        if not self.vector_store and not hasattr(self, 'index'):
            return {
                "answer": "El agente no está inicializado. Asegúrate de que los documentos están en la carpeta data/.",
                "sources": []
            }
        
        # Recuperar chunks relevantes
        relevant = self._retrieve_relevant(question, k=self.config.TOP_K)
        
        # Construir contexto
        context_parts = []
        sources = []
        seen_sources = set()
        
        for chunk, meta, score in relevant:
            context_parts.append(chunk)
            source = meta.get("source", "Desconocido")
            category = meta.get("category", "General")
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append({"source": source, "category": category})
        
        context = "\n\n".join(context_parts)
        
        # Construir prompt
        prompt = self._build_prompt(question, context)
        
        # Generar respuesta
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.TEMPERATURE,
                    max_output_tokens=self.config.MAX_TOKENS
                )
            )
            answer = response.text
        except Exception as e:
            answer = f"Error al generar la respuesta: {str(e)}"
        
        # Actualizar historial
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def clear_memory(self):
        """Limpia el historial de conversación."""
        self.chat_history = []
    
    def rebuild_index(self, data_dir: str = "data"):
        """Reconstruye el índice de documentos."""
        processor = DocumentProcessor(self.config)
        documents = processor.process_documents(data_dir)
        
        if documents:
            self.chunks = [doc.page_content for doc in documents]
            self.metadata = [doc.metadata for doc in documents]
            self._create_vector_store()
            return True
        return False

     
              
