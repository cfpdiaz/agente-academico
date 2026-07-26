"""Agente RAG para consultas académicas usando Google Gemini."""
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from .config import Config
from .document_processor import DocumentProcessor

class AcademicAgent:
    """Agente de IA para consultas académicas usando RAG con Google Gemini."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_output_tokens=self.config.MAX_TOKENS,
            google_api_key=self.config.GOOGLE_API_KEY
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.config.GOOGLE_API_KEY
        )
        self.vector_store = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        self.qa_chain = None
        self._initialize()
    
    def _initialize(self):
        """Inicializa el procesador de documentos y la cadena de QA."""
        processor = DocumentProcessor(self.config)
        documents = processor.process_documents("data")
        
        if documents:
            self.vector_store = FAISS.from_documents(
                documents, self.embeddings
            )
            self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Configura la cadena de preguntas y respuestas."""
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.TOP_K}
        )
        
        template = """Eres un asistente académico inteligente que responde consultas sobre:
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
{chat_history}

Pregunta: {question}
Respuesta:"""
        
        prompt = PromptTemplate.from_template(template)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_prompt=prompt,
            return_source_documents=True,
            verbose=False
        )
    
    def query(self, question: str) -> Dict:
        """Procesa una consulta y devuelve la respuesta."""
        if not self.qa_chain:
            return {
                "answer": "El agente no está inicializado. Asegúrate de que los documentos están en la carpeta data/.",
                "sources": []
            }
        
        result = self.qa_chain({"question": question})
        
        sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Desconocido")
                category = doc.metadata.get("category", "General")
                if source not in [s["source"] for s in sources]:
                    sources.append({
                        "source": source,
                        "category": category
                    })
        
        return {
            "answer": result.get("answer", "No se pudo generar una respuesta."),
            "sources": sources
        }
    
    def clear_memory(self):
        """Limpia el historial de conversación."""
        self.memory.clear()
    
    def rebuild_index(self, data_dir: str = "data"):
        """Reconstruye el índice de documentos."""
        processor = DocumentProcessor(self.config)
        documents = processor.process_documents(data_dir)
        
        if documents:
            self.vector_store = FAISS.from_documents(
                documents, self.embeddings
            )
            self._setup_qa_chain()
            return True
        return False
