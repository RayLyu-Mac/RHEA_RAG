"""
FastAPI Backend for Material Research RAG
Exposes Streamlit functionality as REST APIs for React frontend
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing Streamlit functions
try:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OllamaEmbeddings
    from langchain.llms import Ollama
    from utils.paper_correlations import PaperCorrelationManager, initialize_sss_correlations
    import pandas as pd
    import networkx as nx
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

app = FastAPI(
    title="Material Research RAG API",
    description="REST API for Material Research RAG System",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    question: str
    selected_papers: Optional[List[str]] = None
    search_type: str = "both"
    k: int = 5
    llm_model: str = "qwen3:14b"

class PaperInfo(BaseModel):
    file_name: str
    figure_count: int
    has_figures: bool
    folder: str

class SearchResult(BaseModel):
    file_name: str
    section: str
    content: str
    document_type: str
    figure_count: int

class SearchResponse(BaseModel):
    answer: str
    results: List[SearchResult]
    total_results: int

class CorrelationInfo(BaseModel):
    source: str
    target: str
    relationship_type: str
    description: str
    strength: float
    evidence: Optional[str] = None

# Global state (in production, use proper state management)
vectorstore = None
llm = None
correlation_manager = None
paper_list = []

def load_vectorstore():
    """Load the vector store"""
    global vectorstore
    if vectorstore is None:
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            persist_directory = "VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectorstore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load vector store: {e}")
    return vectorstore

def load_llm(model_name: str):
    """Load the LLM model"""
    global llm
    try:
        llm = Ollama(model=model_name)
        return llm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM model {model_name}: {e}")

def load_paper_list():
    """Load the list of papers"""
    global paper_list
    if not paper_list:
        try:
            tracker_path = "./vectorization_tracker.csv"
            if os.path.exists(tracker_path):
                df = pd.read_csv(tracker_path)
                vectorized_papers = df[df['vectorized'] == True]
                paper_list = []
                for _, row in vectorized_papers.iterrows():
                    paper_info = {
                        'file_name': row['file_name'],
                        'file_path': row['file_path'],
                        'figure_count': row.get('figure_count', 0),
                        'has_figures': row.get('has_figure_descriptions', False),
                        'folder': os.path.basename(os.path.dirname(row['file_path']))
                    }
                    paper_list.append(paper_info)
            return paper_list
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load paper list: {e}")
    return paper_list

def search_papers(question: str, selected_papers: Optional[List[str]] = None, 
                 search_type: str = "both", k: int = 5):
    """Search papers and generate answer"""
    if not vectorstore:
        load_vectorstore()
    
    try:
        # If specific papers are selected, search with a broader query and larger k
        if selected_papers and len(selected_papers) > 0:
            search_results = vectorstore.similarity_search(question, k=k*4)
            
            # Filter by selected papers
            filtered_results = [
                doc for doc in search_results 
                if doc.metadata.get('file_name') in selected_papers
            ]
            
            search_results = filtered_results
        else:
            search_results = vectorstore.similarity_search(question, k=k*2)

        # Filter by document type
        if search_type == "parent":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'parent']
        elif search_type == "child":
            search_results = [doc for doc in search_results if doc.metadata.get('document_type') == 'child']
        
        # Take only k results
        search_results = search_results[:k]
        
        if not search_results:
            return "No relevant documents found for your question.", []
        
        # Generate answer using LLM
        if llm:
            # Prepare context
            context_parts = []
            for i, doc in enumerate(search_results):
                paper_name = doc.metadata.get('file_name', 'Unknown Paper')
                doc_type = doc.metadata.get('document_type', 'unknown')
                section = doc.metadata.get('section', 'Unknown Section')
                
                context_parts.append(f"[Document {i+1}] ({doc_type.upper()}) {paper_name} - {section}")
                content = doc.page_content[:1500] + "..." if len(doc.page_content) > 1500 else doc.page_content
                context_parts.append(content)
                context_parts.append("---")
            
            combined_context = "\n".join(context_parts)
            
            prompt = f"""You are a materials science research expert. Based on the following context from scientific papers, answer the user's question comprehensively and accurately.

Context from papers:
{combined_context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context provided
2. Focus on materials science concepts, mechanisms, and relationships
3. If figures are mentioned in the context, reference them appropriately
4. Use technical terminology appropriately
5. Structure your answer clearly with main points and supporting details
6. If specific papers are mentioned, cite them in your response

Answer:"""
            
            answer = llm.invoke(prompt)
            return answer, search_results
        else:
            return "LLM not available. Please check your model selection.", search_results
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Material Research RAG API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "vectorstore_loaded": vectorstore is not None}

@app.get("/papers", response_model=Dict[str, Any])
async def get_papers():
    """Get all papers"""
    try:
        papers = load_paper_list()
        
        # Group papers by folder
        papers_by_folder = {}
        for paper in papers:
            folder = paper['folder']
            if folder not in papers_by_folder:
                papers_by_folder[folder] = []
            papers_by_folder[folder].append(PaperInfo(**paper))
        
        return {
            "papers": papers_by_folder,
            "total_papers": len(papers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search papers and generate answer"""
    try:
        # Load LLM if needed
        global llm
        if llm is None or getattr(llm, 'model_name', None) != request.llm_model:
            llm = load_llm(request.llm_model)
        
        # Perform search
        answer, results = search_papers(
            request.question, 
            request.selected_papers, 
            request.search_type, 
            request.k
        )
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append(SearchResult(
                file_name=doc.metadata.get('file_name', 'Unknown'),
                section=doc.metadata.get('section', 'Unknown'),
                content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                document_type=doc.metadata.get('document_type', 'unknown'),
                figure_count=doc.metadata.get('figure_count', 0)
            ))
        
        return SearchResponse(
            answer=answer,
            results=formatted_results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get available Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = []
            for model in models_data.get("models", []):
                model_name = model.get("name", "")
                if model_name:
                    models.append(model_name)
            
            # Filter for common LLM models
            llm_models = []
            for model in models:
                if any(keyword in model.lower() for keyword in ["qwen", "gemma", "llama", "mistral", "codellama", "phi", "vicuna", "alpaca"]):
                    llm_models.append(model)
            
            return {"models": sorted(llm_models) if llm_models else ["qwen3:14b", "gemma3:4b"]}
        else:
            return {"models": ["qwen3:14b", "gemma3:4b"]}
    except Exception as e:
        return {"models": ["qwen3:14b", "gemma3:4b"], "error": str(e)}

@app.get("/correlations")
async def get_correlations(topic: Optional[str] = None):
    """Get paper correlations"""
    try:
        global correlation_manager
        if correlation_manager is None:
            correlation_manager = PaperCorrelationManager()
            if "Solid Solution Strengthening (SSS)" not in correlation_manager.topics:
                correlation_manager = initialize_sss_correlations()
        
        if topic:
            df = correlation_manager.export_to_dataframe(topic)
        else:
            df = correlation_manager.export_to_dataframe()
        
        return {
            "correlations": df.to_dict('records'),
            "topics": list(correlation_manager.topics.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network")
async def get_network_data(topic: Optional[str] = None):
    """Get network visualization data"""
    try:
        global correlation_manager
        if correlation_manager is None:
            correlation_manager = PaperCorrelationManager()
            if "Solid Solution Strengthening (SSS)" not in correlation_manager.topics:
                correlation_manager = initialize_sss_correlations()
        
        if topic:
            G = correlation_manager.create_network_graph(topic)
        else:
            G = correlation_manager.create_network_graph()
        
        # Convert networkx graph to JSON-serializable format
        nodes = [{"id": node, "connections": len(list(G.neighbors(node)))} 
                for node in G.nodes()]
        edges = [{"source": edge[0], "target": edge[1], "data": edge[2]} 
                for edge in G.edges(data=True)]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        papers = load_paper_list()
        
        # Calculate stats
        total_papers = len(papers)
        total_figures = sum(paper['figure_count'] for paper in papers)
        papers_with_figures = sum(1 for paper in papers if paper['figure_count'] > 0)
        
        # Group by folder
        folder_stats = {}
        for paper in papers:
            folder = paper['folder']
            if folder not in folder_stats:
                folder_stats[folder] = {"count": 0, "figures": 0}
            folder_stats[folder]["count"] += 1
            folder_stats[folder]["figures"] += paper['figure_count']
        
        return {
            "total_papers": total_papers,
            "total_figures": total_figures,
            "papers_with_figures": papers_with_figures,
            "folder_stats": folder_stats,
            "vectorstore_loaded": vectorstore is not None,
            "llm_loaded": llm is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Load initial data
    print("Loading initial data...")
    try:
        load_vectorstore()
        load_paper_list()
        print("✅ Initial data loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load initial data: {e}")
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
