"""
Custom embedder configuration for CrewAI with local models.
Provides fallback strategies for knowledge management.
"""

import os
from typing import Optional, Union

# Robust import strategy with fallbacks
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    print("⚠️  langchain-ollama not found. Install with: pip install langchain-ollama")
    try:
        # Fallback to older langchain-community implementation
        from langchain_community.embeddings import OllamaEmbeddings
        OLLAMA_AVAILABLE = True
        print("✅ Using langchain-community.embeddings.OllamaEmbeddings as fallback")
    except ImportError:
        print("❌ No Ollama embeddings available. Install langchain-ollama or langchain-community")
        OLLAMA_AVAILABLE = False
        # Create a mock class for graceful degradation
        class OllamaEmbeddings:
            def __init__(self, *args, **kwargs):
                raise ImportError("Ollama embeddings not available - install langchain-ollama")

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

class LocalEmbedderConfig:
    """
    Configures local embedding functions for CrewAI knowledge storage.
    Eliminates dependency on OpenAI embedding services.
    
    Installation Requirements:
    - pip install langchain-ollama
    - Ollama service running on localhost:11434
    - Embedding model pulled: ollama pull nomic-embed-text
    """
    
    def __init__(self, 
                 model_name: str = "nomic-embed-text:latest",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize local embedder configuration.
        
        Args:
            model_name: Ollama embedding model name
            base_url: Ollama service URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self._embedder = None
        
        if not OLLAMA_AVAILABLE:
            print(f"🔧 Setup Instructions:")
            print(f"   1. pip install langchain-ollama")
            print(f"   2. ollama pull {model_name}")
            print(f"   3. Ensure Ollama is running on {base_url}")
    
    def get_embedder(self) -> Optional[OllamaEmbeddings]:
        """
        Get or create Ollama embeddings instance.
        
        Returns:
            Configured OllamaEmbeddings instance or None if unavailable
        """
        if not OLLAMA_AVAILABLE:
            return None
            
        if self._embedder is None:
            try:
                self._embedder = OllamaEmbeddings(
                    model=self.model_name,
                    base_url=self.base_url
                )
            except Exception as e:
                print(f"❌ Failed to initialize Ollama embedder: {e}")
                return None
        return self._embedder
    
    def configure_crew_embedder(self) -> Optional[OllamaEmbeddings]:
        """
        Configure embedder for CrewAI crew initialization.
        
        Returns:
            Embedder instance or None if local model unavailable
        """
        try:
            # Verify Ollama service availability
            embedder = self.get_embedder()
            # Test embedding functionality
            test_embedding = embedder.embed_query("test")
            if test_embedding:
                return embedder
        except Exception as e:
            print(f"⚠️  Local embedder unavailable: {e}")
            print("💡 Falling back to placeholder configuration")
        
        return None

# Usage in your crew configuration
def get_configured_embedder():
    """Factory function for embedder configuration."""
    local_config = LocalEmbedderConfig()
    return local_config.configure_crew_embedder()