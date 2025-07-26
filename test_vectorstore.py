#!/usr/bin/env python3
"""
Simple test script to debug vector store loading issues.
Run this script to see what's happening with the vector store loading.
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vectorstore_loading():
    """Test the vector store loading function"""
    print("🔍 Testing vector store loading...")
    
    # Test 1: Check if the persist directory exists
    persist_directory = "./VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child"
    print(f"📁 Checking persist directory: {persist_directory}")
    print(f"📁 Directory exists: {os.path.exists(persist_directory)}")
    
    if os.path.exists(persist_directory):
        print(f"📁 Directory contents: {os.listdir(persist_directory)}")
    
    # Test 2: Check SQLite version
    import sqlite3
    sqlite_version = sqlite3.sqlite_version
    print(f"📊 SQLite version: {sqlite_version}")
    
    # Test 3: Try to import SentenceTransformer
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        print("✅ SentenceTransformer import successful")
        
        # Test 4: Try to create embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code": True})
        print("✅ SentenceTransformer embeddings created successfully")
        
        # Test 5: Try to import Chroma
        from langchain_community.vectorstores import Chroma
        print("✅ Chroma import successful")
        
        # Test 6: Try to load Chroma
        if os.path.exists(persist_directory):
            try:
                vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                print("✅ Chroma vector store loaded successfully")
                
                # Test 7: Try to get collection info
                try:
                    collection = vectorstore._collection
                    if collection:
                        count = collection.count()
                        print(f"✅ Collection has {count} documents")
                    else:
                        print("⚠️ Collection is None")
                except Exception as e:
                    print(f"❌ Error getting collection info: {e}")
                
            except Exception as e:
                print(f"❌ Error loading Chroma: {e}")
        else:
            print("❌ Persist directory does not exist")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_vectorstore_loading() 