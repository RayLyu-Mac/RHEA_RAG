#!/usr/bin/env python3
"""
Diagnostic script to test SQLite and Chroma compatibility
"""

import sys
import os

def test_sqlite():
    """Test SQLite version and compatibility"""
    print("🔍 Testing SQLite compatibility...")
    
    try:
        import sqlite3
        print(f"✅ SQLite version: {sqlite3.sqlite_version}")
        
        # Check if version is sufficient
        version_parts = [int(x) for x in sqlite3.sqlite_version.split('.')]
        if version_parts[0] > 3 or (version_parts[0] == 3 and version_parts[1] >= 35):
            print("✅ SQLite version is sufficient for Chroma (≥ 3.35.0)")
        else:
            print("❌ SQLite version is too old for Chroma")
            return False
            
        # Test basic SQLite operations
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("INSERT INTO test (data) VALUES (?)", ("test_data",))
            cursor.execute("SELECT * FROM test")
            result = cursor.fetchone()
            print(f"✅ Basic SQLite operations work: {result}")
            conn.close()
        finally:
            os.unlink(db_path)
            
        return True
        
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return False

def test_chroma():
    """Test Chroma installation and basic functionality"""
    print("\n🔍 Testing Chroma compatibility...")
    
    try:
        import chromadb
        print(f"✅ ChromaDB version: {chromadb.__version__}")
        
        # Test basic Chroma operations
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                client = chromadb.PersistentClient(path=tmpdir)
                collection = client.create_collection("test_collection")
                collection.add(
                    documents=["This is a test document"],
                    metadatas=[{"source": "test"}],
                    ids=["test_id"]
                )
                results = collection.query(query_texts=["test document"], n_results=1)
                print(f"✅ Basic Chroma operations work: {results}")
                return True
            except Exception as e:
                print(f"❌ Chroma operations failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Chroma import failed: {e}")
        return False

def test_embeddings():
    """Test embeddings functionality"""
    print("\n🔍 Testing embeddings...")
    
    try:
        from langchain.embeddings import OllamaEmbeddings
        print("✅ OllamaEmbeddings import successful")
        
        # Try to create embeddings instance
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            print("✅ OllamaEmbeddings instance created")
            
            # Test embedding generation
            test_text = "This is a test sentence."
            try:
                embedding = embeddings.embed_query(test_text)
                print(f"✅ Embedding generation successful (dimension: {len(embedding)})")
                return True
            except Exception as e:
                print(f"❌ Embedding generation failed: {e}")
                print("💡 This might be because Ollama is not running or the model is not available")
                return False
                
        except Exception as e:
            print(f"❌ Failed to create embeddings instance: {e}")
            return False
            
    except Exception as e:
        print(f"❌ OllamaEmbeddings import failed: {e}")
        return False

def test_langchain_chroma():
    """Test LangChain Chroma integration"""
    print("\n🔍 Testing LangChain Chroma integration...")
    
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OllamaEmbeddings
        print("✅ LangChain Chroma import successful")
        
        # Test in-memory Chroma
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            vectorstore = Chroma(embedding_function=embeddings)
            print("✅ In-memory Chroma creation successful")
            return True
        except Exception as e:
            print(f"❌ In-memory Chroma creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ LangChain Chroma import failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🔬 RHEA Paper DB - SQLite/Chroma Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("SQLite", test_sqlite),
        ("Chroma", test_chroma),
        ("Embeddings", test_embeddings),
        ("LangChain Chroma", test_langchain_chroma)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your environment should work with the RHEA Paper DB.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\n💡 Common solutions:")
        print("1. Update packages: pip install --upgrade chromadb pysqlite3-binary")
        print("2. Ensure Ollama is running: ollama serve")
        print("3. Pull the embedding model: ollama pull nomic-embed-text:latest")
        print("4. Check Python environment: python --version")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 