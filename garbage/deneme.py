import os
import sys

def main():
    """Main function with error handling"""
    print("🚀 Starting RAG Pipeline...")
    
    try:
        # Import with error handling
        from config import setup_rag_system, build_database, search, show_documents
        
        
        # Step 3: Test search
        print("\n📋 Step 3: Testing search functionality...")
        try:
            result = search("CMDS projesinin uyumlu olduğu mühimmat konfigürasyonları nelerdir?", document=["srs.pdf","SRS.docx"])
            print("✅ Search test completed!")
            return True
        except Exception as e:
            print(f"❌ Search test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install langchain langchain-community langchain-ollama")
        print("  pip install chromadb")
        print("  pip install sentence-transformers")
        print("  pip install docling")  # This might be the problematic one
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    

main()    