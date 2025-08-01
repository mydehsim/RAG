import os
import sys

def main():
    """Main function with error handling"""
    print("🚀 Starting RAG Pipeline...")
    
    try:
        # Import with error handling
        from config import setup_rag_system, build_database, search
        
        # Step 1: Setup system
        print("\n📋 Step 1: Setting up RAG system...")
        try:
            setup_success = setup_rag_system()
            if setup_success:
                print("✅ System setup completed successfully!")
            else:
                print("⚠️ System setup had some issues, but continuing...")
        except Exception as e:
            print(f"⚠️ System setup error: {e}")
            print("Continuing with existing setup...")

            # Step 2: Build database
        print("\n📋 Step 2: Building/updating database...")
        try:
            build_database()
            print("✅ Database build completed!")
        except Exception as e:
            print(f"❌ Database build failed: {e}")
            print("This might be due to:")
            print("  - Missing documents in data directory")
            print("  - Docling library version mismatch")
            print("  - Insufficient permissions")
            return False
        
        # Step 3: Test search
        print("\n📋 Step 3: Testing search functionality...")
        try:
            result = search("Give me all soccer rules below rule 5")
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