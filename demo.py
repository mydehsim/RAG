#!/usr/bin/env python3
"""
RAG Pipeline Demo Script
========================

This script demonstrates how to use the professional RAG pipeline
with your documents. It shows all the main features and provides
a step-by-step walkthrough.

Usage:
    python demo_rag_pipeline.py
"""

import os
import time
from typing import List, Dict, Any

# Import our RAG pipeline components
from config import (
    RAGPipeline, 
    RAGPipelineConfig,
    setup_rag_system,
    build_database,
    search,
    search_multiple_docs,
    show_documents,
    validate_setup
)


class RAGDemo:
    """Demo class that showcases RAG pipeline capabilities."""
    
    def __init__(self):
        """Initialize demo with configuration."""
        self.pipeline = None
        self.demo_queries = [
            "What are the main system requirements?",
            "How do I install the software?",
            "What are the testing procedures?",
            "What is the project scope?",
            "Are there any performance requirements?"
        ]
    
    def print_header(self, title: str, char: str = "=") -> None:
        """Print a formatted header."""
        print(f"\n{char * 60}")
        print(f"{title.center(60)}")
        print(f"{char * 60}")
    
    def print_step(self, step: str, description: str) -> None:
        """Print a step with description."""
        print(f"\n🔹 {step}")
        print(f"   {description}")
    
    def wait_for_user(self, message: str = "Press Enter to continue...") -> None:
        """Wait for user input."""
        input(f"\n{message}")
    
    def demo_system_setup(self) -> bool:
        """Demonstrate system setup."""
        self.print_header("STEP 1: SYSTEM SETUP")
        
        print("Setting up the RAG system with Ollama and required models...")
        print("This includes:")
        print("- Installing Ollama (if not present)")
        print("- Starting Ollama server")
        print("- Downloading required LLM model")
        
        self.wait_for_user("Ready to setup system? Press Enter...")
        
        success = setup_rag_system()
        
        if success:
            print("✅ System setup completed successfully!")
        else:
            print("❌ System setup failed. Check the error messages above.")
            return False
        
        self.wait_for_user()
        return True
    
    def demo_configuration(self) -> None:
        """Demonstrate configuration options."""
        self.print_header("STEP 2: CONFIGURATION OVERVIEW")
        
        config = RAGPipelineConfig()
        
        print("Current Configuration:")
        print(f"📁 Document Path: {config.DATA_PATH}")
        print(f"💾 Database Path: {config.CHROMA_PATH}")
        print(f"🧠 Embedding Model: {config.CURRENT_EMBEDDING_MODEL}")
        print(f"🤖 LLM Model: {config.CURRENT_LLM['model']}")
        print(f"📄 Chunk Size: {config.CHUNK_SIZE}")
        print(f"🔄 Chunk Overlap: {config.CHUNK_OVERLAP}")
        
        print("\nSupported File Types:")
        extensions = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        print(f"   {extensions}")
        
        print("\nTo customize configuration, edit the RAGPipelineConfig class:")
        print("- Change DATA_PATH to your document directory")
        print("- Select different embedding models for different languages")
        print("- Adjust chunk sizes for your use case")
        print("- Enable/disable features like OCR or table extraction")
        
        self.wait_for_user()
    
    def demo_validation(self) -> bool:
        """Demonstrate system validation."""
        self.print_header("STEP 3: SYSTEM VALIDATION")
        
        print("Validating the system setup...")
        
        success = validate_setup()
        
        if success:
            print("✅ Validation passed! System is ready to use.")
        else:
            print("❌ Validation failed. Please check the issues above.")
            print("\nCommon fixes:")
            print("- Ensure your DATA_PATH contains supported documents")
            print("- Check file permissions")
            print("- Verify disk space availability")
            return False
        
        self.wait_for_user()
        return True
    
    def demo_database_building(self) -> None:
        """Demonstrate database building process."""
        self.print_header("STEP 4: BUILDING DOCUMENT DATABASE")
        
        print("Building the document database...")
        print("This process will:")
        print("- Scan your document directory for supported files")
        print("- Process each document (extract text, tables, etc.)")
        print("- Split documents into searchable chunks")
        print("- Generate embeddings for semantic search")
        print("- Store everything in ChromaDB vector database")
        
        self.wait_for_user("Ready to build database? This may take a few minutes...")
        
        start_time = time.time()
        build_database()
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"\n⏱️ Database building completed in {processing_time:.2f} seconds")
        
        self.wait_for_user()
    
    def demo_database_info(self) -> None:
        """Demonstrate database information display."""
        self.print_header("STEP 5: DATABASE OVERVIEW")
        
        print("Here's what's in your database:")
        show_documents()
        
        # Get detailed stats
        pipeline = RAGPipeline()
        stats = pipeline.get_database_stats()
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total files: {len(stats['files'])}")
        print(f"   Total directories: {len(stats['directories'])}")
        
        avg_chunks_per_file = stats['total_chunks'] / len(stats['files']) if stats['files'] else 0
        print(f"   Average chunks per file: {avg_chunks_per_file:.1f}")
        
        self.wait_for_user()
    
    def demo_basic_search(self) -> None:
        """Demonstrate basic search functionality."""
        self.print_header("STEP 6: BASIC SEARCH DEMO")
        
        print("Now let's try some searches!")
        print("We'll demonstrate different types of queries...")
        
        for i, query in enumerate(self.demo_queries, 1):
            print(f"\n--- Search Example {i}/{len(self.demo_queries)} ---")
            print(f"Query: '{query}'")
            
            self.wait_for_user("Press Enter to run this search...")
            
            start_time = time.time()
            result = search(query)
            end_time = time.time()
            
            search_time = end_time - start_time
            print(f"\n⏱️ Search completed in {search_time:.2f} seconds")
            
            if i < len(self.demo_queries):
                self.wait_for_user("Press Enter for next search...")
    
    def demo_filtered_search(self) -> None:
        """Demonstrate filtered search capabilities."""
        self.print_header("STEP 7: FILTERED SEARCH DEMO")
        
        # Get available files for demo
        pipeline = RAGPipeline()
        stats = pipeline.get_database_stats()
        
        if not stats['files']:
            print("No files available for filtered search demo.")
            return
        
        # Demo document-specific search
        print("🔍 Document-specific search:")
        first_file = stats['files'][0]
        filename = os.path.basename(first_file)
        print(f"Searching only in: {filename}")
        
        query = "What is this document about?"
        print(f"Query: '{query}'")
        
        self.wait_for_user("Press Enter to run document-specific search...")
        
        result = search(query, document=filename)
        
        # Demo directory-specific search if directories exist
        if stats['directories']:
            print(f"\n🗂️ Directory-specific search:")
            first_dir = stats['directories'][0]
            print(f"Searching only in directory: {first_dir}")
            
            query = "What files are in this directory?"
            print(f"Query: '{query}'")
            
            self.wait_for_user("Press Enter to run directory-specific search...")
            
            result = search(query, directory=first_dir)
        
        # Demo multiple document search if we have multiple files
        if len(stats['files']) >= 2:
            print(f"\n📚 Multiple document search:")
            file1 = os.path.basename(stats['files'][0])
            file2 = os.path.basename(stats['files'][1])
            print(f"Searching in: {file1} and {file2}")
            
            query = "Compare the content of these documents"
            print(f"Query: '{query}'")
            
            self.wait_for_user("Press Enter to run multiple document search...")
            
            result = search_multiple_docs(query, file1, file2)
        
        self.wait_for_user()
    
    def demo_advanced_features(self) -> None:
        """Demonstrate advanced features."""
        self.print_header("STEP 8: ADVANCED FEATURES")
        
        pipeline = RAGPipeline()
        
        print("🧪 Testing document filter building:")
        stats = pipeline.get_database_stats()
        
        if stats['files']:
            filename = os.path.basename(stats['files'][0])
            result = pipeline.test_filter(document=filename)
            print(f"Filter test result: {result['filter']}")
        
        print(f"\n📈 Database update demonstration:")
        print("The system automatically detects:")
        print("- New files added to your document directory")
        print("- Modified files (based on file hash)")
        print("- Deleted files (removes from database)")
        print("- Unchanged files (skips processing)")
        
        print(f"\n🎛️ Customization options:")
        print("- Adjust chunk sizes for different document types")
        print("- Switch embedding models for different languages")
        print("- Enable OCR for scanned documents")
        print("- Configure table extraction settings")
        print("- Set up custom file filtering rules")
        
        self.wait_for_user()
    
    def demo_interactive_search(self) -> None:
        """Interactive search session."""
        self.print_header("STEP 9: INTERACTIVE SEARCH SESSION")
        
        print("Now you can try your own searches!")
        print("Enter your questions, and I'll search through your documents.")
        print("Type 'quit' or 'exit' to finish the demo.")
        print("\nAvailable commands:")
        print("- Just type your question for basic search")
        print("- 'show docs' to see available documents")
        print("- 'stats' to see database statistics")
        
        while True:
            try:
                query = input("\n🔍 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'show docs':
                    show_documents()
                elif query.lower() == 'stats':
                    pipeline = RAGPipeline()
                    stats = pipeline.get_database_stats()
                    print(f"Database has {stats['total_chunks']} chunks from {len(stats['files'])} files")
                elif query:
                    print(f"\n🔍 Searching for: '{query}'")
                    start_time = time.time()
                    result = search(query)
                    end_time = time.time()
                    print(f"⏱️ Search time: {end_time - start_time:.2f}s")
                else:
                    print("Please enter a question or command.")
                    
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thanks for trying the interactive search!")
    
    def demo_conclusion(self) -> None:
        """Demonstrate conclusion and next steps."""
        self.print_header("DEMO COMPLETE!")
        
        print("🎉 Congratulations! You've successfully:")
        print("✅ Set up the complete RAG system")
        print("✅ Built a searchable document database")
        print("✅ Performed various types of searches")
        print("✅ Explored advanced filtering options")
        print("✅ Tried interactive searching")
        
        print(f"\n🚀 Next Steps:")
        print("1. Add your own documents to the DATA_PATH directory")
        print("2. Run build_database() to process them")
        print("3. Customize the configuration for your needs")
        print("4. Integrate the search functionality into your applications")
        
        print(f"\n📚 Quick Reference:")
        print("from rag_main_setup import search, build_database, show_documents")
        print('search("your question")')
        print('search("your question", document="specific.pdf")')
        print('build_database()  # Update with new documents')
        print('show_documents()  # See what\'s available')
        
        print(f"\n🔧 For customization:")
        print("- Edit RAGPipelineConfig class for different settings")
        print("- Check the README.md for detailed documentation")
        print("- Use RAGPipeline() class for advanced features")
        
        print(f"\n💡 Remember:")
        print("- The system automatically handles file updates")
        print("- Use specific queries for better results")
        print("- Filter by document or directory for focused searches")
        print("- Check validate_setup() if you encounter issues")
    
    def run_full_demo(self) -> None:
        """Run the complete demo."""
        self.print_header("🚀 RAG PIPELINE DEMO", "=")
        
        print("Welcome to the Professional RAG Pipeline Demo!")
        print("This demo will walk you through all the features step by step.")
        print("\nThe demo includes:")
        print("1. System setup and configuration")
        print("2. Document database building")
        print("3. Various search demonstrations")
        print("4. Interactive search session")
        
        self.wait_for_user("Ready to start? Press Enter...")
        
        try:
            # Run all demo steps
            if not self.demo_system_setup():
                return
            
            self.demo_configuration()
            
            if not self.demo_validation():
                return
            
            self.demo_database_building()
            self.demo_database_info()
            self.demo_basic_search()
            self.demo_filtered_search()
            self.demo_advanced_features()
            self.demo_interactive_search()
            self.demo_conclusion()
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Demo interrupted by user.")
        except Exception as e:
            print(f"\n\n❌ Demo error: {e}")
            print("Check the error message and try again.")
        
        print("\n👋 Demo finished. Thank you for trying the RAG Pipeline!")


def quick_demo():
    """Run a quick demo without interactive parts."""
    print("🚀 Quick RAG Pipeline Demo")
    print("=" * 40)
    
    # Setup and build
    print("1. Setting up system...")
    setup_rag_system()
    
    print("2. Building database...")
    build_database()
    
    print("3. Running sample searches...")
    queries = [
        "What is this document about?",
        "What are the requirements?",
        "How do I get started?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        search(query, k=3)  # Limit to 3 results for demo
    
    print("\n✅ Quick demo complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run quick demo
        quick_demo()
    else:
        # Run full interactive demo
        demo = RAGDemo()
        demo.run_full_demo()