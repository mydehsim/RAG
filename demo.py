#!/usr/bin/env python3
"""
RAG Pipeline Demo Script v2.1.0
===============================

Interactive demonstration of the Professional RAG Pipeline with centralized configuration.
This script showcases all features including the new centralized configuration system.

Usage:
    python demo.py              # Full interactive demo
    python demo.py --quick      # Quick demo without interaction
    python demo.py --config     # Configuration demo only

Author: Mustafa Said Oğuztürk
Date: 2025-08-01  
Version: 2.1.0 - Centralized Configuration Demo
"""

import os
import time
import sys
from typing import List, Dict, Any

# Import our RAG pipeline components with centralized config
from config import (
    RAGPipeline, 
    RAGPipelineConfig,
    setup_rag_system,
    build_database,
    search,
    search_multiple_docs,
    show_documents,
    show_config,
    validate_setup
)


class RAGDemo:
    """Demo class showcasing RAG pipeline capabilities with centralized configuration."""
    
    def __init__(self):
        """Initialize demo with configuration."""
        self.pipeline = None
        self.config = RAGPipelineConfig()
        self.demo_queries = [
            "Akrep radar projesinden kapsamlıca bahseder misin?",
            "What is SRS?", 
            "Akrep radar projesinin Test Konfigürasyon Şemalarını verir misin?",
            "What is the project scope?"
        ]
    
    def print_header(self, title: str, char: str = "=") -> None:
        """Print a formatted header."""
        print(f"\n{char * 70}")
        print(f"{title.center(70)}")
        print(f"{char * 70}")
    
    def print_step(self, step: str, description: str) -> None:
        """Print a step with description."""
        print(f"\n🔹 {step}")
        print(f"   {description}")
    
    def wait_for_user(self, message: str = "Press Enter to continue...") -> None:
        """Wait for user input."""
        input(f"\n💡 {message}")
    
    def demo_centralized_config(self) -> None:
        """🎯 NEW - Demonstrate centralized configuration system."""
        self.print_header("🎯 NEW FEATURE: CENTRALIZED CONFIGURATION")
        
        print("🚀 Welcome to RAG Pipeline v2.1.0!")
        print("✨ Major improvement: All configuration is now centralized!")
        print()
        print("🔧 Before v2.1.0:")
        print("   ❌ Configuration scattered across multiple files")
        print("   ❌ Duplicate settings everywhere")
        print("   ❌ Hard to manage and maintain")
        print("   ❌ LangChain deprecation warnings")
        print()
        print("✅ Now in v2.1.0:")
        print("   ✅ Single source of truth in config.py")
        print("   ✅ Change any setting in one place")
        print("   ✅ No more duplicate configurations")
        print("   ✅ Fixed all deprecation warnings")
        print("   ✅ Easy customization and maintenance")
        
        self.wait_for_user("Press Enter to see your current configuration...")
        
        # Show current configuration
        print("\n🔧 Your Current Configuration:")
        show_config()
        
        print("\n💡 To customize your setup:")
        print("1. Open config.py in your editor")
        print("2. Edit the RAGPipelineConfig class")  
        print("3. Save and restart - changes apply everywhere!")
        print()
        print("📝 Key settings you can customize:")
        print(f"   📁 DATA_PATH: {self.config.DATA_PATH}")
        print(f"   🤖 Embedding Model: {self.config.CURRENT_EMBEDDING_MODEL}")
        print(f"   💬 LLM Model: {self.config.CURRENT_LLM['model']}")
        print(f"   📄 Chunk Size: {self.config.CHUNK_SIZE}")
        print(f"   🔄 Chunk Overlap: {self.config.CHUNK_OVERLAP}")
        print(f"   🔍 Default Results: {self.config.DEFAULT_K}")
        
        self.wait_for_user()
    
    def demo_system_setup(self) -> bool:
        """Demonstrate system setup."""
        self.print_header("STEP 1: SYSTEM SETUP")
        
        print("Setting up the RAG system with Ollama and required models...")
        print("This includes:")
        print("- Installing Ollama (if not present)")
        print("- Starting Ollama server")
        print("- Downloading required LLM model")
        print(f"- Current LLM model: {self.config.CURRENT_LLM['model']}")
        
        self.wait_for_user("Ready to setup system? This may take a few minutes...")
        
        success = setup_rag_system()
        
        if success:
            print("✅ System setup completed successfully!")
            print("🔧 Ollama server is running")
            print(f"🤖 LLM model '{self.config.CURRENT_LLM['model']}' is ready")
        else:
            print("❌ System setup failed. Check the error messages above.")
            return False
        
        self.wait_for_user()
        return True
    
    def demo_validation(self) -> bool:
        """Demonstrate system validation."""
        self.print_header("STEP 2: SYSTEM VALIDATION")
        
        print("Validating the system setup using centralized configuration...")
        print(f"📁 Checking document path: {self.config.DATA_PATH}")
        print(f"💾 Checking database path: {self.config.CHROMA_PATH}")
        print(f"📋 Supported extensions: {len(self.config.SUPPORTED_EXTENSIONS)} types")
        
        success = validate_setup()
        
        if success:
            print("✅ Validation passed! System is ready to use.")
        else:
            print("❌ Validation failed. Please check the issues above.")
            print("\n🔧 Common fixes:")
            print("- Edit DATA_PATH in config.py to point to your documents")
            print("- Ensure your document directory contains supported files") 
            print("- Check file permissions and disk space")
            return False
        
        self.wait_for_user()
        return True
    
    def demo_database_building(self) -> None:
        """Demonstrate database building process."""
        self.print_header("STEP 3: BUILDING DOCUMENT DATABASE")
        
        print("Building the document database with your configuration...")
        print("📋 Process overview:")
        print("- Scan your document directory for supported files")
        print("- Process each document (extract text, tables, etc.)")
        print("- Split documents into searchable chunks")
        print("- Generate embeddings for semantic search")
        print("- Store everything in ChromaDB vector database")
        print()
        print("🔧 Your processing settings:")
        print(f"   📄 Chunk size: {self.config.CHUNK_SIZE} characters")
        print(f"   🔄 Chunk overlap: {self.config.CHUNK_OVERLAP} characters")
        print(f"   🧠 Embedding model: {self.config.CURRENT_EMBEDDING_MODEL}")
        print(f"   🎛️ ToC filtering: {'Enabled' if self.config.ENABLE_TOC_FILTERING else 'Disabled'}")
        print(f"   📊 OCR processing: {'Enabled' if self.config.ENABLE_OCR else 'Disabled'}")
        print(f"   📋 Table extraction: {'Enabled' if self.config.ENABLE_TABLE_EXTRACTION else 'Disabled'}")
        
        self.wait_for_user("Ready to build database? This may take a few minutes...")
        
        start_time = time.time()
        build_database()
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"\n⏱️ Database building completed in {processing_time:.2f} seconds")
        
        self.wait_for_user()
    
    def demo_database_info(self) -> None:
        """Demonstrate database information display."""
        self.print_header("STEP 4: DATABASE OVERVIEW")
        
        print("Here's what's in your database:")
        show_documents()
        
        # Get detailed stats
        pipeline = RAGPipeline()
        stats = pipeline.get_database_stats()
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total files: {len(stats['files'])}")
        print(f"   Total directories: {len(stats['directories'])}")
        
        if stats['files']:
            avg_chunks_per_file = stats['total_chunks'] / len(stats['files'])  
            print(f"   Average chunks per file: {avg_chunks_per_file:.1f}")
            
            # Show some example file paths
            print(f"\n📁 Sample files:")
            for i, file_path in enumerate(stats['files'][:5], 1):
                filename = os.path.basename(file_path)
                print(f"   {i}. {filename}")
            if len(stats['files']) > 5:
                print(f"   ... and {len(stats['files']) - 5} more files")
        
        self.wait_for_user()
    
    def demo_basic_search(self) -> None:
        """Demonstrate basic search functionality."""
        self.print_header("STEP 5: BASIC SEARCH DEMO")
        
        print("Now let's try some searches using your configuration!")
        print(f"🔍 Search settings:")
        print(f"   Default results: {self.config.DEFAULT_K}")
        print(f"   ToC filtering: {'Enabled' if self.config.ENABLE_TOC_FILTERING else 'Disabled'}")
        print(f"   LLM model: {self.config.CURRENT_LLM['model']}")
        print(f"   Temperature: {self.config.CURRENT_LLM['temp']}")
        
        for i, query in enumerate(self.demo_queries, 1):
            print(f"\n{'='*50}")
            print(f"🔍 Search Example {i}/{len(self.demo_queries)}")
            print(f"{'='*50}")
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
        self.print_header("STEP 6: FILTERED SEARCH DEMO")
        
        # Get available files for demo
        pipeline = RAGPipeline()
        stats = pipeline.get_database_stats()
        
        if not stats['files']:
            print("No files available for filtered search demo.")
            return
        
        print("🎯 Demonstrating advanced filtering capabilities...")
        
        # Demo document-specific search
        print("\n🔍 Document-specific search:")
        first_file = stats['files'][0]
        filename = os.path.basename(first_file)
        print(f"   Searching only in: {filename}")
        
        query = "Akrep radar projesinden kapsamlıca bahseder misin?"
        print(f"   Query: '{query}'")
        
        self.wait_for_user("Press Enter to run document-specific search...")
        
        result = search(query, document=filename)
        
        # Demo directory-specific search if directories exist
        if stats['directories']:
            print(f"\n🗂️ Directory-specific search:")
            first_dir = stats['directories'][0]
            print(f"   Searching only in directory: {first_dir}")
            
            query = "What content is available in this directory?"
            print(f"   Query: '{query}'")
            
            self.wait_for_user("Press Enter to run directory-specific search...")
            
            result = search(query, directory=first_dir)
        
        # Demo multiple document search if we have multiple files
        if len(stats['files']) >= 2:
            print(f"\n📚 Multiple document search:")
            file1 = os.path.basename(stats['files'][0])
            file2 = os.path.basename(stats['files'][1])
            print(f"   Searching in: {file1} and {file2}")
            
            query = "Compare the content of these documents"
            print(f"   Query: '{query}'")
            
            self.wait_for_user("Press Enter to run multiple document search...")
            
            result = search_multiple_docs(query, file1, file2)
        
        self.wait_for_user()
    
    def demo_configuration_features(self) -> None:
        """🎯 NEW - Demonstrate configuration features."""
        self.print_header("STEP 7: CONFIGURATION FEATURES DEMO")
        
        pipeline = RAGPipeline()
        
        print("🔧 Testing configuration-driven features...")
        
        # Test filter building
        print("\n🧪 Document filter testing:")
        stats = pipeline.get_database_stats()
        
        if stats['files']:
            filename = os.path.basename(stats['files'][0])
            print(f"   Testing filter for: {filename}")
            result = pipeline.test_filter(document=filename)
            print(f"   Filter result: {result['filter']}")
            print(f"   Total files in database: {result['total_files']}")
        
        # Show feature flags
        print(f"\n🎛️ Current feature flags:")
        print(f"   ToC Filtering: {'✅ Enabled' if self.config.ENABLE_TOC_FILTERING else '❌ Disabled'}")
        print(f"   OCR Processing: {'✅ Enabled' if self.config.ENABLE_OCR else '❌ Disabled'}")
        print(f"   Table Extraction: {'✅ Enabled' if self.config.ENABLE_TABLE_EXTRACTION else '❌ Disabled'}")
        print(f"   Comprehensive Logging: {'✅ Enabled' if self.config.ENABLE_COMPREHENSIVE_LOGGING else '❌ Disabled'}")
        
        # Show model information
        print(f"\n🤖 Model configuration:")
        print(f"   Embedding device: {self.config.EMBEDDING_DEVICE}")
        print(f"   Available embedding models: {len(self.config.EMBEDDING_MODELS)}")
        print(f"   Available LLM models: {len(self.config.LLM_MODELS)}")
        
        # Database update demonstration
        print(f"\n📈 Smart update capabilities:")
        print("   The system automatically detects:")
        print("   - New files added to your document directory")
        print("   - Modified files (based on file hash)")
        print("   - Deleted files (removes from database)")
        print("   - Unchanged files (skips processing for efficiency)")
        
        print(f"\n🎯 Customization examples:")
        print("   Edit config.py to:")
        print("   - Switch to Turkish-optimized embedding model")
        print("   - Enable OCR for scanned documents")
        print("   - Adjust chunk sizes for different document types")
        print("   - Change the number of search results returned")
        print("   - Modify the LLM prompt template")
        
        self.wait_for_user()
    
    def demo_interactive_search(self) -> None:
        """Interactive search session."""
        self.print_header("STEP 8: INTERACTIVE SEARCH SESSION")
        
        print("Now you can try your own searches!")
        print("Enter your questions, and I'll search through your documents.")
        print("Type 'quit' or 'exit' to finish the demo.")
        print("\n📋 Available commands:")
        print("   • Just type your question for basic search")
        print("   • 'show docs' to see available documents")
        print("   • 'show config' to see current configuration")
        print("   • 'stats' to see database statistics")
        print("   • 'help' to see this help again")
        
        while True:
            try:
                query = input("\n🔍 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'show docs':
                    show_documents()
                elif query.lower() == 'show config':
                    show_config()
                elif query.lower() == 'stats':
                    pipeline = RAGPipeline()
                    stats = pipeline.get_database_stats()
                    print(f"📊 Database: {stats['total_chunks']} chunks from {len(stats['files'])} files")
                elif query.lower() == 'help':
                    print("📋 Available commands:")
                    print("   • Just type your question for basic search")
                    print("   • 'show docs' to see available documents")
                    print("   • 'show config' to see current configuration")
                    print("   • 'stats' to see database statistics")
                    print("   • 'quit' or 'exit' to finish")
                elif query:
                    print(f"\n🔍 Searching for: '{query}'")
                    print(f"   Using {self.config.DEFAULT_K} results, {self.config.CURRENT_LLM['model']} model")
                    start_time = time.time()
                    result = search(query)
                    end_time = time.time()
                    print(f"⏱️ Search time: {end_time - start_time:.2f}s")
                else:
                    print("Please enter a question or command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thanks for trying the interactive search!")
    
    def demo_conclusion(self) -> None:
        """Demonstrate conclusion and next steps."""
        self.print_header("🎉 DEMO COMPLETE!")
        
        print("🏆 Congratulations! You've successfully explored:")
        print("✅ Centralized configuration system (NEW in v2.1.0!)")
        print("✅ Complete RAG system setup with Ollama")
        print("✅ Document database building and management")
        print("✅ Various types of intelligent searches")
        print("✅ Advanced filtering and customization options")
        print("✅ Interactive search capabilities")
        
        print(f"\n🚀 Next Steps:")
        print("1. 📁 Add your own documents to the DATA_PATH directory")
        print("2. 🔧 Customize config.py for your specific needs")
        print("3. 🔄 Run build_database() to process your documents")
        print("4. 🔗 Integrate the search functionality into your applications")
        
        print(f"\n📚 Quick Reference (Centralized Config):")
        print("```python")
        print("from config import search, build_database, show_documents, show_config")
        print("")
        print("# Basic usage")
        print('search("your question")')
        print('search("specific query", document="file.pdf")')  
        print('build_database()  # Smart update with new documents')
        print('show_documents()  # See what\'s available')
        print("")
        print("# Configuration")
        print('show_config()     # View current settings')
        print('validate_setup()  # Check system health')
        print("```")
        
        print(f"\n🔧 For customization:")
        print("- 📝 Edit RAGPipelineConfig class in config.py")
        print("- 📖 Check the updated README.md for detailed documentation")
        print("- 🔍 Use RAGPipeline() class for advanced features")
        print("- 🧪 Test changes with validate_setup()")
        
        print(f"\n💡 Key improvements in v2.1.0:")
        print("- 🎯 Single source of truth for all configuration")
        print("- ✅ Fixed all LangChain deprecation warnings")
        print("- 🔧 Easier maintenance and customization")
        print("- 📦 Updated to latest LangChain packages")
        
        print(f"\n🎖️ Remember:")
        print("- The system automatically handles file updates")
        print("- Use specific queries for better results")
        print("- Filter by document or directory for focused searches")
        print("- All settings are managed in config.py")
        print("- Run validate_setup() if you encounter issues")
    
    def run_full_demo(self) -> None:
        """Run the complete demo with centralized configuration focus."""
        self.print_header("🚀 RAG PIPELINE v2.1.0 DEMO", "=")
        
        print("Welcome to the Professional RAG Pipeline Demo!")
        print("🎯 NEW: Now featuring centralized configuration management!")
        print()
        print("This demo will walk you through:")
        print("1. 🎯 NEW: Centralized configuration system") 
        print("2. System setup and validation")
        print("3. Document database building")
        print("4. Various search demonstrations")
        print("5. Configuration-driven features")
        print("6. Interactive search session")
        
        self.wait_for_user("Ready to start? Press Enter...")
        
        try:
            # Run all demo steps with new config focus
            self.demo_centralized_config()  # NEW step
            
            if not self.demo_system_setup():
                return
            
            if not self.demo_validation():
                return
            
            self.demo_database_building()
            self.demo_database_info()
            self.demo_basic_search()
            self.demo_filtered_search()
            self.demo_configuration_features()  # NEW step
            self.demo_interactive_search()
            self.demo_conclusion()
        
        except KeyboardInterrupt:
            print("\n\n⏹️ Demo interrupted by user.")
        except Exception as e:
            print(f"\n\n❌ Demo error: {e}")
            print("Check the error message and try again.")
        
        print("\n👋 Demo finished. Thank you for trying RAG Pipeline v2.1.0!")


def quick_demo():
    """Run a quick demo without interactive parts."""
    print("🚀 Quick RAG Pipeline v2.1.0 Demo")
    print("=" * 50)
    
    # Show configuration first
    print("🔧 Current Configuration:")
    show_config()
    
    # Setup and build  
    print("\n1. Setting up system...")
    if setup_rag_system():
        print("✅ System ready!")
    
    print("\n2. Validating setup...")
    if validate_setup():
        print("✅ Validation passed!")
    
    print("\n3. Building database...")
    build_database()
    
    print("\n4. Running sample searches...")
    queries = [
        "Akrep radar projesinden kapsamlıca bahseder misin?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        search(query, k=3)  # Limit to 3 results for demo
    
    print("\n✅ Quick demo complete!")
    print("💡 Run 'python demo.py' for the full interactive experience!")


def config_demo():
    """Demo focusing only on configuration features."""
    print("🔧 RAG Pipeline v2.1.0 - Configuration Demo")
    print("=" * 60)
    
    config = RAGPipelineConfig()
    
    print("🎯 NEW: Centralized Configuration System")
    print("All settings are now managed in config.py!")
    print()
    
    # Show current config
    print("📋 Your Current Configuration:")
    show_config()
    
    print("\n🔄 Available Embedding Models:")
    for name, model in config.EMBEDDING_MODELS.items():
        current = "👈 CURRENT" if model == config.CURRENT_EMBEDDING_MODEL else ""
        print(f"   {name}: {model} {current}")
    
    print("\n🤖 Available LLM Models:")
    for name, model_info in config.LLM_MODELS.items():
        current = "👈 CURRENT" if model_info == config.CURRENT_LLM else ""
        print(f"   {name}: {model_info['model']} (temp: {model_info['temp']}) {current}")
    
    print("\n🎛️ Feature Flags:")
    features = [
        ("ToC Filtering", config.ENABLE_TOC_FILTERING),
        ("OCR Processing", config.ENABLE_OCR), 
        ("Table Extraction", config.ENABLE_TABLE_EXTRACTION),
        ("Comprehensive Logging", config.ENABLE_COMPREHENSIVE_LOGGING)
    ]
    
    for feature, enabled in features:
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"   {feature}: {status}")
    
    print("\n📝 To customize:")
    print("1. Open config.py in your editor")
    print("2. Edit RAGPipelineConfig class")
    print("3. Save and restart Python")
    print("4. Changes apply everywhere automatically!")
    
    print("\n✅ Configuration demo complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Run quick demo
            quick_demo()
        elif sys.argv[1] == "--config":
            # Run configuration demo only
            config_demo()
        else:
            print("Usage:")
            print("  python demo.py           # Full interactive demo")
            print("  python demo.py --quick   # Quick demo without interaction")
            print("  python demo.py --config  # Configuration demo only")
    else:
        # Run full interactive demo
        demo = RAGDemo()
        demo.run_full_demo()