"""
Professional RAG Pipeline - Document Processing Module
==============================================================

Retrieval-Augmented Generation (RAG) uygulamaları için kapsamlı bir
belge işleme ve vektör veritabanı yönetim sistemi.

Author: Mustafa Said Oğuztürk
Date: 2025-08-01
Version: 2.1.0
"""

import os
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Core imports
import pandas as pd
import numpy as np

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
# from langchain_chroma import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader

# Import centralized configuration
from config import RAGPipelineConfig

# Docling imports for advanced document processing
try:
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter as DoclingDocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError as e:
    print(f"Uyarı: Docling yüklemesi başarısız oldu {e}")
    print("Temel dosya yükleme sistemine geri dönülüyor...")
    DOCLING_AVAILABLE = False

# Configuration
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocumentConverter:
    """Handles document conversion and loading with optimized settings."""
    
    def __init__(self, config: RAGPipelineConfig = None):
        """Initialize document converter with config."""
        self.config = config or RAGPipelineConfig()
        
        if DOCLING_AVAILABLE:
            try:
                self.pipeline_options = self._setup_pipeline_options()
                self.doc_converter = self._create_converter_v2()
            except Exception as e:
                print(f"Uyarı: Docling çalıştırması başarısız oldu: {e}")
                print("Temel dosya yükleme sistemine geri dönülüyor...")
                self.doc_converter = self._create_converter_basic()
        else:
            print("Temel dosya yükleme sistemi kullanılıyor (docling not available)")
            self.doc_converter = None
    
    def _setup_pipeline_options(self) -> 'PdfPipelineOptions':
        """Setup optimized PDF pipeline options using config."""
        try:
            pipeline_options = PdfPipelineOptions(do_table_structure=self.config.ENABLE_TABLE_EXTRACTION)
            pipeline_options.do_ocr = self.config.ENABLE_OCR
            if self.config.ENABLE_TABLE_EXTRACTION:
                pipeline_options.table_structure_options.do_cell_matching = True
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            return pipeline_options
        except Exception as e:
            print(f"UYARI: Kurulum ayarları tamamlanamadı: {e}")
            return None
    
    def _create_converter_v2(self) -> 'DoclingDocumentConverter':
        """Create document converter with format options (newer API)."""
        if self.pipeline_options:
            return DoclingDocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
                }
            )
        else:
            return DoclingDocumentConverter()
    
    def _create_converter_basic(self) -> 'DoclingDocumentConverter':
        """Create basic document converter (fallback)."""
        return DoclingDocumentConverter()
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load documents from a single file with fallback methods."""
        if not DOCLING_AVAILABLE or self.doc_converter is None:
            return self._load_document_basic(file_path)
        
        try:
            # Try with docling
            loader = DoclingLoader(file_path=str(file_path), converter=self.doc_converter)
            return loader.load()
        except Exception as e:
            print(f"Bu dosya {file_path} docling ile yüklenemedi!: {e}")
            return self._load_document_basic(file_path)
    
    def _load_document_basic(self, file_path: str) -> List[Document]:
        """Basic document loading fallback."""
        try:
            # Simple text file reading for basic cases
            if file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": file_path})]
            else:
                print(f"UYARI: Bu dosya {file_path} docling kullanılmadan yüklenemedi!")
                return []
        except Exception as e:
            print(f"Temel dosya kurulum methoduna dönülüyor: {file_path}: {e}")
            return []


class TableOfContentsFilter:
    """Handles detection and filtering of Table of Contents content."""
    
    def __init__(self, config: RAGPipelineConfig = None):
        """Initialize with config."""
        self.config = config or RAGPipelineConfig()
    
    def is_toc_content(self, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Comprehensive ToC detection for database filtering using config patterns.
        
        Args:
            content: Text content to analyze
            metadata: Optional metadata for additional context
            
        Returns:
            bool: True if content appears to be Table of Contents
        """
        content_clean = content.strip()
        content_lower = content_clean.lower()
        
        # Check for explicit ToC headers using config patterns
        for pattern in self.config.TOC_PATTERNS:
            if pattern.lower() in content_lower:
                return True
        
        # Check structural patterns
        lines = content_clean.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) == 0:
            return False
        
        # Dot pattern detection (classic ToC)
        dot_lines = sum(1 for line in non_empty_lines if '........' in line)
        if dot_lines >= 2:
            return True
        
        # Page number patterns
        page_number_lines = sum(
            1 for line in non_empty_lines 
            if (line.endswith(tuple(str(i) for i in range(10))) and 
                ('.' * 8) in line)
        )
        if page_number_lines >= 2:
            return True
        
        return False
    
    def filter_toc_chunks(self, chunks: List[Document]) -> List[Document]:
        """Filter out Table of Contents chunks from document list."""
        return [
            chunk for chunk in chunks 
            if not self.is_toc_content(chunk.page_content, chunk.metadata)
        ]


class PageNumberExtractor:
    """Handles page number extraction from different document types."""
    
    @staticmethod
    def extract_page_number(chunk: Document, chunk_index: int = None, total_chunks: int = None) -> Optional[int]:
        """
        Smart page number extraction for different file formats.
        
        Args:
            chunk: Document chunk
            chunk_index: Index of chunk within document
            total_chunks: Total chunks in document
            
        Returns:
            Optional[int]: Page number if extractable
        """
        source = chunk.metadata.get("source", "")
        file_ext = source.split('.')[-1].lower() if '.' in source else ""
        
        # Try direct extraction from metadata (works for PDFs)
        try:
            dl_meta = chunk.metadata.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            if doc_items:
                prov = doc_items[0].get("prov", [])
                if prov:
                    page = prov[0].get("page_no")
                    if page is not None:
                        return page
        except (KeyError, IndexError, TypeError):
            pass
        
        # Estimate for DOCX files
        if file_ext == 'docx' and chunk_index is not None and total_chunks is not None:
            chunks_per_page = 4  # Rough estimate
            return (chunk_index // chunks_per_page) + 1
        
        return None


class ChunkProcessor:
    """Handles text splitting and chunk ID assignment."""
    
    def __init__(self, config: RAGPipelineConfig = None):
        """Initialize chunk processor with config parameters."""
        self.config = config or RAGPipelineConfig()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.toc_filter = TableOfContentsFilter(self.config)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []
        return self.text_splitter.split_documents(documents)
    
    def assign_chunk_ids(self, chunks: List[Document], filter_toc: bool = None) -> List[Document]:
        """
        Assign unique IDs to chunks and optionally filter ToC content.
        
        Args:
            chunks: List of document chunks
            filter_toc: Whether to filter Table of Contents (uses config if None)
            
        Returns:
            List[Document]: Processed chunks with IDs
        """
        if not chunks:
            return []
        
        # Use config setting if not specified
        if filter_toc is None:
            filter_toc = self.config.ENABLE_TOC_FILTERING
            
        if filter_toc:
            chunks = self.toc_filter.filter_toc_chunks(chunks)
        
        # Group chunks by source file
        chunks_by_file = self._group_chunks_by_file(chunks)
        
        # Process chunks and assign IDs
        processed_chunks = []
        for source, file_chunks in chunks_by_file.items():
            processed_chunks.extend(self._process_file_chunks(source, file_chunks))
        
        return processed_chunks
    
    def _group_chunks_by_file(self, chunks: List[Document]) -> Dict[str, List[Tuple[int, Document]]]:
        """Group chunks by source file."""
        chunks_by_file = {}
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            if source not in chunks_by_file:
                chunks_by_file[source] = []
            chunks_by_file[source].append((i, chunk))
        return chunks_by_file
    
    def _process_file_chunks(self, source: str, file_chunks: List[Tuple[int, Document]]) -> List[Document]:
        """Process chunks for a single file and assign IDs."""
        file_ext = source.split('.')[-1].lower() if '.' in source else "unknown"
        last_page_id = None
        current_chunk_index = 0
        
        processed_chunks = []
        for chunk_index_in_file, (global_index, chunk) in enumerate(file_chunks):
            # Extract page number
            page = PageNumberExtractor.extract_page_number(
                chunk, chunk_index_in_file, len(file_chunks)
            )
            page = page or 1  # Default to page 1 if extraction fails
            
            # Create page ID
            current_page_id = f"{source}:pg.{page}"
            
            # Calculate chunk index within page
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            
            # Assign chunk ID
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            chunk.metadata["id"] = chunk_id
            last_page_id = current_page_id
            
            processed_chunks.append(chunk)
        
        return processed_chunks


class FileManager:
    """Handles file system operations and metadata tracking."""
    
    def __init__(self, config: RAGPipelineConfig = None):
        """Initialize with config."""
        self.config = config or RAGPipelineConfig()
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> Optional[str]:
        """Calculate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_modification_time(file_path: str) -> Optional[float]:
        """Get file's last modification time."""
        try:
            return os.path.getmtime(file_path)
        except Exception as e:
            print(f"Error getting modification time for {file_path}: {e}")
            return None
    
    def get_supported_files(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """Get list of supported document files with metadata using config extensions."""
        files_info = {}
        
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist: {directory}")
            return files_info
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                _, ext = os.path.splitext(filename.lower())
                
                if ext in self.config.SUPPORTED_EXTENSIONS:
                    file_hash = self.calculate_file_hash(file_path)
                    mod_time = self.get_file_modification_time(file_path)
                    files_info[file_path] = {
                        "hash": file_hash,
                        "mod_time": mod_time
                    }
        
        return files_info


class DatabaseManager:
    """Manages ChromaDB operations and file tracking."""
    
    def __init__(self, chroma_path: str, embeddings, config: RAGPipelineConfig = None):
        """Initialize database manager with config."""
        self.config = config or RAGPipelineConfig()
        self.chroma_path = chroma_path
        self.embeddings = embeddings
        self.db = None
    
    def get_database(self):
        """Get or create database connection."""
        if self.db is None:
            self.db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
            )
        return self.db
    
    def get_files_in_database(self) -> Dict[str, Dict[str, Any]]:
        """Get all files currently stored in the database with metadata."""
        try:
            db = self.get_database()
            all_items = db.get(include=["metadatas"])
            files_info = {}
            
            for metadata in all_items["metadatas"]:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    if source not in files_info:
                        files_info[source] = {
                            "hash": metadata.get("file_hash"),
                            "mod_time": metadata.get("file_mod_time"),
                            "chunk_count": 0
                        }
                    files_info[source]["chunk_count"] += 1
            
            return files_info
        except Exception as e:
            print(f"Error getting files from database: {e}")
            return {}
    
    def delete_file_from_database(self, file_path: str) -> int:
        """Delete all chunks belonging to a specific file from database."""
        try:
            db = self.get_database()
            all_items = db.get(include=["metadatas"])
            chunk_ids_to_delete = []
            
            for i, metadata in enumerate(all_items["metadatas"]):
                if metadata and metadata.get("source") == file_path:
                    chunk_ids_to_delete.append(all_items["ids"][i])
            
            if chunk_ids_to_delete:
                db.delete(ids=chunk_ids_to_delete)
                print(f"    DELETED: {len(chunk_ids_to_delete)} chunks from {file_path}")
                return len(chunk_ids_to_delete)
            return 0
        except Exception as e:
            print(f"Error deleting file {file_path} from database: {e}")
            return 0
    
    def add_documents_to_database(self, chunks: List[Document]) -> bool:
        """Add document chunks to database."""
        if not chunks:
            return False
        
        try:
            db = self.get_database()
            filtered_chunks = self._filter_complex_metadata(chunks)
            db.add_documents(
                filtered_chunks,
                ids=[chunk.metadata["id"] for chunk in filtered_chunks]
            )
            return True
        except Exception as e:
            print(f"Error adding documents to database: {e}")
            return False
    
    def _filter_complex_metadata(self, chunks: List[Document]) -> List[Document]:
        """Filter out complex metadata that might cause issues with Chroma."""
        filtered_chunks = []
        for chunk in chunks:
            new_metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    new_metadata[key] = value
                elif key == "id":
                    new_metadata[key] = str(value)
            
            new_chunk = Document(
                page_content=chunk.page_content,
                metadata=new_metadata
            )
            filtered_chunks.append(new_chunk)
        
        return filtered_chunks


class RAGDocumentProcessor:
    """Main class that orchestrates the document processing pipeline."""
    
    def __init__(self, data_path: str = None, chroma_path: str = None, config: RAGPipelineConfig = None):
        """Initialize the RAG document processor with centralized configuration."""
        self.config = config or RAGPipelineConfig()
        
        # Use config values if parameters not provided
        self.data_path = data_path or self.config.DATA_PATH
        self.chroma_path = chroma_path or self.config.CHROMA_PATH
        
        # Initialize components with config
        self.embeddings = self._initialize_embeddings()
        self.document_converter = DocumentConverter(self.config)
        self.chunk_processor = ChunkProcessor(self.config)
        self.file_manager = FileManager(self.config)
        self.database_manager = DatabaseManager(self.chroma_path, self.embeddings, self.config)
        
        print(f"📁 Veri Yolu: {self.data_path}")
        print(f"💾 Chroma Database yolu: {self.chroma_path}")
        
        if self.config.ENABLE_COMPREHENSIVE_LOGGING:
            print("🔧 Kullanılan konfigürasyonlar:")
            self.config.print_config()
    
    def _initialize_embeddings(self):
        """Initialize embedding function using config."""
        return HuggingFaceEmbeddings(
            model_name=self.config.CURRENT_EMBEDDING_MODEL,
            model_kwargs={'device': self.config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def process_single_file(self, file_path: str, file_info: Dict[str, Any]) -> List[Document]:
        """Process a single file and return processed chunks."""
        print(f"    Processing: {os.path.basename(file_path)}")
        
        # Load documents
        documents = self.document_converter.load_document(file_path)
        if not documents:
            print(f"    HATA: Dosya yüklenemedi")
            return []
        
        # Split into chunks
        chunks = self.chunk_processor.split_documents(documents)
        if not chunks:
            print(f"    HATA: Chunk yaratılamadı")
            return []
        
        # Add file metadata to chunks
        for chunk in chunks:
            chunk.metadata.update({
                "file_hash": file_info["hash"],
                "file_mod_time": file_info["mod_time"]
            })
        
        # Assign chunk IDs and filter ToC (uses config settings)
        processed_chunks = self.chunk_processor.assign_chunk_ids(chunks)
        
        print(f"     {len(processed_chunks)} chunk yaratıldı")
        return processed_chunks
    
    def update_database(self, comprehensive_toc_filter: bool = None) -> Dict[str, int]:
        """
        Update database with smart file change detection.
        
        Args:
            comprehensive_toc_filter: Whether to apply comprehensive ToC filtering (uses config if None)
            
        Returns:
            Dict[str, int]: Statistics about the update operation
        """
        # Use config setting if not specified
        if comprehensive_toc_filter is None:
            comprehensive_toc_filter = self.config.ENABLE_TOC_FILTERING
            
        print(f"🔄 Database şu dizinden güncelleniyor: {self.data_path}")
        print(f"📋 İÇİNDEKİLER tablo filtrelemesi: {comprehensive_toc_filter}")
        
        # Get current state
        current_files = self.file_manager.get_supported_files(self.data_path)
        db_files = self.database_manager.get_files_in_database()
        
        print(f"📁  Dosya sistemindeki dosya sayısı: {len(current_files)}")
        print(f"💾 Database de bulunan dosya sayı: {len(db_files)}")
        
        if not current_files:
            print("⚠️ No supported files found in data directory!")
            return {"error": "No files found"}
        
        # Initialize statistics
        stats = {
            "new_files": 0,
            "deleted_files": 0,
            "modified_files": 0,
            "unchanged_files": 0,
            "total_chunks_added": 0,
            "total_chunks_deleted": 0
        }
        
        # Handle deleted files
        print(f"\n🗑️ Silinen dosyalar kontrol ediliyor...")
        for db_file in db_files.keys():
            if db_file not in current_files:
                print(f"    SİLİNİYOR: {os.path.basename(db_file)} (varolmayan dosya)")
                deleted_count = self.database_manager.delete_file_from_database(db_file)
                stats["deleted_files"] += 1
                stats["total_chunks_deleted"] += deleted_count
        
        # Handle new and modified files
        print(f"\n📁 Dosyalardaki değişiklikler düzenleniyor...")
        for file_path, file_info in current_files.items():
            relative_path = os.path.relpath(file_path, self.data_path)
            file_hash = file_info["hash"]
            
            if file_hash is None:
                print(f"    ATLANDI: {relative_path} (hash değeri hesaplanamadı)")
                continue
            
            process_file = False
            
            if file_path in db_files:
                db_hash = db_files[file_path].get("hash")
                if db_hash != file_hash:
                    print(f"    MODIFIED: {relative_path}")
                    # Delete old version
                    deleted_count = self.database_manager.delete_file_from_database(file_path)
                    stats["total_chunks_deleted"] += deleted_count
                    stats["modified_files"] += 1
                    process_file = True
                else:
                    print(f"    DEĞİŞMEDİ: {relative_path}")
                    stats["unchanged_files"] += 1
            else:
                print(f"    YENİ: {relative_path}")
                stats["new_files"] += 1
                process_file = True
            
            # Process file if needed
            if process_file:
                processed_chunks = self.process_single_file(file_path, file_info)
                if processed_chunks:
                    success = self.database_manager.add_documents_to_database(processed_chunks)
                    if success:
                        stats["total_chunks_added"] += len(processed_chunks)
                        print(f"    EKLENDİ: {len(processed_chunks)} chunk")
                    else:
                        print(f"    HATA: database chunk eklenmesi başarısız oldu")
        
        # Print summary
        self._print_update_summary(stats)
        return stats
    
    def build_initial_database(self, comprehensive_toc_filter: bool = None) -> None:
        """Build database from scratch using config settings."""
        # Use config setting if not specified
        if comprehensive_toc_filter is None:
            comprehensive_toc_filter = self.config.ENABLE_TOC_FILTERING
            
        print(f"🏗️ Database temelinden oluşturuluyor: {self.data_path}")
        print(f"📋 İçindekiler tablo filtresi: {comprehensive_toc_filter}")
        
        files_info = self.file_manager.get_supported_files(self.data_path)
        print(f"📁 {len(files_info)} desteklenen dosya bulundu")
        
        if not files_info:
            print("⚠️ Desteklenen dosya bulunamadı! Lütfen veri yolunu kontrol ediniz.")
            return
        
        total_chunks = 0
        processed_files = 0
        
        for i, (file_path, file_info) in enumerate(files_info.items(), 1):
            relative_path = os.path.relpath(file_path, self.data_path)
            print(f"\n📄 [{i}/{len(files_info)}] Processing: {relative_path}")
            
            if file_info["hash"] is None:
                print(f"    SKIPPED (couldn't calculate hash)")
                continue
            
            processed_chunks = self.process_single_file(file_path, file_info)
            if processed_chunks:
                success = self.database_manager.add_documents_to_database(processed_chunks)
                if success:
                    processed_files += 1
                    total_chunks += len(processed_chunks)
                    print(f"    BAŞARILI: {len(processed_chunks)} chunk eklendi")
                else:
                    print(f"    HATA: Database'e chunk eklenirken bir hata oluştu")
        
        print(f"\n" + "="*60)
        print(f"🎉 İLK OLUŞTURMA TAMAMLANDI:")
        print(f"    İşlenen dosya sayısı: {processed_files}")
        print(f"    Eklenen chunk sayısı: {total_chunks}")
        print(f"="*60)
    
    def _print_update_summary(self, stats: Dict[str, int]) -> None:
        """Print update operation summary."""
        if "error" in stats:
            return
            
        try:
            db = self.database_manager.get_database()
            final_items = db.get(include=[])
            final_count = len(final_items['ids']) if final_items else 0
        except:
            final_count = "Unknown"
        
        print(f"\n" + "="*60)
        print(f"📊 Güncelleme Özeti:")
        print(f"    Yeni dosya sayısı: {stats['new_files']}")
        print(f"    Değişen dosya sayısı: {stats['modified_files']}")
        print(f"    Silinen dosya sayısı: {stats['deleted_files']}")
        print(f"    Değişmeyen dosya sayısı: {stats['unchanged_files']}")
        print(f"    Eklenen toplam chunk sayısı: {stats['total_chunks_added']}")
        print(f"    Silinenen toplam chunk sayısı: {stats['total_chunks_deleted']}")
        print(f"    Son Database Boyutu: {final_count} chunks")
        print(f"="*60)


# Main execution functions using centralized config
def main_update_database(data_path: str = None, chroma_path: str = None, config: RAGPipelineConfig = None):
    """Main function to update existing database using centralized config."""
    if config is None:
        config = RAGPipelineConfig()
    
    processor = RAGDocumentProcessor(data_path, chroma_path, config)
    return processor.update_database()


def main_build_initial_database(data_path: str = None, chroma_path: str = None, config: RAGPipelineConfig = None):
    """Main function to build database from scratch using centralized config."""
    if config is None:
        config = RAGPipelineConfig()
    
    processor = RAGDocumentProcessor(data_path, chroma_path, config)
    processor.build_initial_database()


if __name__ == "__main__":
    # Example usage with centralized configuration
    print("🚀 RAG Dosya işleme sistemi başlıyor...")
    print("🔧 Konfigürasyon dosyası : config.py")
    
    # Show current configuration
    config = RAGPipelineConfig()
    config.print_config()
    
    # For regular updates (recommended for existing databases)
    print("\n📄 DATABASE güncellemesi başlıyor...")
    main_update_database()
    
    # For initial database creation (uncomment if building from scratch)
    # print("\n🏗️ Building initial database...")
    # main_build_initial_database()