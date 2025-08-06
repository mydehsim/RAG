"""
RAG Pipeline - Enhanced Chainlit UI (Simplified Version)
========================================================

Simplified enhanced web interface compatible with all Chainlit versions.
Focuses on stability and source display with hyperlinks.

Author: Mustafa Said Oğuztürk
Date: 2025-08-01
Version: 1.2
"""

import os
import time
import sys
import asyncio
from typing import Dict, List, Any, Optional
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, TextInput

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import RAG pipeline
try:
    from config import (
        RAGPipeline, 
        RAGPipelineConfig,
        search,
        build_database,
        show_documents,
        validate_setup,
        setup_rag_system
    )
    RAG_AVAILABLE = True
    print("✅ RAG Pipeline başarıyla import edildi")
except ImportError as e:
    print(f"❌ RAG Pipeline import edilemedi: {e}")
    RAG_AVAILABLE = False

# Global variables
config = None
pipeline = None

if RAG_AVAILABLE:
    try:
        config = RAGPipelineConfig()
        print("✅ Konfigürasyon başarıyla yüklendi")
    except Exception as e:
        print(f"❌ Konfigürasyon yüklenemedi: {e}")


async def simple_typing_animation(text: str, delay: float = 0.05, author: str = "RAG Assistant"):
    """Simple typing animation that works with all Chainlit versions."""
    words = text.split()
    current_text = ""
    
    for i, word in enumerate(words):
        current_text += word + " "
        
        # Send progress every few words to create animation effect
        if i % 3 == 0 or i == len(words) - 1:
            await cl.Message(content=current_text.strip(), author=author).send()
            if i < len(words) - 1:  # Don't delay on the last word
                await asyncio.sleep(delay)
    
    return current_text.strip()


def format_sources_with_links(sources: List[str]) -> str:
    """Format sources as clickable hyperlinks."""
    if not sources or isinstance(sources, str):
        return ""
    
    sources_html = "\n\n---\n\n## 📚 **Kaynaklar:**\n\n"
    
    for i, source in enumerate(sources, 1):
        # Extract file path from source ID (format: filepath:pg.X:Y)
        if ':' in source:
            file_path = source.split(':')[0]
            page_info = source.split(':')[1] if len(source.split(':')) > 1 else ""
            
            # Create a more readable display name
            file_name = os.path.basename(file_path)
            display_name = f"{file_name} ({page_info})" if page_info else file_name
            
            # Create file:// URL for local files (normalize path separators)
            normalized_path = file_path.replace('\\', '/')
            file_url = f"file:///{normalized_path}"
            
            sources_html += f"{i}. **[{display_name}]({file_url})**\n"
            sources_html += f"   *Chunk ID: `{source}`*\n\n"
        else:
            sources_html += f"{i}. **{source}**\n\n"
    
    return sources_html


@cl.on_chat_start
async def start():
    """Initialize the chat session with settings panel."""
    global pipeline, config
    
    if not RAG_AVAILABLE:
        await cl.Message(
            content="❌ **RAG Pipeline Müsait Değil**\n\nRAG Pipeline'ın başarıyla konfigüre edildiğinden ve kurulduğundan emin olun!",
            author="System"
        ).send()
        return
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline()
        
        # Setup settings panel
        await setup_settings_panel()
        
        # Welcome message
        await cl.Message(
            content="""# 🚀 RAG Pipeline - Enhanced Version

Hoş geldiniz! Belgelerinizde arama yapmanıza ve RAG sisteminizi yönetmenize yardımcı olabilirim.


Başlamaya hazır mısınız?
            """,
            author="RAG Assistant"
        ).send()
        
        # Show initial database status
        await show_database_status()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Initialization Error:** {str(e)}",
            author="System"
        ).send()


async def setup_settings_panel():
    """Setup the comprehensive settings panel."""
    global config
    
    if not config:
        return
    
    # Get available model options
    embedding_options = list(config.EMBEDDING_MODELS.keys())
    llm_options = list(config.LLM_MODELS.keys())
    
    # Find current model indices
    current_embedding_idx = 0
    current_llm_idx = 0
    
    for i, (key, model) in enumerate(config.EMBEDDING_MODELS.items()):
        if model == config.CURRENT_EMBEDDING_MODEL:
            current_embedding_idx = i
            break
    
    for i, (key, model_info) in enumerate(config.LLM_MODELS.items()):
        if model_info == config.CURRENT_LLM:
            current_llm_idx = i
            break
    
    # Create settings panel
    settings = await cl.ChatSettings(
        [
            # Path Configuration
            TextInput(
                id="data_path",
                label="📂 Veri Dizini",
                initial=config.DATA_PATH,
                placeholder="Dokümanlarınızın olduğu dizini girin"
            ),
            TextInput(
                id="chroma_path", 
                label="💾 Veritabanı Dizini",
                initial=config.CHROMA_PATH,
                placeholder="Veri tabanınızın olduğu dizini girin"
            ),
            
            # Model Configuration
            Select(
                id="embedding_model",
                label="🧠 Embedding Model",
                values=embedding_options,
                initial_index=current_embedding_idx
            ),
            Select(
                id="llm_model", 
                label="💬 LLM Model",
                values=llm_options,
                initial_index=current_llm_idx
            ),
            Select(
                id="embedding_device",
                label="⚡ İşleme Cihazı",
                values=["cpu", "cuda"],
                initial_index=0 if config.EMBEDDING_DEVICE == "cpu" else 1
            ),
            
            # Processing Parameters
            Slider(
                id="chunk_size",
                label="📄 Chunk Boyutu (karakter)",
                initial=config.CHUNK_SIZE,
                min=100,
                max=3000,
                step=100
            ),
            Slider(
                id="chunk_overlap",
                label="🔄 Chunk Overlap (karakter)", 
                initial=config.CHUNK_OVERLAP,
                min=50,
                max=800,
                step=50
            ),
            Slider(
                id="default_k",
                label="🔍 Arama Sonucu Sayısı",
                initial=config.DEFAULT_K,
                min=1,
                max=20,
                step=1
            ),
            
            # Feature Settings
            Switch(
                id="enable_toc_filtering",
                label="📋 İÇİNDEKİLER Tablosu Filtresi",
                initial=config.ENABLE_TOC_FILTERING
            ),
            Switch(
                id="enable_ocr",
                label="👁️ OCR İşlemesini Etkinleştir",
                initial=config.ENABLE_OCR
            ),
            Switch(
                id="enable_table_extraction",
                label="📊 Belgelerden Tabloları Çıkar", 
                initial=config.ENABLE_TABLE_EXTRACTION
            ),
            Switch(
                id="enable_comprehensive_logging",
                label="📝 Detaylı Günlük Kaydını Etkinleştir",
                initial=config.ENABLE_COMPREHENSIVE_LOGGING
            ),
            
            # Animation Settings (simplified)
            Switch(
                id="enable_typing_animation",
                label="⌨️ Yazma Animasyonunu Etkinleştir",
                initial=True
            ),
            
            # Database Management
            Switch(
                id="trigger_db_update",
                label="🔄 Veritabanını Güncelle",
                initial=False
            ),
            Switch(
                id="trigger_db_rebuild", 
                label="🔨 Veritabanını Yeniden Oluştur",
                initial=False
            ),
        ]
    ).send()


@cl.on_settings_update
async def update_settings(settings):
    """Handle settings updates."""
    global config, pipeline
    
    # Check for database operations first
    if settings.get("trigger_db_update", False):
        await handle_database_update()
        return
        
    if settings.get("trigger_db_rebuild", False):
        await handle_database_rebuild()
        return
    
    # Handle regular settings updates
    await cl.Message("⚙️ **Ayarlar Güncelleniyor...**", author="System").send()
    
    try:
        settings_changed = False
        
        # Update configurations (simplified)
        for key, value in settings.items():
            if hasattr(config, key.upper()) and getattr(config, key.upper()) != value:
                setattr(config, key.upper(), value)
                settings_changed = True
        
        if settings_changed:
            # Reinitialize pipeline with new settings
            pipeline = RAGPipeline(config)
            await cl.Message("✅ **Ayarlar başarıyla güncellendi!**", author="System").send()
        else:
            await cl.Message("ℹ️ **Değişim algılanmadı**", author="System").send()
        
    except Exception as e:
        await cl.Message(f"❌ **Settings update failed:** {str(e)}", author="System").send()


async def handle_database_update():
    """Handle database update operation."""
    await cl.Message("🔄 **Veritabanı güncelleniyor...**", author="Database").send()
    
    try:
        build_database(force_rebuild=False)
        
        global pipeline
        pipeline = RAGPipeline(config)
        stats = pipeline.get_database_stats()
        
        await cl.Message(
            f"✅ **Veritabanı güncellendi!**\n\n- Dosya sayısı: {len(stats.get('files', []))}\n- Chunk sayısı: {stats.get('total_chunks', 0)}",
            author="Database"
        ).send()
        
    except Exception as e:
        await cl.Message(f"❌ **Veritabanı güncelleme hatası:** {str(e)}", author="Database").send()


async def handle_database_rebuild():
    """Handle database rebuild operation."""
    await cl.Message("🔨 **Veritabanı yeniden oluşturuluyor...**", author="Database").send()
    
    try:
        build_database(force_rebuild=True)
        
        global pipeline
        pipeline = RAGPipeline(config)
        stats = pipeline.get_database_stats()
        
        await cl.Message(
            f"✅ **Veritabanı yeniden oluşturuldu!**\n\n- Dosya sayısı: {len(stats.get('files', []))}\n- Chunk sayısı: {stats.get('total_chunks', 0)}",
            author="Database"
        ).send()
        
    except Exception as e:
        await cl.Message(f"❌ **Veritabanı yeniden oluşturma hatası:** {str(e)}", author="Database").send()


async def show_database_status():
    """Show current database status."""
    try:
        if not pipeline:
            return
            
        stats = pipeline.get_database_stats()
        
        if stats.get('total_chunks', 0) > 0:
            await cl.Message(
                f"📊 **Database Durumu:**\n- Dosya: {len(stats.get('files', []))}\n- Chunk: {stats.get('total_chunks', 0)}\n\n✅ **Arama için hazır!**",
                author="Database"
            ).send()
        else:
            await cl.Message(
                f"⚠️ **Database boş!**\n\nVeri yolunu kontrol edin ve veritabanını güncelleyin.",
                author="Database"
            ).send()
            
    except Exception as e:
        await cl.Message(f"⚠️ **Database kontrol hatası:** {str(e)}", author="Database").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with enhanced features."""
    global pipeline, config
    
    if not RAG_AVAILABLE:
        await cl.Message("❌ RAG Pipeline kullanılamıyor!", author="System").send()
        return
    
    user_query = message.content.strip()
    
    # Handle special commands
    if user_query.lower() in ['/help', 'help']:
        await cl.Message(
            "📖 **Yardım:**\n\n- Doğal dilde sorular sorun\n- `/status` - sistem durumu\n- `/docs` - doküman listesi\n\n⚙️ Ayarlar panelini kullanarak konfigürasyonları değiştirin",
            author="Help"
        ).send()
        return
    elif user_query.lower() in ['/status', 'status']:
        await show_database_status()
        return
    elif user_query.lower() in ['/docs', 'show docs', 'documents']:
        await show_available_docs()
        return
    
    # Handle regular search
    if not pipeline:
        await cl.Message("❌ Pipeline başlatılamadı. Ayarları kontrol edin (⚙️).", author="System").send()
        return
    
    # Show searching indicator
    await cl.Message("🔍 **Aranıyor...**", author="System").send()
    
    try:
        start_time = time.time()
        
        print(f"🔍 Arama başlıyor: {user_query}")  # Debug log
        
        # Use the enhanced search engine
        from query_engine import RAGQueryEngine
        engine = RAGQueryEngine(config=config)
        
        print("🔧 Engine oluşturuldu")  # Debug log
        
        # Get sources and response separately
        sources, response = engine.search_documents(user_query)
        
        print(f"📊 Sonuçlar: sources={type(sources)}, response={type(response)}")  # Debug log
        
        end_time = time.time()
        search_time = end_time - start_time
        
        if response is None:
            await cl.Message("❌ **Yanıt üretilemedi**\n\nAyarları kontrol edin.", author="RAG Assistant").send()
            return
        
        # Format response content
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"📝 Response content length: {len(response_content)}")  # Debug log
        
        # Get user settings for animation
        user_settings = cl.user_session.get("settings", {})
        enable_animation = user_settings.get("enable_typing_animation", False)  # Default to False for stability
        
        # Handle successful search
        if isinstance(sources, list) and sources:
            print(f"✅ Başarılı arama - {len(sources)} kaynak bulundu")  # Debug log
            
            # Send response (with or without animation)
            if enable_animation:
                final_content = await simple_typing_animation(response_content, 0.1, "RAG Assistant")
            else:
                await cl.Message(content=response_content, author="RAG Assistant").send()
                final_content = response_content
            
            # Add sources
            sources_section = format_sources_with_links(sources)
            timing_info = f"\n\n---\n⏱️ *Arama süresi: {search_time:.2f} saniye*"
            
            await cl.Message(
                content=sources_section + timing_info,
                author="Sources"
            ).send()
            
        else:
            # Handle error cases
            print(f"❌ Hata durumu: {sources}")  # Debug log
            error_content = f"❌ **Hata:** {sources}\n\n---\n⏱️ *Süre: {search_time:.2f} saniye*"
            await cl.Message(content=error_content, author="RAG Assistant").send()
        
    except Exception as e:
        print(f"❌ HATA DETAYI: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Full error trace
        
        await cl.Message(
            content=f"❌ **Arama Hatası:** {str(e)}\n\nLütfen ayarları kontrol edin (⚙️).",
            author="RAG Assistant"
        ).send()


async def show_available_docs():
    """Show available documents with hyperlinks."""
    global pipeline
    
    try:
        if not pipeline:
            pipeline = RAGPipeline(config)
        
        stats = pipeline.get_database_stats()
        files = stats.get('files', [])
        
        if not files:
            await cl.Message("📂 **Doküman bulunamadı**\n\nVeri dizinini kontrol edin ve veritabanını güncelleyin.", author="Documents").send()
            return
        
        docs_content = "# 📚 Müsait Dokümanlar\n\n"
        
        for i, file_path in enumerate(files[:20], 1):  # Show first 20 files
            file_name = os.path.basename(file_path)
            normalized_path = file_path.replace('\\', '/')
            file_url = f"file:///{normalized_path}"
            docs_content += f"{i}. **[{file_name}]({file_url})**\n{normalized_path}\n\n"
        
        if len(files) > 20:
            docs_content += f"\n... ve {len(files) - 20} dosya daha\n"
        
        docs_content += f"\n**Toplam:** {len(files)} doküman"
        
        await cl.Message(content=docs_content, author="Documents").send()
        
    except Exception as e:
        await cl.Message(f"❌ **Doküman listesi hatası:** {str(e)}", author="Documents").send()


if __name__ == "__main__":
    print("""
🚀 RAG Pipeline 

✨ Features:
- Source display with clickable hyperlinks  
- Enhanced UI with better error handling
- Debug logging for troubleshooting
- Stable Chainlit API compatibility

🎛️ Enhanced Features:
- Comprehensive settings panel
- Database management
- Processing parameter controls
- Feature toggle switches
- Optional typing animation

To run: chainlit run app.py
    """)

    from chainlit.cli import run_chainlit
    run_chainlit(__file__)