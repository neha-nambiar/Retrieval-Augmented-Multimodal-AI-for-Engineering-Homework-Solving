"""Modal configuration and shared resources
"""
import modal
from config.settings import HF_CACHE_VOLUME, VLLM_CACHE_VOLUME

# Persistent volumes for performance optimization
# HF_CACHE_VOLUME: Caches downloaded Hugging Face models (ColPali, Qwen, DeepSeek)
# Prevents re-downloading 12GB+ of models on every function restart
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)

# VLLM_CACHE_VOLUME: Persistent storage for vLLM's cache files 
# Improves server startup time by preserving vLLM's cached data between restarts
vllm_cache_vol = modal.Volume.from_name(VLLM_CACHE_VOLUME, create_if_missing=True)

# Shared image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        "transformers",
        "torch",
        "pdf2image",
        "colpali_engine",
        "qwen_vl_utils",
        "httpx",
        "Pillow",
        "schemdraw",
        "matplotlib",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable fast model downloads using hf_transfer
        "VLLM_USE_V1": "1",                 # Use vLLM v1 API for better performance
        "HF_HOME": "/hf_cache",             
        "VLLM_CACHE_ROOT": "/vllm_cache"    
    })
    # Copy local Python modules into container for imports
    .add_local_python_source("config", "src")
    # Copy essential files needed by Modal functions in cloud environment
    .add_local_file("modal_app.py", "/root/modal_app.py")                              
    .add_local_file("config/qwen_instructions.txt", "/root/config/qwen_instructions.txt")    
    .add_local_file("config/deepseek_prompt.txt", "/root/config/deepseek_prompt.txt")        
)