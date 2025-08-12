"""Configuration settings for the EE Tutor System"""

# Model configurations
COLPALI_MODEL = "vidore/colpali-v1.3"          # Direct Modal function calls (no server/port needed)

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"     # vLLM HTTP server
QWEN_PORT = 8000                               # Port for Qwen vLLM server

DEEPSEEK_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"  # vLLM HTTP server  
DEEPSEEK_PORT = 8001                                         # Port for DeepSeek vLLM server 

# Modal persistent volume names
HF_CACHE_VOLUME = "huggingface-cache"    
VLLM_CACHE_VOLUME = "vllm-cache"                 

# Processing configurations
PDF_BATCH_SIZE = 4
TOP_K_PAGES = 3
MAX_TOKENS = 1024
TEMPERATURE = 0.1
CIRCUIT_DPI = 150                      # Circuit diagram image resolution (DPI)

# Health check configurations - wait for vLLM servers to load models and become ready
HEALTH_CHECK_MAX_RETRIES = 30          # Max attempts (30 × 10s = 5min total wait)
HEALTH_CHECK_TIMEOUT = 30              # Per-request timeout in seconds
HEALTH_CHECK_SLEEP = 10                # Wait between retry attempts in seconds

# Load prompt templates from files
from pathlib import Path
QWEN_INSTRUCTIONS = (Path(__file__).parent / 'qwen_instructions.txt').read_text().strip()
DEEPSEEK_PROMPT_TEMPLATE = (Path(__file__).parent / 'deepseek_prompt.txt').read_text().strip()