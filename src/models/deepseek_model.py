"""DeepSeek model for circuit diagram code generation"""
import modal
import httpx
import time
import json
import subprocess
from config.settings import (
    DEEPSEEK_MODEL, DEEPSEEK_PORT, DEEPSEEK_PROMPT_TEMPLATE, MAX_TOKENS, TEMPERATURE,
    HEALTH_CHECK_MAX_RETRIES, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_SLEEP
)
from config.modal_config import image, hf_cache_vol, vllm_cache_vol
from modal_app import app
import logging

logger = logging.getLogger(__name__)

@app.function(
    image=image,
    gpu="A10",
    min_containers=1,  # Keep one container always warm
    max_containers=1,  # Prevent multiple instances
    timeout=30*60,
    volumes={"/hf_cache": hf_cache_vol, "/vllm_cache": vllm_cache_vol}
)
@modal.web_server(port=DEEPSEEK_PORT, startup_timeout=10*60)
def serve_deepseek():
    cmd = [
        "vllm", "serve",
        DEEPSEEK_MODEL,
        "--host", "0.0.0.0", 
        "--port", str(DEEPSEEK_PORT),
        "--served-model-name", DEEPSEEK_MODEL,
        "--enforce-eager",  # Faster boot time, trades off some inference performance
    ]
    logger.info(f"Starting DeepSeek vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd)

@app.function(
    image=image,
    gpu="A10",
    timeout=20*60,
    volumes={"/hf_cache": hf_cache_vol}
)
def generate_circuit_code(server_url: str, user_query: str, qwen_solution: str):
    """Generate schemdraw code using DeepSeek with examples and context"""
    
    # Wait for server to be ready
    logger.info(f"Checking if DeepSeek server is ready at {server_url}")
    for i in range(HEALTH_CHECK_MAX_RETRIES):
        try:
            with httpx.Client(base_url=server_url, timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = client.get("/health")
                if response.status_code == 200:
                    logger.info("DeepSeek server is ready")
                    break
        except Exception as e:
            logger.warning(f"Attempt {i+1}: Server not ready, waiting... ({str(e)})")
            time.sleep(HEALTH_CHECK_SLEEP)
            if i == HEALTH_CHECK_MAX_RETRIES - 1:
                logger.error(f"DeepSeek server failed to start after {HEALTH_CHECK_MAX_RETRIES} attempts")
                raise Exception("DeepSeek server failed to start")
    
    # Create enhanced prompt by combining:
    # - DEEPSEEK_PROMPT_TEMPLATE: Contains 6 schemdraw examples for few-shot learning
    # - user_query: The original question (e.g., "Find voltage across R2")
    # - qwen_solution: Provides component values (resistor ohms, capacitor farads, etc.) for DeepSeek to label circuits
    enhanced_prompt = DEEPSEEK_PROMPT_TEMPLATE.replace("{question}", user_query).replace("{solution}", qwen_solution)
    
    # Structured output for reliable code extraction - JSON format ensures consistent parsing
    messages = [
        {"role": "system", "content": "You are a Python expert. Return ONLY valid JSON with a 'code' field containing Python code."},
        {"role": "user", "content": enhanced_prompt}
    ]
    
    # Call vLLM server
    with httpx.Client(base_url=server_url, timeout=600) as client:
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Debug: Log complete raw response
        logger.debug(f"Raw DeepSeek response: {content}")
        
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                json_response = json.loads(content)
                if 'code' in json_response:
                    extracted_code = json_response['code']
                    logger.debug(f"Extracted from JSON: {extracted_code}")
                    return extracted_code
        except json.JSONDecodeError:
            logger.debug("Not valid JSON, falling back to text extraction")
        
        # Fallback: Extract code between backticks
        if "```python" in content:
            code_start = content.find("```python") + 9
            code_end = content.rfind("```")
            if code_end != -1 and code_end > code_start:
                content = content[code_start:code_end].strip()
        elif "```" in content:
            first_backticks = content.find("```")
            last_backticks = content.rfind("```")
            if first_backticks != -1 and last_backticks != -1 and last_backticks > first_backticks:
                content = content[first_backticks + 3:last_backticks].strip()
        
        logger.debug(f"Final extracted code: {content}")
        return content