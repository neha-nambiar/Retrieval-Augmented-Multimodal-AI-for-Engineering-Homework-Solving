"""Qwen model for question analysis and solution generation"""
import modal
import base64
import io
import httpx
import time
import subprocess
from config.settings import (
    QWEN_MODEL, QWEN_PORT, QWEN_INSTRUCTIONS, MAX_TOKENS, TEMPERATURE,
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
@modal.web_server(port=QWEN_PORT, startup_timeout=10*60)
def serve_qwen():
    cmd = [
        "vllm", "serve",
        QWEN_MODEL,
        "--host", "0.0.0.0",
        "--port", str(QWEN_PORT),
        "--served-model-name", QWEN_MODEL,
        "--enforce-eager",  # Faster boot time, trades off some inference performance
    ]
    logger.info(f"Starting Qwen vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd)

@app.function(
    image=image,
    gpu="A10",
    timeout=20*60,
    volumes={"/hf_cache": hf_cache_vol}
)
def analyze_question_with_qwen_url(server_url: str, user_query: str, user_images_b64=None, context_images=None):
    """Call the vLLM server at given URL to analyze question with Qwen2.5-VL"""
    
    # Wait for server to be ready
    logger.info(f"Checking if Qwen server is ready at {server_url}")
    for i in range(HEALTH_CHECK_MAX_RETRIES):
        try:
            with httpx.Client(base_url=server_url, timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = client.get("/health")
                if response.status_code == 200:
                    logger.info("Qwen server is ready")
                    break
        except Exception as e:
            logger.warning(f"Attempt {i+1}: Server not ready, waiting... ({str(e)})")
            time.sleep(HEALTH_CHECK_SLEEP)
            if i == HEALTH_CHECK_MAX_RETRIES - 1:
                logger.error(f"Qwen server failed to start after {HEALTH_CHECK_MAX_RETRIES} attempts")
                raise Exception("Qwen server failed to start")

    # Prepare message content
    content = [{"type": "text", "text": QWEN_INSTRUCTIONS + "\n\nQuestion: " + user_query}]
    
    # Add user uploaded images as base64
    if user_images_b64:
        for img_b64 in user_images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
    
    # Add context images from PDF
    if context_images:
        content.append({"type": "text", "text": "\n\nRelevant textbook pages for reference:"})
        for img in context_images:
            if hasattr(img, 'save'):  # PIL Image
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })

    messages = [{"role": "user", "content": content}]
    
    # Call vLLM server
    with httpx.Client(base_url=server_url, timeout=600) as client:
        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"]