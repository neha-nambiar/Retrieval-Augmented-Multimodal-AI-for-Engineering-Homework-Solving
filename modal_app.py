"""Main Modal application entry point

Follows Modal documentation pattern:
- modal_app.py defines the modal.App as 'app'
- Other modules import 'app' from modal_app and decorate functions
"""
import modal
import base64
import time
import logging

logger = logging.getLogger(__name__)

# Create the shared app instance - other modules will import this
app = modal.App("ee-tutor")

# Import modules to register their decorated functions with the shared app
# This ensures Modal discovers all functions when deploying modal_app.py
import src.models.colpali_model
import src.models.qwen_model
import src.models.deepseek_model
import src.services.circuit_generator

# Import dependencies for the orchestrator function
from typing import List, Optional
from config.modal_config import image, hf_cache_vol

@app.function(
    image=image,
    timeout=20*60,
    volumes={"/hf_cache": hf_cache_vol}
)
def solve_ee_problem(
    user_query: str,
    pdf_data: bytes,
    user_images_bytes: Optional[List[bytes]] = None
):
    """
    Main orchestrator function that solves electrical engineering problems step by step
    
    Args:
        user_query: The student's question text
        pdf_data: The PDF file as bytes
        user_images_bytes: List of image bytes (optional)
    
    Returns:
        Dictionary containing the complete solution with text, circuit diagram, and metadata
    """
    
    try:
        logger.info(f"User submitted question: {user_query[:100]}...")
        start_time = time.time()
        
        # Step 1: Process user images (convert bytes to base64 for Qwen)
        step_start = time.time()
        logger.info("Processing user images...")
        user_images_b64 = []
        if user_images_bytes:
            for img_bytes in user_images_bytes:
                img_b64 = base64.b64encode(img_bytes).decode()
                user_images_b64.append(img_b64)
        logger.info(f"Image processing completed in {time.time() - step_start:.2f}s")
        
        # Steps 2-3: Parallel execution of PDF processing and server setup
        step_start = time.time()
        logger.info("Processing PDF and setting up servers in parallel...")
        
        # Start PDF processing in parallel
        from src.models.colpali_model import ColPaliModel
        from src.models.qwen_model import serve_qwen
        from src.models.deepseek_model import serve_deepseek
        
        colpali = ColPaliModel()
        pdf_task = colpali.index_pdf_from_bytes.spawn(pdf_data)
        
        # Setup server URLs for Qwen and DeepSeek
        qwen_server_url = serve_qwen.get_web_url()
        deepseek_server_url = serve_deepseek.get_web_url()
        
        # Wait for PDF processing and get relevant pages
        pdf_embeddings, pdf_images = pdf_task.get()
        relevant_pages = colpali.get_top_k_pages.remote(
            user_query, pdf_embeddings, pdf_images, k=3
        )
        logger.info(f"PDF processing and page retrieval completed in {time.time() - step_start:.2f}s")
        
        # Step 4: Use Qwen2.5-VL to analyze question and generate solution
        step_start = time.time()
        logger.info("Analyzing question and generating solution...")
        from src.models.qwen_model import analyze_question_with_qwen_url
        textual_solution = analyze_question_with_qwen_url.remote(
            qwen_server_url, user_query, user_images_b64, relevant_pages
        )
        logger.info(f"Qwen analysis completed in {time.time() - step_start:.2f}s")
        
        # Step 5: Use DeepSeek to generate schemdraw code
        step_start = time.time()
        logger.info("Generating circuit diagram code...")
        from src.models.deepseek_model import generate_circuit_code
        circuit_schemdraw = generate_circuit_code.remote(
            deepseek_server_url, user_query, textual_solution
        )
        logger.info(f"DeepSeek code generation completed in {time.time() - step_start:.2f}s")
        
        # Step 6: Compile schemdraw to generate circuit image
        step_start = time.time()
        logger.info("Compiling circuit diagram...")
        from src.services.circuit_generator import run_generated_code
        circuit_result = run_generated_code.remote(circuit_schemdraw)
        logger.info(f"Circuit compilation completed in {time.time() - step_start:.2f}s")
        
        # Step 7: Prepare final response
        result = {
            "success": True,
            "question": user_query,
            "textual_solution": textual_solution,
            "circuit_diagram": circuit_result,
            "metadata": {
                "num_relevant_pages": len(relevant_pages),
                "has_user_images": len(user_images_b64) > 0 if user_images_b64 else False,
                "generated_code": circuit_schemdraw,
                "total_processing_time": f"{time.time() - start_time:.2f}s"
            }
        }
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving EE problem: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "question": user_query
        }

