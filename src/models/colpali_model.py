"""ColPali model for PDF indexing and retrieval

Key features:
- Converts PDF pages to embeddings using vision-language model
- Retrieves most relevant pages based on text queries
- Optimized for memory efficiency with batch processing
- Uses T4 GPU 
"""
import modal
from config.settings import PDF_BATCH_SIZE, COLPALI_MODEL
from config.modal_config import image, hf_cache_vol
from modal_app import app
import logging

logger = logging.getLogger(__name__)

@app.cls(
    image=image,
    gpu="T4",
    timeout=10*60, 
    volumes={"/hf_cache": hf_cache_vol}
)
class ColPaliModel:
    @modal.enter()
    def load_models(self):
        """Load ColPali model and processor on container startup"""
        import torch

        from colpali_engine.models import ColPali, ColPaliProcessor
        
        logger.info(f"Loading ColPali model: {COLPALI_MODEL}")
        
        # Load model with bfloat16 precision for memory efficiency
        self.colpali_model = ColPali.from_pretrained(
            COLPALI_MODEL,
            torch_dtype=torch.bfloat16,  
            device_map="cuda:0"          
        )
        
        # Load processor
        self.colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL)
        
        logger.info("ColPali model loaded successfully")

    @modal.method()
    def index_pdf_from_bytes(self, pdf_data: bytes):
        """Convert PDF bytes to images and embeddings"""
        from pdf2image import convert_from_bytes
        import torch
        
        try:
            logger.info("Converting PDF to images")
            images = convert_from_bytes(pdf_data, dpi=150)  
            logger.info(f"PDF converted to {len(images)} pages")
            
            pdf_embeddings = []
            
            # Process in batches
            batches = [images[i : i + PDF_BATCH_SIZE] for i in range(0, len(images), PDF_BATCH_SIZE)]
            logger.info(f"Processing {len(batches)} batches of size {PDF_BATCH_SIZE}")
            
            for i, batch in enumerate(batches):
                logger.debug(f"Processing batch {i+1}/{len(batches)}")
                
                # Process batch and move to CPU to free GPU memory
                batch_images = self.colpali_processor.process_images(batch).to(
                    self.colpali_model.device
                )
                with torch.no_grad():  # Disable gradients for inference
                    batch_embeddings = self.colpali_model(**batch_images)
                    pdf_embeddings += list(batch_embeddings.to("cpu"))
                
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
            
            logger.info(f"PDF indexing completed: {len(pdf_embeddings)} embeddings generated")
            return pdf_embeddings, images
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {str(e)}")

    @modal.method()
    def get_top_k_pages(self, query, pdf_embeddings, images, k=3):
        """Retrieve the top k most relevant pages from the PDF for the input query"""
        import torch
        
        try:
            logger.info(f"Retrieving top {k} pages for query: {query}")
            
            # Process query
            query_input = self.colpali_processor.process_queries([query]).to(self.colpali_model.device)
            
            with torch.no_grad():  # Disable gradients for inference
                query_embedding = self.colpali_model(**query_input)
                scores = self.colpali_processor.score_multi_vector(query_embedding, pdf_embeddings)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Get top k indices with scores
            scored_indices = [(i, scores[i]) for i in range(len(scores))]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, score in scored_indices[:k]]
            

            
            # Return top k images
            return [images[i] for i in top_k_indices]
            
        except Exception as e:
            raise Exception(f"Failed to retrieve relevant pages: {str(e)}")