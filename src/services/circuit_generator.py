"""Circuit diagram generation service"""
import modal
import io
import base64
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr
from config.modal_config import image
from config.settings import CIRCUIT_DPI
from modal_app import app
import logging

logger = logging.getLogger(__name__)

@app.function(
    image=image,
    timeout=10*60
)
def run_generated_code(schemdraw_code: str):
    """
    Execute schemdraw code and return the generated circuit diagram
    """
    try:
        # Create a safe execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'schemdraw': None,
            'elm': None,
            'matplotlib': matplotlib,
            'plt': matplotlib.pyplot
        }
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Debug: Log code being executed
            logger.debug(f"Executing schemdraw code:\n {schemdraw_code}")
            
            # Execute the schemdraw code
            exec(schemdraw_code, exec_globals)
            
            # Save matplotlib figure to PNG bytes (figures can't be directly encoded)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=CIRCUIT_DPI)
            img_buffer.seek(0)
            
            # Convert PNG bytes to base64 string (JSON can't contain binary data)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close('all')  
            
        return {
            "success": True,
            "image_base64": img_base64,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "schemdraw_code": schemdraw_code
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error executing schemdraw code: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Error executing schemdraw code: {str(e)}",
            "traceback": error_trace,
            "schemdraw_code": schemdraw_code
        }