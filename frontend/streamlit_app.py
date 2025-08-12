"""Streamlit frontend application"""
import streamlit as st
import base64
import modal
import io
from PIL import Image
import re
import logging
import time

logger = logging.getLogger(__name__)

# Connect to deployed Modal function
solve_ee_problem = modal.Function.from_name("ee-tutor", "solve_ee_problem")

def render_file_uploader():
    """Render file upload sidebar"""
    with st.sidebar:
        st.header("📁 Upload Files")
        uploaded_pdf = st.file_uploader("Upload textbook PDF", type="pdf")
        uploaded_images = st.file_uploader("Upload question images (optional)", 
                                         type=["png", "jpg", "jpeg"], 
                                         accept_multiple_files=True)
    
    return uploaded_pdf, uploaded_images

def render_status_panel(uploaded_pdf, uploaded_images):
    """Render status panel"""
    st.header("📊 Status")
    if uploaded_pdf:
        st.success("✅ PDF uploaded")
    else:
        st.warning("⚠️ PDF required")
    
    if uploaded_images:
        st.info(f"📷 {len(uploaded_images)} image(s) uploaded")

def display_solution_results(result):
    """Display the complete solution results in tabs"""
    if result["success"]:
        st.success("✅ Solution generated successfully!")
        
        tab1, tab2, tab3 = st.tabs(["📝 Solution", "💻 Generated Code", "ℹ️ Metadata"])
        
        with tab1:
            # Display textual solution with LaTeX support
            solution_text = result["textual_solution"]
            latex_text = re.sub(r'\[ ([^\]]+) \]', r'$ \1 $', solution_text)
            st.markdown(latex_text)
            
            # Display circuit diagram
            if result["circuit_diagram"]["success"]:
                st.subheader("🔌 Circuit Diagram")
                img_data = base64.b64decode(result["circuit_diagram"]["image_base64"])
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption="Generated Circuit Diagram", use_container_width=True)
            else:
                st.error(f"❌ Circuit generation failed: {result['circuit_diagram']['error']}")
                st.text(result["circuit_diagram"].get("traceback", "No traceback available"))
        
        with tab2:
            if result["circuit_diagram"]["success"]:
                st.code(result["metadata"]["generated_code"], language="python")
            else:
                st.error("No code generated due to circuit diagram failure")
        
        with tab3:
            # Convert processing time to minutes and seconds
            time_str = result["metadata"]["total_processing_time"]
            total_seconds = float(time_str.replace('s', ''))
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            formatted_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            st.json(result["metadata"])
            
    else:
        st.error(f"❌ Error: {result['error']}")

def main():
    st.title("⚡ Electrical Engineering Solution Generator")
    st.markdown("Upload a textbook PDF and ask electrical engineering questions to get detailed solutions with circuit diagrams.")
    
    # File upload components
    uploaded_pdf, uploaded_images = render_file_uploader()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("❓ Your Question")
        user_query = st.text_area("Enter your electrical engineering question:", 
                                 height=150,
                                 placeholder="e.g., Find the voltage across R2 in the given circuit...")
    
    with col2:
        render_status_panel(uploaded_pdf, uploaded_images)
    
    # Generate solution button
    if st.button("🚀 Generate Solution", type="primary", use_container_width=True):
        logger.info("User clicked Generate Solution button")
        st.session_state.demo_start_time = time.time()
        
        if not user_query.strip():
            st.error("Please enter a question.")
            return
        
        if not uploaded_pdf:
            st.error("Please upload a textbook PDF.")
            return
        
        with st.spinner("🔄 Processing your question..."):
            try:
                # Convert uploads to required format
                pdf_data = uploaded_pdf.read()
                user_images_bytes = []
                
                if uploaded_images:
                    for img_file in uploaded_images:
                        img_data = img_file.read()
                        user_images_bytes.append(img_data)
                
                # Call Modal function
                result = solve_ee_problem.remote(
                    user_query=user_query,
                    pdf_data=pdf_data,
                    user_images_bytes=user_images_bytes if user_images_bytes else None
                )
                
                # Display results
                display_solution_results(result)
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="EE Solution Generator",
        page_icon="⚡",
        layout="wide"
    )
    main()