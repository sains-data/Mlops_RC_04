"""
User Interface for Pothole Detection
Upload images and get predictions
"""

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🚧",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
    }
    .detection-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚧 Pothole Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

# API endpoint
api_url = st.sidebar.text_input("API Endpoint", "http://localhost:8000")

# Get available models
try:
    response = requests.get(f"{api_url}/models")
    if response.status_code == 200:
        available_models = response.json()["models"]
    else:
        available_models = ["yolov8n", "yolov8s"]
except:
    available_models = ["yolov8n", "yolov8s"]
    st.sidebar.warning("⚠️ Cannot connect to API. Using default models.")

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=0
)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# IOU threshold
iou_threshold = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info("""
### 📖 How to use:
1. Upload an image of a road
2. Select a model
3. Adjust thresholds
4. Click 'Detect Potholes'
5. View results
""")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a road image to detect potholes"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Detect button
        if st.button("🔍 Detect Potholes", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Prepare request
                    files = {
                        'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    params = {
                        'model_name': selected_model,
                        'conf_threshold': conf_threshold,
                        'iou_threshold': iou_threshold
                    }
                    
                    # Get prediction with visualization
                    response = requests.post(
                        f"{api_url}/predict/visualize",
                        files=files,
                        params=params
                    )
                    
                    if response.status_code == 200:
                        # Display result in col2
                        with col2:
                            st.header("🎯 Detection Results")
                            
                            # Display annotated image
                            result_image = Image.open(io.BytesIO(response.content))
                            st.image(result_image, caption="Detected Potholes", use_column_width=True)
                        
                        # Get detection details
                        uploaded_file.seek(0)
                        files = {
                            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        
                        detail_response = requests.post(
                            f"{api_url}/predict",
                            files=files,
                            params=params
                        )
                        
                        if detail_response.status_code == 200:
                            result_data = detail_response.json()
                            
                            # Display metrics
                            st.markdown("---")
                            
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            
                            with metrics_col1:
                                st.metric("Model Used", result_data['model_name'])
                            
                            with metrics_col2:
                                st.metric("Detections", result_data['num_detections'])
                            
                            with metrics_col3:
                                st.metric("Inference Time", f"{result_data['inference_time_ms']:.2f} ms")
                            
                            # Display detections
                            if result_data['num_detections'] > 0:
                                st.subheader("📋 Detection Details")
                                
                                for idx, detection in enumerate(result_data['detections'], 1):
                                    with st.expander(f"Detection #{idx}"):
                                        col_a, col_b = st.columns(2)
                                        
                                        with col_a:
                                            st.write(f"**Class:** {detection['class_name']}")
                                            st.write(f"**Confidence:** {detection['confidence']:.3f}")
                                        
                                        with col_b:
                                            bbox = detection['bbox']
                                            st.write(f"**BBox:** ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
                            else:
                                st.info("No potholes detected in this image.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Make sure the API server is running: `python cli.py serve`")

with col2:
    if uploaded_file is None:
        st.header("🎯 Detection Results")
        st.info("👈 Upload an image to start detection")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚧 Pothole Detection MLOps System | Powered by YOLOv8 & MLflow</p>
</div>
""", unsafe_allow_html=True)
