"""
Admin Interface for Model Management and Monitoring
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import mlflow
from pathlib import Path
import json

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="🛠️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🛠️ Admin Dashboard - Pothole Detection MLOps")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Configuration")

api_url = st.sidebar.text_input("API Endpoint", "http://localhost:8000")
mlflow_url = st.sidebar.text_input("MLflow URI", "http://localhost:5000")

# Set MLflow tracking URI
try:
    mlflow.set_tracking_uri(mlflow_url)
except:
    st.sidebar.warning("⚠️ Cannot connect to MLflow")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🤖 Models", "📈 Experiments", "📉 Metrics", "🔍 Monitoring"]
)

st.sidebar.markdown("---")

# Helper functions
@st.cache_data(ttl=60)
def get_api_status(url):
    """Check API status"""
    try:
        response = requests.get(f"{url}/health")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


@st.cache_data(ttl=60)
def get_available_models(url):
    """Get available models from API"""
    try:
        response = requests.get(f"{url}/models")
        if response.status_code == 200:
            return response.json()
        return {"models": [], "count": 0}
    except:
        return {"models": [], "count": 0}


def get_mlflow_experiments():
    """Get MLflow experiments"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return experiments
    except:
        return []


def get_experiment_runs(experiment_id):
    """Get runs for an experiment"""
    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment_id])
        return runs
    except:
        return []


# PAGE: OVERVIEW
if page == "📊 Overview":
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # API Status
    with col1:
        api_status = get_api_status(api_url)
        if api_status:
            st.success("✅ API Online")
            st.metric("Status", "Healthy")
        else:
            st.error("❌ API Offline")
            st.metric("Status", "Down")
    
    # MLflow Status
    with col2:
        try:
            experiments = get_mlflow_experiments()
            st.success("✅ MLflow Online")
            st.metric("Experiments", len(experiments))
        except:
            st.error("❌ MLflow Offline")
            st.metric("Experiments", "N/A")
    
    # Available Models
    with col3:
        models_info = get_available_models(api_url)
        st.info("🤖 Models")
        st.metric("Available Models", models_info['count'])
    
    # Timestamp
    with col4:
        st.info("🕐 Last Updated")
        st.metric("Time", datetime.now().strftime("%H:%M:%S"))
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with action_col2:
        if st.button("📊 View MLflow", use_container_width=True):
            st.markdown(f"[Open MLflow UI]({mlflow_url})")
    
    with action_col3:
        if st.button("📖 View API Docs", use_container_width=True):
            st.markdown(f"[Open API Docs]({api_url}/docs)")
    
    st.markdown("---")
    
    # System Information
    st.subheader("💻 System Information")
    
    if api_status:
        info_df = pd.DataFrame({
            "Property": ["Status", "Timestamp", "Available Models"],
            "Value": [
                api_status.get('status', 'N/A'),
                api_status.get('timestamp', 'N/A'),
                ", ".join(api_status.get('available_models', []))
            ]
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)


# PAGE: MODELS
elif page == "🤖 Models":
    st.header("🤖 Model Management")
    
    models_info = get_available_models(api_url)
    
    if models_info['count'] > 0:
        st.success(f"Found {models_info['count']} available models")
        
        # Display models
        for model_name in models_info['models']:
            with st.expander(f"📦 {model_name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Model Name:**", model_name)
                
                with col2:
                    # Check if model exists
                    model_path = Path(f"runs/train/{model_name}_experiment/weights/best.pt")
                    if model_path.exists():
                        st.write("**Status:** 🟢 Available")
                    else:
                        st.write("**Status:** 🔴 Not Found")
                
                with col3:
                    if st.button(f"🧪 Test {model_name}", key=f"test_{model_name}"):
                        st.info(f"Testing {model_name}...")
                
                # Model details
                st.markdown("**Model Details:**")
                
                # Try to load evaluation results
                eval_path = Path(f"outputs/evaluation/{model_name}_metrics.json")
                if eval_path.exists():
                    with open(eval_path, 'r') as f:
                        metrics = json.load(f)
                    
                    metrics_cols = st.columns(5)
                    with metrics_cols[0]:
                        st.metric("mAP@0.5", f"{metrics.get('map50', 0):.3f}")
                    with metrics_cols[1]:
                        st.metric("mAP@0.5:0.95", f"{metrics.get('map50_95', 0):.3f}")
                    with metrics_cols[2]:
                        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    with metrics_cols[3]:
                        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                    with metrics_cols[4]:
                        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                else:
                    st.info("No evaluation metrics available")
    else:
        st.warning("No models available. Please train models first.")


# PAGE: EXPERIMENTS
elif page == "📈 Experiments":
    st.header("📈 MLflow Experiments")
    
    try:
        experiments = get_mlflow_experiments()
        
        if experiments:
            # Experiment selector
            exp_names = [exp.name for exp in experiments]
            selected_exp = st.selectbox("Select Experiment", exp_names)
            
            # Get selected experiment
            exp_obj = next((exp for exp in experiments if exp.name == selected_exp), None)
            
            if exp_obj:
                st.subheader(f"Experiment: {selected_exp}")
                
                # Get runs
                runs = get_experiment_runs(exp_obj.experiment_id)
                
                if runs:
                    st.write(f"**Total Runs:** {len(runs)}")
                    
                    # Create runs dataframe
                    runs_data = []
                    for run in runs:
                        runs_data.append({
                            "Run ID": run.info.run_id[:8],
                            "Status": run.info.status,
                            "Start Time": datetime.fromtimestamp(run.info.start_time/1000).strftime("%Y-%m-%d %H:%M"),
                            "mAP@0.5": run.data.metrics.get('val_map50', 'N/A'),
                            "Precision": run.data.metrics.get('val_precision', 'N/A'),
                            "Recall": run.data.metrics.get('val_recall', 'N/A')
                        })
                    
                    runs_df = pd.DataFrame(runs_data)
                    st.dataframe(runs_df, use_container_width=True, hide_index=True)
                    
                    # Plot metrics
                    if len(runs) > 1:
                        st.subheader("📊 Metrics Comparison")
                        
                        metric_to_plot = st.selectbox(
                            "Select Metric",
                            ["val_map50", "val_precision", "val_recall"]
                        )
                        
                        metric_values = []
                        run_ids = []
                        
                        for run in runs:
                            metric_value = run.data.metrics.get(metric_to_plot)
                            if metric_value is not None:
                                metric_values.append(metric_value)
                                run_ids.append(run.info.run_id[:8])
                        
                        if metric_values:
                            fig = px.bar(
                                x=run_ids,
                                y=metric_values,
                                labels={'x': 'Run ID', 'y': metric_to_plot},
                                title=f'{metric_to_plot} across runs'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No runs found for this experiment")
        else:
            st.warning("No experiments found in MLflow")
    
    except Exception as e:
        st.error(f"Error loading experiments: {str(e)}")


# PAGE: METRICS
elif page == "📉 Metrics":
    st.header("📉 Model Metrics")
    
    # Load comparison results if available
    comparison_path = Path("outputs/evaluation/model_comparison.png")
    
    if comparison_path.exists():
        st.image(str(comparison_path), caption="Model Comparison", use_column_width=True)
    else:
        st.info("No model comparison available. Run comparison first.")
    
    st.markdown("---")
    
    # Individual model metrics
    st.subheader("Individual Model Metrics")
    
    models_info = get_available_models(api_url)
    
    if models_info['count'] > 0:
        selected_model = st.selectbox("Select Model", models_info['models'])
        
        # Try to load model's confusion matrix
        cm_path = Path(f"runs/train/{selected_model}_experiment/confusion_matrix.png")
        
        if cm_path.exists():
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(str(cm_path), caption="Confusion Matrix", use_column_width=True)
            
            with col2:
                # Display metrics table
                eval_path = Path(f"outputs/evaluation/{selected_model}_metrics.json")
                if eval_path.exists():
                    with open(eval_path, 'r') as f:
                        metrics = json.load(f)
                    
                    metrics_df = pd.DataFrame({
                        "Metric": list(metrics.keys()),
                        "Value": [f"{v:.4f}" for v in metrics.values()]
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No confusion matrix found for {selected_model}")


# PAGE: MONITORING
elif page == "🔍 Monitoring":
    st.header("🔍 System Monitoring")
    
    st.info("🚧 Monitoring features coming soon!")
    
    # Placeholder for monitoring features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Inference Latency")
        st.write("Average latency tracking")
        # Placeholder chart
        latency_data = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'Latency (ms)': [45, 50, 48, 52, 49, 51, 47, 50, 48, 49]
        })
        fig = px.line(latency_data, x='Time', y='Latency (ms)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Request Count")
        st.write("API request monitoring")
        # Placeholder chart
        request_data = pd.DataFrame({
            'Time': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'Requests': [120, 145, 132, 168, 155, 142, 138, 150, 145, 152]
        })
        fig = px.bar(request_data, x='Time', y='Requests')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🛠️ Admin Dashboard | MLOps Pothole Detection System</p>
</div>
""", unsafe_allow_html=True)
