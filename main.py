"""
Main entry point for the application
Can be used to run different components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("🚧 Pothole Detection MLOps System")
    print("=" * 50)
    print("\nAvailable commands:")
    print("  python cli.py --help          # See all CLI commands")
    print("  python cli.py train           # Train a model")
    print("  python cli.py serve           # Start API server")
    print("  streamlit run ui/user_app.py  # Start User UI")
    print("  streamlit run ui/admin_app.py # Start Admin UI")
    print("  mlflow ui                     # Start MLflow UI")
    print("\nFor detailed setup: see SETUP.md")
    print("=" * 50)
