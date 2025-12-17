"""
Unit tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "available_models" in data


def test_list_models():
    """Test list models endpoint"""
    response = client.get("/models")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data
    assert isinstance(data["models"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
