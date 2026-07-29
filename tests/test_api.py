"""
Integration tests for the FastAPI /evaluate endpoint.
Uses FastAPI's TestClient (runs in-process, no server needed).
"""
import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


def test_evaluate_endpoint_status():
    """Endpoint should return 200 for a valid request."""
    response = client.post(
        "/evaluate",
        json={
            "generated_text": "The Eiffel Tower is in London.",
            "source_text": "The Eiffel Tower is in Paris, France.",
        },
    )
    assert response.status_code == 200


def test_evaluate_endpoint_schema():
    """Response must conform to the HallucinationReport schema."""
    response = client.post(
        "/evaluate",
        json={
            "generated_text": "The Eiffel Tower is in London.",
            "source_text": "The Eiffel Tower is in Paris, France.",
        },
    )
    data = response.json()
    assert "execution_time_seconds" in data
    assert "total_claims_checked" in data
    assert "details" in data
    assert isinstance(data["details"], list)


def test_evaluate_catches_contradiction():
    """A clear factual error should be labelled as Contradiction."""
    response = client.post(
        "/evaluate",
        json={
            "generated_text": "The Eiffel Tower is in London.",
            "source_text": "The Eiffel Tower is in Paris, France.",
        },
    )
    data = response.json()
    assert data["total_claims_checked"] == 1
    assert data["details"][0]["nli_label"] == "Contradiction"
    assert data["details"][0]["confidence"] > 0.5


def test_evaluate_faithful_response():
    """A factually correct response should not be labelled as Contradiction."""
    response = client.post(
        "/evaluate",
        json={
            "generated_text": "The Eiffel Tower is in Paris.",
            "source_text": "The Eiffel Tower is a landmark located in Paris, France.",
        },
    )
    data = response.json()
    assert data["status_code"] != 500 if "status_code" in data else True
    labels = [d["nli_label"] for d in data["details"]]
    assert "Contradiction" not in labels


def test_evaluate_missing_field_returns_422():
    """Missing required field should return 422 Unprocessable Entity."""
    response = client.post(
        "/evaluate",
        json={"generated_text": "Only one field provided."},
    )
    assert response.status_code == 422


def test_evaluate_empty_strings():
    """Empty strings should not crash the server."""
    response = client.post(
        "/evaluate",
        json={"generated_text": "", "source_text": ""},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_claims_checked"] == 0
