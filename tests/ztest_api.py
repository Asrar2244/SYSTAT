import pytest
from app import create_app  # Import from the 'app' package


@pytest.fixture
def client():
    app = create_app()
    with app.test_client() as client:
        yield client

def test_ztest(client):
    data = {
        "confidence_interval": 95,
        "groups": [
            {"group": 1, "size": 40, "proportion": 0.3},
            {"group": 2, "size": 60, "proportion": 0.7}
        ]
    }
    
    # Adjust URL based on the blueprint registration
    response = client.post('/ztest/one_sample_ztest', json=data)  # Ensure URL is correct
    
    assert response.status_code == 200  # Ensure response is successful
    assert 'z_score' in response.json  # Check if z_score is in the response
    assert 'p_value' in response.json  # Check if p_value is in the response
    assert 'confidence_interval' in response.json  # Make sure the confidence interval is in the response
