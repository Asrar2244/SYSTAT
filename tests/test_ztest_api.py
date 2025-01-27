import pytest
from run import create_app

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
    
    response = client.post('/ztest', json=data)
    
    assert response.status_code == 200
    assert 'z_score' in response.json
    assert 'p_value' in response.json
