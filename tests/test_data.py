import os
from src.data_processing import load_data

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_creditcard.csv")

def test_data():
    df = load_data(FIXTURE_PATH)
    assert len(df) > 0