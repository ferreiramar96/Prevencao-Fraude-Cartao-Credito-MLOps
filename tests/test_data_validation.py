import os
from src.data_processing import load_data

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_creditcard.csv")

def test_columns_exist():
    df = load_data(FIXTURE_PATH)

    expected = ["Time", "Amount", "Class"]

    for col in expected:
        assert col in df.columns