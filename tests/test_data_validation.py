from src.data_processing import load_data

def test_columns_exist():
    url_dados = "https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1"
    df = load_data(url_dados)

    expected = ["Time", "Amount", "Class"]

    for col in expected:
        assert col in df.columns