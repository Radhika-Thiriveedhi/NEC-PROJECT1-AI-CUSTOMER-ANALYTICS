import pandas as pd


def load_data():
    """
    Load Customer Personality Analysis dataset
    """
    df = pd.read_csv("dataset/marketing_campaign.csv", sep="\t")
    return df