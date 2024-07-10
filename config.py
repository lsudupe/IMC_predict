import os

ROOT_DIR = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    + "/IMC_Spatial_predictions"
)

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data/raw_data")
