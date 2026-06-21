from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.analysis.eda import create_eda_overview
from src.configs.settings import Config
from src.data_processing.cleaning import DataCleaner, clean_raw_data
from src.data_processing.loader import load_dataset
from 