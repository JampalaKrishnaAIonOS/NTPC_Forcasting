"""
Model Loader Module
Loads pre-trained models from pickle files.
"""
import pickle
import os
from pathlib import Path
from typing import Optional, List

MODEL_DIR = Path(__file__).parent.parent / "model" / "model"


PLANT_DIR_MAP = {
    'barh': 'Barh',
    'dadri': 'Dadri',
    'kudgi': 'Kudgi'
}


def _find_model_file(model_name: str, plant_key: str) -> Optional[Path]:
    """Find a model file for given model name and plant key.

    Supports common extensions: .pkl, .cbm, .keras
    """
    plant_folder = PLANT_DIR_MAP.get(plant_key.lower())
    if not plant_folder:
        return None

    plant_dir = MODEL_DIR / plant_folder
    if not plant_dir.exists():
        return None

    candidates = []
    # typical filename patterns: <model>_<Plant>.<ext> or <model>_<Plant>.pkl
    for ext in ('.pkl', '.cbm', '.keras', '.h5'):
        fname = f"{model_name}_{plant_folder}{ext}"
        fpath = plant_dir / fname
        if fpath.exists():
            candidates.append(fpath)

    # if multiple candidates, prefer .pkl, then .keras/.h5, then .cbm
    if not candidates:
        return None

    for ext in ('.pkl', '.keras', '.h5', '.cbm'):
        for c in candidates:
            if c.suffix == ext:
                return c

    return candidates[0]


def load_model(model_name: str, plant_key: str):
    """Load a plant-specific pre-trained model.

    Args:
        model_name: model type prefix (e.g., 'sarima', 'xgboost', 'lstm')
        plant_key: plant identifier ('barh','dadri','kudgi')

    Returns:
        Loaded model object
    """
    model_file = _find_model_file(model_name, plant_key)
    if model_file is None:
        raise FileNotFoundError(f"Model file for {model_name} at plant {plant_key} not found in {MODEL_DIR}")

    # Keras LSTM models use .keras/.h5
    if model_file.suffix in ('.keras', '.h5'):
        try:
            try:
                from tensorflow.keras.models import load_model as _kload
            except Exception:
                from keras.models import load_model as _kload
        except Exception as e:
            raise RuntimeError("Keras is required to load LSTM models") from e
        return _kload(str(model_file))

    # CatBoost could be .cbm (native) or .pkl
    if model_file.suffix == '.cbm':
        try:
            from catboost import CatBoost
            model = CatBoost()
            model.load_model(str(model_file))
            return model
        except Exception:
            # fallback to returning path for caller
            raise

    # Default: load with pickle
    with open(model_file, 'rb') as f:
        return pickle.load(f)


def list_available_models() -> List[str]:
    """Scan model directory subfolders and return set of model types available.

    Scans filenames like "sarima_Barh.pkl" and returns ['sarima', 'xgboost', ...]
    """
    models = set()
    if not MODEL_DIR.exists():
        return []

    for plant_folder in PLANT_DIR_MAP.values():
        plant_dir = MODEL_DIR / plant_folder
        if not plant_dir.exists():
            continue
        for f in plant_dir.iterdir():
            if not f.is_file():
                continue
            name = f.stem  # e.g., 'sarima_Barh'
            if '_' in name:
                mtype = name.split('_')[0].lower()
                models.add(mtype)

    return sorted(models)
