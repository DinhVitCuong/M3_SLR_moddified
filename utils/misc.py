from copyreg import pickle
import yaml
from sklearn.model_selection import KFold
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def neq_load_customized(model, pretrained_dict, verbose=False):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print(list(model_dict.keys()))
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape==v.shape:
            tmp[k] = v
        else:
            if verbose:
                print(k)
    if verbose:
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
            elif model_dict[k].shape != pretrained_dict[k].shape:
                print(k, 'shape mis-matched, not loaded')
        print('===================================\n')

    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def parse_csv_like_label_map(path: Path) -> Optional[Dict[int, str]]:
    suffix = path.suffix.lower()
    delimiter = ","
    if suffix == ".tsv":
        delimiter = "\t"

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            lower_to_real = {c.lower(): c for c in fieldnames}

            id_candidates = ["label_id", "id", "idx", "class_id", "label", "label_idx"]
            text_candidates = ["gloss", "keyword", "word", "meaning", "name", "label_name", "class_name", "text"]

            id_col = next((lower_to_real[c] for c in id_candidates if c in lower_to_real), None)
            txt_col = next((lower_to_real[c] for c in text_candidates if c in lower_to_real), None)

            if id_col is not None and txt_col is not None:
                out = {}
                for row in reader:
                    try:
                        out[int(row[id_col])] = str(row[txt_col])
                    except Exception:
                        continue
                if out:
                    return out
    except Exception:
        return None

    # fallback: try first 2 columns
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            raw_reader = csv.reader(f, delimiter=delimiter)
            rows = list(raw_reader)
        if len(rows) < 2:
            return None

        out = {}
        ok_any = False
        for row in rows[1:]:
            if len(row) < 2:
                continue
            try:
                out[int(row[0])] = str(row[1])
                ok_any = True
            except Exception:
                continue

        if ok_any:
            return out
    except Exception:
        return None

    return None

def load_label_map(label_map_csv: Optional[Path], num_classes: int) -> Tuple[Dict[int, str], Optional[Path]]:
    if label_map_csv is not None:
        if not label_map_csv.exists():
            raise FileNotFoundError(f"Không tìm thấy label map csv: {label_map_csv}")
        parsed = parse_csv_like_label_map(label_map_csv)
        if parsed is None:
            raise ValueError(
                "Không parse được label map csv."
            )
        for i in range(num_classes):
            parsed.setdefault(i, f"cls_{i}")
        return parsed, label_map_csv

    return {i: f"cls_{i}" for i in range(num_classes)}, None



