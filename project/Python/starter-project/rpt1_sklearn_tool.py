"""
rpt1_sklearn_tool.py
--------------------
Drop-in replacement for the SAP-RPT-1 CrewAI tool used in the SAP CodeJam
"Build code-based AI Agents on SAP BTP".

It accepts exactly the same payload schema that Exercise 03 defines and returns
predictions in the same shape that RPT-1 would return, so the rest of
basic_agent.py can stay unchanged.

Usage
-----
Replace these two imports in basic_agent.py:

    from gen_ai_hub.proxy.native.sap.client import RPTClient
    from payload import payload

    rpt1_client = RPTClient()

    @tool("call_rpt1")
    def call_rpt1(payload: dict) -> str:
        ...

With:

    from rpt1_sklearn_tool import call_rpt1
    from payload import payload

Everything else (agent definition, task, crew, main) stays the same.

Requirements
------------
    pip install scikit-learn pandas numpy
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

try:
    from crewai.tools import tool
except ImportError:
    # Graceful fallback so the module can be tested without a full crewai install
    def tool(name):
        def decorator(fn):
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Core prediction logic (no CrewAI dependency)
# ---------------------------------------------------------------------------

def _encode_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Convert all columns to numeric values the sklearn models can consume.

    - date columns  → integer (days since Unix epoch)
    - string columns → LabelEncoder integer codes  (-1 for unknowns / NaN)
    - numeric columns → pass through (NaN → column mean)
    """
    out = df.copy()
    for col, meta in schema.items():
        if col not in out.columns:
            continue
        dtype = meta.get("dtype", "string")

        if dtype == "date":
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out[col] = (out[col] - pd.Timestamp("1970-01-01")).dt.days.fillna(-1)

        elif dtype == "numeric":
            out[col] = pd.to_numeric(out[col], errors="coerce")
            col_mean = out[col].mean()
            out[col] = out[col].fillna(col_mean if not np.isnan(col_mean) else 0)

        else:  # string / categorical
            le = LabelEncoder()
            out[col] = out[col].astype(str).fillna("__missing__")
            le.fit(out[col])
            out[col] = le.transform(out[col])

    return out


def _predict_payload(payload: dict) -> dict:
    """
    Core logic: parse the payload, train a model per target column on the
    labelled rows, then predict the [PREDICT] rows.

    Returns a dict that mirrors the RPT-1 response shape:
        {
          "predictions": [
            {
              "ITEM_ID": "ART_003",
              "predictions": {
                "INSURANCE_VALUE": {"value": 41500000.0},
                "ITEM_CATEGORY": {"value": "Painting", "confidence": 0.87}
              }
            },
            ...
          ]
        }
    """
    rows = payload["rows"]
    schema = payload.get("data_schema", {})
    target_configs = payload["prediction_config"]["target_columns"]
    index_col = payload.get("index_column", None)

    placeholder = "__PREDICT__"  # internal sentinel

    # -----------------------------------------------------------------------
    # Build a single dataframe; mark prediction rows per target
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)

    target_names = [tc["name"] for tc in target_configs]
    task_types   = {tc["name"]: tc["task_type"] for tc in target_configs}
    pred_placeholder = {tc["name"]: tc.get("prediction_placeholder", "[PREDICT]")
                        for tc in target_configs}

    # Replace the [PREDICT] strings with NaN so we can distinguish them
    for tname in target_names:
        if tname in df.columns:
            df[tname] = df[tname].apply(
                lambda v: np.nan if str(v) == pred_placeholder[tname] else v
            )

    # Feature columns = everything that isn't a target and isn't the index
    feature_cols = [
        c for c in df.columns
        if c not in target_names and c != index_col
    ]

    # Encode all features once
    df_enc = _encode_features(df[feature_cols + target_names], schema)

    feature_matrix = df_enc[feature_cols].values

    results = {}  # index_value → {target: prediction_detail}

    # -----------------------------------------------------------------------
    # Train one model per target column, predict missing rows
    # -----------------------------------------------------------------------
    for tc in target_configs:
        tname     = tc["name"]
        task_type = tc["task_type"]

        target_series = df[tname]          # original (with NaN for [PREDICT])
        train_mask    = target_series.notna()
        pred_mask     = target_series.isna()

        if train_mask.sum() < 2:
            # Not enough labelled data to train; fall back to mode/mean
            if task_type == "regression":
                fallback = float(target_series[train_mask].mean() or 0)
                for idx in df[pred_mask].index:
                    key = str(df.loc[idx, index_col]) if index_col else str(idx)
                    results.setdefault(key, {})[tname] = {"value": fallback}
            else:
                fallback = str(target_series[train_mask].mode().iloc[0]) if train_mask.sum() else "Unknown"
                for idx in df[pred_mask].index:
                    key = str(df.loc[idx, index_col]) if index_col else str(idx)
                    results.setdefault(key, {})[tname] = {"value": fallback, "confidence": 1.0}
            continue

        X_train = feature_matrix[train_mask]
        X_pred  = feature_matrix[pred_mask]

        if task_type == "regression":
            y_train = pd.to_numeric(target_series[train_mask], errors="coerce").fillna(0).values
            model   = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_hat   = model.predict(X_pred)

            for i, idx in enumerate(df[pred_mask].index):
                key = str(df.loc[idx, index_col]) if index_col else str(idx)
                results.setdefault(key, {})[tname] = {
                    "value": round(float(y_hat[i]), 2)
                }

        else:  # classification
            le       = LabelEncoder()
            y_train  = le.fit_transform(target_series[train_mask].astype(str))
            model    = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_hat    = model.predict(X_pred)
            y_proba  = model.predict_proba(X_pred)
            y_labels = le.inverse_transform(y_hat)

            for i, idx in enumerate(df[pred_mask].index):
                key = str(df.loc[idx, index_col]) if index_col else str(idx)
                confidence = float(y_proba[i].max())
                results.setdefault(key, {})[tname] = {
                    "value": str(y_labels[i]),
                    "confidence": round(confidence, 4)
                }

    # -----------------------------------------------------------------------
    # Format output to match RPT-1 response shape
    # -----------------------------------------------------------------------
    predictions = [
        {"item_id": k, "predictions": v}
        for k, v in results.items()
    ]
    return {"predictions": predictions}


# ---------------------------------------------------------------------------
# CrewAI tool
# ---------------------------------------------------------------------------

@tool("call_rpt1")
def call_rpt1(payload: dict) -> str:
    """Call a local scikit-learn model to predict missing values in the payload.

    This is a drop-in replacement for the SAP-RPT-1 API tool.  It accepts the
    same payload format used in the CodeJam exercises (rows with [PREDICT]
    placeholders, a data_schema, and a prediction_config) and returns
    predictions in a compatible JSON shape.

    Args:
        payload: A dictionary containing the stolen items data with
                 prediction placeholders.  This should be the exact payload
                 provided in the task inputs.

    Returns:
        JSON string with predicted insurance values and item categories.
    """
    try:
        result = _predict_payload(payload)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error running local predictor: {str(e)}"


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python rpt1_sklearn_tool.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal payload matching the CodeJam payload.py structure
    test_payload = {
        "prediction_config": {
            "target_columns": [
                {"name": "INSURANCE_VALUE",  "prediction_placeholder": "[PREDICT]", "task_type": "regression"},
                {"name": "ITEM_CATEGORY",    "prediction_placeholder": "[PREDICT]", "task_type": "classification"},
            ]
        },
        "index_column": "ITEM_ID",
        "rows": [
            {"ITEM_ID": "ART_001", "ITEM_NAME": "Water Lilies",          "ARTIST": "Claude Monet",    "ACQUISITION_DATE": "1987-03-15", "INSURANCE_VALUE": 45000000, "ITEM_CATEGORY": "Painting",  "CONDITION_SCORE": 9, "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 8},
            {"ITEM_ID": "ART_002", "ITEM_NAME": "Japanese Bridge",       "ARTIST": "Claude Monet",    "ACQUISITION_DATE": "1995-06-22", "INSURANCE_VALUE": 42000000, "ITEM_CATEGORY": "Painting",  "CONDITION_SCORE": 8, "RARITY_SCORE": 8,  "PROVENANCE_CLARITY": 9},
            {"ITEM_ID": "ART_003", "ITEM_NAME": "Irises",                "ARTIST": "Van Gogh",        "ACQUISITION_DATE": "2001-11-08", "INSURANCE_VALUE": "[PREDICT]","ITEM_CATEGORY": "Painting", "CONDITION_SCORE": 7, "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 8},
            {"ITEM_ID": "ART_004", "ITEM_NAME": "Starry Night Over Rhone","ARTIST": "Van Gogh",       "ACQUISITION_DATE": "1998-09-14", "INSURANCE_VALUE": 48000000, "ITEM_CATEGORY": "Painting",  "CONDITION_SCORE": 8, "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 9},
            {"ITEM_ID": "ART_005", "ITEM_NAME": "Birth of Venus",        "ARTIST": "Botticelli",      "ACQUISITION_DATE": "1992-04-30", "INSURANCE_VALUE": 55000000, "ITEM_CATEGORY": "Painting",  "CONDITION_SCORE": 6, "RARITY_SCORE": 10, "PROVENANCE_CLARITY": 10},
            {"ITEM_ID": "ART_009", "ITEM_NAME": "Persistence of Memory", "ARTIST": "Salvador Dalí",   "ACQUISITION_DATE": "2005-03-10", "INSURANCE_VALUE": 35000000, "ITEM_CATEGORY": "[PREDICT]", "CONDITION_SCORE": 9, "RARITY_SCORE": 9,  "PROVENANCE_CLARITY": 10},
            {"ITEM_ID": "ART_011", "ITEM_NAME": "The Bronze Dancer",     "ARTIST": "Auguste Rodin",   "ACQUISITION_DATE": "1991-07-22", "INSURANCE_VALUE": 8500000,  "ITEM_CATEGORY": "Sculpture", "CONDITION_SCORE": 9, "RARITY_SCORE": 7,  "PROVENANCE_CLARITY": 8},
            {"ITEM_ID": "ART_013", "ITEM_NAME": "Hope Diamond Replica",  "ARTIST": "Unknown Jeweler", "ACQUISITION_DATE": "1988-02-19", "INSURANCE_VALUE": 12000000, "ITEM_CATEGORY": "Jewelry",   "CONDITION_SCORE": 10,"RARITY_SCORE": 10, "PROVENANCE_CLARITY": 7},
        ],
        "data_schema": {
            "ITEM_ID":             {"dtype": "string"},
            "ITEM_NAME":           {"dtype": "string"},
            "ARTIST":              {"dtype": "string"},
            "ACQUISITION_DATE":    {"dtype": "date"},
            "INSURANCE_VALUE":     {"dtype": "numeric"},
            "ITEM_CATEGORY":       {"dtype": "string", "categories": ["Painting", "Sculpture", "Jewelry"]},
            "CONDITION_SCORE":     {"dtype": "numeric"},
            "RARITY_SCORE":        {"dtype": "numeric"},
            "PROVENANCE_CLARITY":  {"dtype": "numeric"},
        },
    }

    output = _predict_payload(test_payload)
    print(json.dumps(output, indent=2))