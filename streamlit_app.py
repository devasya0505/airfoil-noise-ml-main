import os
import io
import json
import time
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Lazy imports for ML libs to reduce cold-start time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

try:
    from xgboost import XGBRegressor  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


DATA_PATH = os.path.join("data", "airfoil_self_noise.dat")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


FEATURE_COLUMNS = [
    "Frequency (Hz)",
    "Angle of attack (deg)",
    "Chord length (m)",
    "Free-stream velocity (m/s)",
    "Suction side displacement thickness (m)",
]
TARGET_COLUMN = "Sound pressure level (dB)"


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df


def model_artifact_paths(name: str) -> Dict[str, str]:
    safe = name.lower().replace(" ", "_")
    return {
        "model": os.path.join(MODEL_DIR, f"{safe}.bin"),  # sklearn via joblib; xgb via native json
        "scaler": os.path.join(MODEL_DIR, f"{safe}_scaler.bin"),
        "meta": os.path.join(MODEL_DIR, f"{safe}.meta.json"),
        "xgb_json": os.path.join(MODEL_DIR, f"{safe}.json"),
    }


def save_model_artifacts(name: str, model_obj: Dict):
    paths = model_artifact_paths(name)
    meta = {
        "name": name,
        "needs_scaling": bool(model_obj.get("needs_scaling", False)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "r2": float(model_obj.get("r2", np.nan)),
        "rmse": float(model_obj.get("rmse", np.nan)),
    }

    if name == "XGBoost" and XGB_AVAILABLE:
        # Use native XGBoost serialization
        model_obj["model"].save_model(paths["xgb_json"])  # type: ignore[attr-defined]
    else:
        joblib.dump(model_obj["model"], paths["model"])  # sklearn models

    if model_obj.get("scaler") is not None:
        joblib.dump(model_obj["scaler"], paths["scaler"]) 

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_saved_model(name: str) -> Optional[Dict]:
    paths = model_artifact_paths(name)
    if not os.path.exists(paths["meta"]):
        return None
    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        needs_scaling = bool(meta.get("needs_scaling", False))

        if name == "XGBoost" and XGB_AVAILABLE and os.path.exists(paths["xgb_json"]):
            xgb = XGBRegressor()
            xgb.load_model(paths["xgb_json"])  # type: ignore[attr-defined]
            model = xgb
            scaler = None
        else:
            if not os.path.exists(paths["model"]):
                return None
            model = joblib.load(paths["model"]) 
            scaler = joblib.load(paths["scaler"]) if os.path.exists(paths["scaler"]) else None

        return {
            "model": model,
            "scaler": scaler,
            "needs_scaling": needs_scaling,
            "r2": float(meta.get("r2", np.nan)),
            "rmse": float(meta.get("rmse", np.nan)),
            # Placeholders; will be filled after evaluation if needed
            "X_test": None,
            "y_test": None,
            "y_pred": None,
            "feature_importances_": getattr(model, "feature_importances_", None),
        }
    except Exception:
        return None


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: Dict[str, Dict] = {}

    # Linear
    linear = LinearRegression()
    linear.fit(X_train_scaled, y_train)
    y_pred = linear.predict(X_test_scaled)
    results["Linear Regression"] = {
        "model": linear,
        "scaler": scaler,
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importances_": None,
        "needs_scaling": True,
    }

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_ridge = ridge.predict(X_test_scaled)
    results["Ridge Regression"] = {
        "model": ridge,
        "scaler": scaler,
        "r2": r2_score(y_test, y_ridge),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_ridge))),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_ridge,
        "feature_importances_": None,
        "needs_scaling": True,
    }

    # Random Forest (no scaling)
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "model": rf,
        "scaler": None,
        "r2": r2_score(y_test, y_rf),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_rf))),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_rf,
        "feature_importances_": getattr(rf, "feature_importances_", None),
        "needs_scaling": False,
    }

    # XGBoost (optional)
    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        y_xgb = xgb.predict(X_test)
        results["XGBoost"] = {
            "model": xgb,
            "scaler": None,
            "r2": r2_score(y_test, y_xgb),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_xgb))),
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_xgb,
            "feature_importances_": getattr(xgb, "feature_importances_", None),
            "needs_scaling": False,
        }

    return results


def predict_dataframe(model_obj: Dict, df: pd.DataFrame) -> np.ndarray:
    model = model_obj["model"]
    scaler = model_obj.get("scaler")
    needs_scaling = model_obj.get("needs_scaling", False)
    X = df[FEATURE_COLUMNS]
    if needs_scaling and scaler is not None:
        X = scaler.transform(X)
    return model.predict(X)


def render_single_input_form() -> pd.DataFrame:
    cols = st.columns(2)
    with cols[0]:
        frequency = st.number_input("Frequency (Hz)", min_value=0.0, value=1000.0, step=100.0)
        chord_length = st.number_input("Chord length (m)", min_value=0.0, value=0.3048, step=0.01)
        thickness = st.number_input(
            "Suction side displacement thickness (m)", min_value=0.0, value=0.002, step=0.0001
        )
    with cols[1]:
        angle = st.number_input("Angle of attack (deg)", min_value=-10.0, value=5.0, step=0.5)
        velocity = st.number_input("Free-stream velocity (m/s)", min_value=0.0, value=50.0, step=1.0)

    single_df = pd.DataFrame(
        {
            "Frequency (Hz)": [frequency],
            "Angle of attack (deg)": [angle],
            "Chord length (m)": [chord_length],
            "Free-stream velocity (m/s)": [velocity],
            "Suction side displacement thickness (m)": [thickness],
        }
    )
    return single_df


def render_batch_uploader() -> Optional[pd.DataFrame]:
    st.write("Upload a CSV with columns exactly matching feature names below:")
    st.code("\n".join(FEATURE_COLUMNS))
    uploaded = st.file_uploader("Choose CSV", type=["csv"]) 
    if uploaded is None:
        return None
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
        return df
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return None


def show_feature_importances(model_obj: Dict):
    importances = model_obj.get("feature_importances_")
    if importances is None:
        st.info("Feature importance not available for this model.")
        return
    imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    st.bar_chart(imp_df.set_index("feature"))


def show_residuals(y_true: pd.Series, y_pred: np.ndarray):
    resid = y_true.values - y_pred
    resid_df = pd.DataFrame({"Residual": resid})
    st.line_chart(resid_df)


def main():
    st.set_page_config(page_title="Airfoil Noise ML", layout="wide")
    st.title("Airfoil Self-Noise Prediction")
    st.caption("Predict sound pressure level (dB) from flow and geometry conditions.")

    # Sidebar: model selection and training
    st.sidebar.header("Modeling")
    df = load_dataset()

    with st.sidebar.expander("Dataset", expanded=False):
        st.write(df.head())
        st.write(df.describe())

    use_saved = st.sidebar.toggle("Use saved models if available", value=True)

    # Try load saved; otherwise train
    results: Dict[str, Dict] = {}
    loaded_any = False
    if use_saved:
        for candidate in ["Linear Regression", "Ridge Regression", "Random Forest", "XGBoost"]:
            loaded = load_saved_model(candidate)
            if loaded is not None:
                results[candidate] = loaded
                loaded_any = True

    if not results:
        with st.spinner("Training models..."):
            results = train_models(df)

    model_names = list(results.keys())
    default_idx = model_names.index("XGBoost") if "XGBoost" in results else model_names.index("Random Forest")
    selected_name = st.sidebar.selectbox("Select model", model_names, index=default_idx)
    model_obj = results[selected_name]

    if model_obj.get("r2") is not None and not np.isnan(model_obj.get("r2", np.nan)):
        st.sidebar.metric("R²", f"{model_obj['r2']:.4f}")
    if model_obj.get("rmse") is not None and not np.isnan(model_obj.get("rmse", np.nan)):
        st.sidebar.metric("RMSE", f"{model_obj['rmse']:.3f}")

    if st.sidebar.button("Save selected model"):
        # If the metrics are missing (loaded-only path), evaluate quickly on a holdout to store metrics
        if model_obj.get("X_test") is None:
            X = df[FEATURE_COLUMNS]
            y = df[TARGET_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = predict_dataframe(model_obj, X_test)
            model_obj["X_test"], model_obj["y_test"], model_obj["y_pred"] = X_test, y_test, y_pred
            model_obj["r2"] = r2_score(y_test, y_pred)
            model_obj["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        save_model_artifacts(selected_name, model_obj)
        st.sidebar.success("Model saved to 'models/' directory.")

    tab_single, tab_batch, tab_analysis = st.tabs(["Single Prediction", "Batch Prediction", "Analysis"])

    with tab_single:
        input_df = render_single_input_form()
        if st.button("Predict", type="primary"):
            pred = predict_dataframe(model_obj, input_df)
            st.success(f"Predicted Sound Pressure Level: {pred[0]:.2f} dB")

    with tab_batch:
        batch_df = render_batch_uploader()
        if batch_df is not None:
            preds = predict_dataframe(model_obj, batch_df)
            out = batch_df.copy()
            out[TARGET_COLUMN] = preds
            st.dataframe(out)

            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv_buf.getvalue(),
                file_name="airfoil_noise_predictions.csv",
                mime="text/csv",
            )

    with tab_analysis:
        st.subheader("Predicted vs Actual (Test Set)")
        if model_obj.get("y_test") is not None and model_obj.get("y_pred") is not None:
            compare_df = pd.DataFrame(
                {"Actual": model_obj["y_test"].values, "Predicted": model_obj["y_pred"]}
            )
            st.scatter_chart(compare_df)
        else:
            st.info("Evaluate the model (e.g., by saving it) to populate test-set comparisons.")

        st.subheader("Feature Importance")
        show_feature_importances(model_obj)

        st.subheader("Residuals (Test Set)")
        if model_obj.get("y_test") is not None and model_obj.get("y_pred") is not None:
            show_residuals(model_obj["y_test"], model_obj["y_pred"])
        else:
            st.info("Residuals will appear after evaluation.")

        st.subheader("Metrics")
        metrics_payload = {}
        if model_obj.get("r2") is not None and not np.isnan(model_obj.get("r2", np.nan)):
            metrics_payload["r2"] = round(float(model_obj["r2"]), 4)
        if model_obj.get("rmse") is not None and not np.isnan(model_obj.get("rmse", np.nan)):
            metrics_payload["rmse"] = round(float(model_obj["rmse"]), 4)
        if model_obj.get("y_test") is not None:
            metrics_payload["test_samples"] = int(len(model_obj["y_test"]))
        if metrics_payload:
            st.json(metrics_payload)
        else:
            st.info("Metrics will appear after evaluation.")


if __name__ == "__main__":
    main()


