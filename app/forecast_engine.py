# import pandas as pd
# import numpy as np
# import os, joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import load_model

# os.makedirs("models", exist_ok=True)

# # ---------- Loan Default Model ----------
# def train_default_model():
#     df = pd.read_csv("data/loans.csv")
#     X = df[["age","income","loan_amount","tenure_months","emi_paid","balance"]]
#     y = df["defaulted"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = XGBClassifier(
#         n_estimators=100, learning_rate=0.1, max_depth=5,
#         subsample=0.8, colsample_bytree=0.8, random_state=42
#     )
#     model.fit(X_train, y_train)
#     joblib.dump(model, "models/xgb_default.pkl")
#     print("✅ Loan Default model trained & saved.")
#     return model

# def predict_default(df):
#     model = joblib.load("models/xgb_default.pkl")
#     X = df[["age","income","loan_amount","tenure_months","emi_paid","balance"]]
#     df["default_probability"] = model.predict_proba(X)[:,1]
#     return df[["customer_id","default_probability"]]

# # ---------- Liquidity Forecast Model ----------
# # def train_liquidity_model():
# #     df = pd.read_csv("data/liquidity.csv")
# #     df["date"] = pd.to_datetime(df["date"])
# #     df = df.sort_values(["branch_id","date"])
# #     X, y = [], []
# #     seq_len = 7
# #     for branch in df.branch_id.unique():
# #         subset = df[df.branch_id==branch]["balance"].values
# #         for i in range(len(subset)-seq_len):
# #             X.append(subset[i:i+seq_len])
# #             y.append(subset[i+seq_len])
# #     X, y = np.array(X), np.array(y)
# #     X = np.expand_dims(X, axis=2)

# #     model = Sequential([
# #         LSTM(32, input_shape=(seq_len,1)),
# #         Dense(16, activation="relu"),
# #         Dense(1)
# #     ])
# #     model.compile(optimizer="adam", loss="mse")
# #     es = EarlyStopping(monitor="loss", patience=3)
# #     model.fit(X, y, epochs=10, batch_size=16, verbose=0, callbacks=[es])
# #     model.save("models/lstm_liquidity.h5")
# #     print("✅ Liquidity Forecast model trained & saved.")
# #     return model

# # def forecast_liquidity(branch_id):
# #     model = load_model("models/lstm_liquidity.h5")
# #     df = pd.read_csv("data/liquidity.csv")
# #     df["date"] = pd.to_datetime(df["date"])
# #     subset = df[df.branch_id==branch_id].sort_values("date")["balance"].values
# #     seq_len = 7
# #     last_seq = subset[-seq_len:]
# #     forecasts = []
# #     current = last_seq.copy()
# #     for _ in range(7):
# #         x_in = np.expand_dims(current.reshape((1,seq_len,1)), axis=0)
# #         pred = model.predict(x_in, verbose=0)[0][0]
# #         forecasts.append(pred)
# #         current = np.append(current[1:], pred)
# #     future_dates = pd.date_range(df.date.max()+pd.Timedelta(days=1), periods=7)
# #     return pd.DataFrame({"date":future_dates, "forecast_balance":forecasts})

# # ---------- Liquidity Forecast Model ----------
# def train_liquidity_model():
#     df = pd.read_csv("data/liquidity.csv")
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values(["branch_id", "date"])
    
#     X, y = [], []
#     seq_len = 7
#     for branch in df.branch_id.unique():
#         subset = df[df.branch_id == branch]["balance"].values
#         for i in range(len(subset) - seq_len):
#             X.append(subset[i:i+seq_len])
#             y.append(subset[i+seq_len])
#     X, y = np.array(X), np.array(y)
#     X = np.expand_dims(X, axis=2)

#     model = Sequential([
#         LSTM(32, input_shape=(seq_len, 1)),
#         Dense(16, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mse")
#     es = EarlyStopping(monitor="loss", patience=3)
#     model.fit(X, y, epochs=10, batch_size=16, verbose=0, callbacks=[es])

#     # ✅ Use modern Keras format instead of HDF5 (.h5)
#     # model.save("models/lstm_liquidity.keras")

#     model = load_model("models/lstm_liquidity.keras", compile=False)

#     print("✅ Liquidity Forecast model trained & saved.")
#     return model


# def forecast_liquidity(branch_id):
#     import tensorflow as tf

#     # ✅ Load safely in new Keras versions
#     try:
#         model = load_model("models/lstm_liquidity.keras", compile=False)
#     except Exception:
#         # fallback in case the old .h5 exists
#         model = load_model("models/lstm_liquidity.h5", compile=False)

#     df = pd.read_csv("data/liquidity.csv")
#     df["date"] = pd.to_datetime(df["date"])
#     subset = df[df.branch_id == branch_id].sort_values("date")["balance"].values
#     seq_len = 7
#     last_seq = subset[-seq_len:]
#     forecasts = []
#     current = last_seq.copy()

#     for _ in range(7):
#         x_in = np.expand_dims(current.reshape((1, seq_len, 1)), axis=0)
#         pred = model.predict(x_in, verbose=0)[0][0]
#         forecasts.append(pred)
#         current = np.append(current[1:], pred)

#     future_dates = pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=7)
#     return pd.DataFrame({"date": future_dates, "forecast_balance": forecasts})


# if __name__ == "__main__":
#     if not os.path.exists("models/xgb_default.pkl"):
#         train_default_model()
#     if not os.path.exists("models/lstm_liquidity.h5"):
#         train_liquidity_model()



# import pandas as pd
# import numpy as np
# import os, joblib
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# # Ensure model directory exists
# os.makedirs("models", exist_ok=True)

# # ==============================================================
# # ---------- Loan Default Model ----------
# # ==============================================================

# def train_default_model():
#     """Train and save the XGBoost model for loan default prediction."""
#     df = pd.read_csv("data/loans.csv")

#     X = df[["age", "income", "loan_amount", "tenure_months", "emi_paid", "balance"]]
#     y = df["defaulted"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = XGBClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )

#     model.fit(X_train, y_train)
#     joblib.dump(model, "models/xgb_default.pkl")
#     print("✅ Loan Default model trained & saved at models/xgb_default.pkl")
#     return model


# def predict_default(df):
#     """Predict default probabilities for given customers."""
#     model = joblib.load("models/xgb_default.pkl")
#     X = df[["age", "income", "loan_amount", "tenure_months", "emi_paid", "balance"]]
#     df["default_probability"] = model.predict_proba(X)[:, 1]
#     return df[["customer_id", "default_probability"]]

# # ==============================================================
# # ---------- Liquidity Forecast Model ----------
# # ==============================================================

# def train_liquidity_model():
#     """Train and save an LSTM model for liquidity forecasting."""
#     df = pd.read_csv("data/liquidity.csv")
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values(["branch_id", "date"])
    
#     X, y = [], []
#     seq_len = 7

#     # Create sliding windows of liquidity sequences
#     for branch in df.branch_id.unique():
#         subset = df[df.branch_id == branch]["balance"].values
#         for i in range(len(subset) - seq_len):
#             X.append(subset[i:i + seq_len])
#             y.append(subset[i + seq_len])

#     X, y = np.array(X), np.array(y)
#     X = np.expand_dims(X, axis=2)

#     model = Sequential([
#         LSTM(32, input_shape=(seq_len, 1)),
#         Dense(16, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mse")

#     es = EarlyStopping(monitor="loss", patience=3)
#     model.fit(X, y, epochs=10, batch_size=16, verbose=0, callbacks=[es])

#     # ✅ Save model in modern Keras format
#     model.save("models/lstm_liquidity.keras")

#     print("✅ Liquidity Forecast model trained & saved at models/lstm_liquidity.keras")
#     return model


# def forecast_liquidity(branch_id):
#     """Forecast the next 7 days of liquidity for a given branch."""
#     # Try loading modern Keras format first, fallback to legacy if needed
#     try:
#         model = load_model("models/lstm_liquidity.keras", compile=False)
#     except Exception:
#         model = load_model("models/lstm_liquidity.keras", compile=False)

#     df = pd.read_csv("data/liquidity.csv")
#     df["date"] = pd.to_datetime(df["date"])

#     subset = df[df.branch_id == branch_id].sort_values("date")["balance"].values
#     seq_len = 7
#     last_seq = subset[-seq_len:]
#     forecasts = []
#     current = last_seq.copy()

#     # Predict next 7 days iteratively
#     for _ in range(7):
#         x_in = current.reshape((1, seq_len, 1))
#         pred = model.predict(x_in, verbose=0)[0][0]
#         forecasts.append(pred)
#         current = np.append(current[1:], pred)

#     future_dates = pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=7)
#     forecast_df = pd.DataFrame({
#         "date": future_dates,
#         "forecast_balance": forecasts
#     })
#     return forecast_df


# # ==============================================================
# # ---------- Main Entry Point ----------
# # ==============================================================

# if __name__ == "__main__":
#     # Train models if missing
#     if not os.path.exists("models/xgb_default.pkl"):
#         train_default_model()
#     else:
#         print("✅ Loan Default model already exists.")

#     if not os.path.exists("models/lstm_liquidity.keras"):
#         train_liquidity_model()
#     else:
#         print("✅ Liquidity Forecast model already exists.")








# import pandas as pd
# import numpy as np
# import os
# import joblib
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# # ------------------ Setup ------------------
# MODEL_DIR = "models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# XGB_PATH = os.path.join(MODEL_DIR, "xgb_default.pkl")
# LSTM_PATH = os.path.join(MODEL_DIR, "lstm_liquidity.keras")

# # ------------------ Loan Default Model ------------------
# def train_default_model():
#     """Train and save the XGBoost model for loan default prediction."""
#     print("[DEBUG] Training Loan Default model...")
#     df = pd.read_csv("data/loans.csv")
#     X = df[["age", "income", "loan_amount", "tenure_months", "emi_paid", "balance"]]
#     y = df["defaulted"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = XGBClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )

#     model.fit(X_train, y_train)
#     joblib.dump(model, XGB_PATH)
#     print(f"✅ Loan Default model trained & saved at {XGB_PATH}")
#     return model

# def predict_default(df):
#     """Predict default probabilities for given customers."""
#     if not os.path.exists(XGB_PATH):
#         print("[DEBUG] XGB model missing. Training now...")
#         train_default_model()
#     print("[DEBUG] Loading Loan Default model...")
#     model = joblib.load(XGB_PATH)
#     X = df[["age", "income", "loan_amount", "tenure_months", "emi_paid", "balance"]]
#     df["default_probability"] = model.predict_proba(X)[:, 1]
#     return df[["customer_id", "default_probability"]]

# # ------------------ Liquidity Forecast Model ------------------
# def train_liquidity_model():
#     """Train and save an LSTM model for liquidity forecasting."""
#     print("[DEBUG] Training Liquidity Forecast model...")
#     df = pd.read_csv("data/liquidity.csv")
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values(["branch_id", "date"])
    
#     X, y = [], []
#     seq_len = 7

#     for branch in df.branch_id.unique():
#         subset = df[df.branch_id == branch]["balance"].values
#         for i in range(len(subset) - seq_len):
#             X.append(subset[i:i + seq_len])
#             y.append(subset[i + seq_len])

#     X, y = np.array(X), np.array(y)
#     X = np.expand_dims(X, axis=2)

#     model = Sequential([
#         LSTM(32, input_shape=(seq_len, 1)),
#         Dense(16, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer="adam", loss="mse")

#     es = EarlyStopping(monitor="loss", patience=3)
#     model.fit(X, y, epochs=10, batch_size=16, verbose=1, callbacks=[es])

#     model.save(LSTM_PATH)
#     print(f"✅ Liquidity Forecast model trained & saved at {LSTM_PATH}")
#     return model

# def forecast_liquidity(branch_id):
#     """Forecast the next 7 days of liquidity for a given branch."""
#     # Load model with debug
#     if not os.path.exists(LSTM_PATH):
#         print("[DEBUG] LSTM model missing. Training now...")
#         train_liquidity_model()

#     print("[DEBUG] Loading Liquidity Forecast model...")
#     try:
#         model = load_model(LSTM_PATH, compile=False)
#     except Exception as e:
#         print(f"[ERROR] Failed to load LSTM model: {e}")
#         print("[DEBUG] Retraining LSTM model...")
#         model = train_liquidity_model()

#     df = pd.read_csv("data/liquidity.csv")
#     df["date"] = pd.to_datetime(df["date"])
#     subset = df[df.branch_id == branch_id].sort_values("date")["balance"].values

#     if len(subset) < 7:
#         raise ValueError(f"[ERROR] Not enough data to forecast for branch {branch_id}")

#     seq_len = 7
#     last_seq = subset[-seq_len:]
#     forecasts = []
#     current = last_seq.copy()

#     for _ in range(7):
#         x_in = current.reshape((1, seq_len, 1))
#         pred = model.predict(x_in, verbose=0)[0][0]
#         forecasts.append(pred)
#         current = np.append(current[1:], pred)

#     future_dates = pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=7)
#     forecast_df = pd.DataFrame({
#         "date": future_dates,
#         "forecast_balance": forecasts
#     })
#     print("[DEBUG] Forecasting complete.")
#     return forecast_df

# # ------------------ Main Entry Point ------------------
# if __name__ == "__main__":
#     # Train Loan Default model if missing
#     if not os.path.exists(XGB_PATH):
#         train_default_model()
#     else:
#         print("✅ Loan Default model already exists.")

#     # Train Liquidity Forecast model if missing
#     if not os.path.exists(LSTM_PATH):
#         train_liquidity_model()
#     else:
#         print("✅ Liquidity Forecast model already exists.")

#     # Example debug run
#     print("[DEBUG] Example forecast for branch 1:")
#     forecast_df = forecast_liquidity(branch_id=1)
#     print(forecast_df)
# #########################################################
import os
import json
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

os.makedirs("models", exist_ok=True)

logging.basicConfig(
    filename=os.path.join("models", "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "tenure_months",
    "emi_paid",
    "balance",
]

# -------------------------------------------------------------------
# Loan Default Model (XGBoost)
# -------------------------------------------------------------------


def train_default_model():
    """
    Train XGBoost model for loan default prediction
    with safe evaluation & realistic metrics.
    """
    try:
        df = pd.read_csv("data/loans.csv")

        X = df[DEFAULT_FEATURES]
        y = df["defaulted"]

        # ✅ Proper stratified split (VERY IMPORTANT)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        base_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            n_jobs=-1,
        )

        param_dist = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=10,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )

        logger.info("Starting RandomizedSearchCV for XGBoost default model...")
        search.fit(X_train, y_train)

        model = search.best_estimator_

        # ---------------- Evaluation ----------------
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # ✅ Safe ROC-AUC calculation
        if len(np.unique(y_test)) > 1:
            roc = float(roc_auc_score(y_test, y_pred_proba))
        else:
            roc = None
            logger.warning("ROC-AUC not computed: only one class in y_test")

        cls_report = classification_report(
            y_test, y_pred, output_dict=True
        )

        metrics = {
            "roc_auc": roc,
            "classification_report": cls_report,
            "best_params": search.best_params_,
            "test_size": len(y_test),
            "class_distribution": y_test.value_counts().to_dict(),
        }

        with open("models/xgb_default_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        joblib.dump(model, "models/xgb_default.pkl")

        logger.info("Loan Default model trained & saved.")
        print("✅ Loan Default model trained & saved.")

        return model

    except Exception as e:
        logger.exception(f"Error training default model: {e}")
        raise


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def _ensure_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_id" not in df.columns:
        df = df.copy()
        df["customer_id"] = np.arange(1, len(df) + 1)
    return df


def _risk_bucket(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


# -------------------------------------------------------------------
# Default Prediction & Explainability
# -------------------------------------------------------------------

def predict_default(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load("models/xgb_default.pkl")
    df = _ensure_customer_id(df)

    X = df[DEFAULT_FEATURES]
    proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["default_probability"] = proba
    df["risk_level"] = df["default_probability"].apply(_risk_bucket)

    return df[["customer_id", "default_probability", "risk_level"]]


def explain_top_risks(df: pd.DataFrame, top_n: int = 5) -> str:
    high_risk = df.sort_values("default_probability", ascending=False).head(top_n)

    reasons = []
    for _, row in high_risk.iterrows():
        reason = f"Customer {row['customer_id']} ({_risk_bucket(row['default_probability'])} risk): "

        if row["balance"] < row["emi_paid"]:
            reason += "Low balance compared to EMI. "
        if row["loan_amount"] > 2 * row["income"]:
            reason += "High loan amount relative to income. "
        if row["tenure_months"] > 60:
            reason += "Very long loan tenure. "

        reasons.append(reason.strip())

    return "\n".join(reasons)


def get_default_feature_importance(top_n: int = 6):
    model = joblib.load("models/xgb_default.pkl")
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")

    mapping = {f"f{i}": feat for i, feat in enumerate(DEFAULT_FEATURES)}
    named = {mapping.get(k, k): float(v) for k, v in raw.items()}

    return sorted(named.items(), key=lambda x: x[1], reverse=True)[:top_n]


# -------------------------------------------------------------------
# Liquidity Forecast Model (LSTM)
# -------------------------------------------------------------------

def train_liquidity_model(seq_len: int = 7):
    try:
        df = pd.read_csv("data/liquidity.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["branch_id", "date"])

        scaler = MinMaxScaler()
        df["balance_scaled"] = scaler.fit_transform(df[["balance"]])

        X, y = [], []
        for branch in df.branch_id.unique():
            series = df[df.branch_id == branch]["balance_scaled"].values
            if len(series) <= seq_len:
                continue
            for i in range(len(series) - seq_len):
                X.append(series[i:i + seq_len])
                y.append(series[i + seq_len])

        X = np.array(X)
        y = np.array(y)
        X = np.expand_dims(X, axis=2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = Sequential([
            LSTM(32, input_shape=(seq_len, 1)),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=16,
            verbose=0,
            callbacks=[es],
        )

        y_pred = model.predict(X_test, verbose=0).flatten()

        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))

        metrics = {"mse": mse, "mae": mae, "rmse": rmse}

        with open("models/lstm_liquidity_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        model.save("models/lstm_liquidity.keras")
        joblib.dump(scaler, "models/liquidity_scaler.pkl")

        print("✅ Liquidity Forecast model trained & saved.")
        return model

    except Exception as e:
        logger.exception(f"Error training liquidity model: {e}")
        raise


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists("models/xgb_default.pkl"):
        train_default_model()

    if not (
        os.path.exists("models/lstm_liquidity.keras")
        and os.path.exists("models/liquidity_scaler.pkl")
    ):
        train_liquidity_model()
def forecast_liquidity(branch_id, seq_len: int = 7) -> pd.DataFrame:
    """
    Forecast next 7 days of liquidity for a branch.
    Returns DataFrame with columns: date, forecast_balance
    """
    try:
        model = load_model("models/lstm_liquidity.keras")
        scaler: MinMaxScaler = joblib.load("models/liquidity_scaler.pkl")

        df = pd.read_csv("data/liquidity.csv")
        df["date"] = pd.to_datetime(df["date"])

        if branch_id not in df.branch_id.unique():
            future_dates = pd.date_range(start=pd.to_datetime("today"), periods=7)
            return pd.DataFrame({
                "date": future_dates,
                "forecast_balance": [0.0] * 7
            })

        subset = (
            df[df.branch_id == branch_id]
            .sort_values("date")["balance"]
            .values
        )

        if len(subset) < seq_len:
            future_dates = pd.date_range(start=pd.to_datetime("today"), periods=7)
            return pd.DataFrame({
                "date": future_dates,
                "forecast_balance": [0.0] * 7
            })

        subset_scaled = scaler.transform(subset.reshape(-1, 1)).flatten()
        current = subset_scaled[-seq_len:]

        forecasts = []
        for _ in range(7):
            x = current.reshape((1, seq_len, 1))
            pred_scaled = model.predict(x, verbose=0)[0][0]
            pred = scaler.inverse_transform([[pred_scaled]])[0][0]
            forecasts.append(float(pred))
            current = np.append(current[1:], pred_scaled)

        future_dates = pd.date_range(
            df["date"].max() + pd.Timedelta(days=1),
            periods=7
        )

        return pd.DataFrame({
            "date": future_dates,
            "forecast_balance": forecasts
        })

    except Exception as e:
        logger.exception(f"Error in forecast_liquidity: {e}")
        future_dates = pd.date_range(start=pd.to_datetime("today"), periods=7)
        return pd.DataFrame({
            "date": future_dates,
            "forecast_balance": [0.0] * 7
        })
