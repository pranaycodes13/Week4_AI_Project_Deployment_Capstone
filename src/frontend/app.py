"""Streamlit full-stack dashboard for Wine Classification & Prediction service."""

import io

import pandas as pd
import streamlit as st

from src.config import WINE_FEATURE_NAMES
from src.frontend.api_client import (
    ApiClientError,
    ApiConnectionError,
    WineApiClient,
    resolve_api_base_url,
)

# Page configuration
st.set_page_config(
    page_title="Wine Classification & Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Reference sample presets derived from dataset partitions
PRESET_SAMPLES: dict[str, dict[str, float]] = {
    "Cultivar 0 (High Proline/Flavanoids)": {
        "alcohol": 13.74,
        "malic_acid": 1.67,
        "ash": 2.25,
        "alcalinity_of_ash": 16.4,
        "magnesium": 118.0,
        "total_phenols": 2.60,
        "flavanoids": 2.90,
        "nonflavanoid_phenols": 0.21,
        "proanthocyanins": 1.62,
        "color_intensity": 5.85,
        "hue": 0.92,
        "od280/od315_of_diluted_wines": 3.20,
        "proline": 1060.0,
    },
    "Cultivar 1 (Low Alcohol/Color)": {
        "alcohol": 12.37,
        "malic_acid": 0.94,
        "ash": 1.36,
        "alcalinity_of_ash": 10.6,
        "magnesium": 88.0,
        "total_phenols": 1.98,
        "flavanoids": 0.57,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 0.42,
        "color_intensity": 1.95,
        "hue": 1.05,
        "od280/od315_of_diluted_wines": 1.82,
        "proline": 520.0,
    },
    "Cultivar 2 (High Malic Acid/Low Flavanoids)": {
        "alcohol": 13.11,
        "malic_acid": 1.90,
        "ash": 2.17,
        "alcalinity_of_ash": 21.0,
        "magnesium": 80.0,
        "total_phenols": 1.65,
        "flavanoids": 0.68,
        "nonflavanoid_phenols": 0.53,
        "proanthocyanins": 0.62,
        "color_intensity": 7.10,
        "hue": 0.64,
        "od280/od315_of_diluted_wines": 1.56,
        "proline": 515.0,
    },
}

FEATURE_LABELS: dict[str, str] = {
    "alcohol": "Alcohol (% vol)",
    "malic_acid": "Malic Acid (g/L)",
    "ash": "Ash Content (g/L)",
    "alcalinity_of_ash": "Alcalinity of Ash",
    "magnesium": "Magnesium (mg/L)",
    "total_phenols": "Total Phenols",
    "flavanoids": "Flavanoids",
    "nonflavanoid_phenols": "Nonflavanoid Phenols",
    "proanthocyanins": "Proanthocyanins",
    "color_intensity": "Color Intensity",
    "hue": "Hue",
    "od280/od315_of_diluted_wines": "OD280/OD315 of Diluted Wines",
    "proline": "Proline (mg/L)",
}


def resolve_default_api_url() -> str:
    """Resolve API base URL respecting precedence: session_state -> API_BASE_URL -> API_URL -> local default."""
    return resolve_api_base_url(st.session_state.get("api_url"))


def get_api_client(base_url: str) -> WineApiClient:
    """Retrieve or initialize persistent API client in session state."""
    clean_url = base_url.strip().rstrip("/")
    if "api_client" not in st.session_state or st.session_state.get("cached_url") != clean_url:
        st.session_state.api_client = WineApiClient(base_url=clean_url)
        st.session_state.cached_url = clean_url
    return st.session_state.api_client


# -----------------------------------------------------------------------------
# SIDEBAR: Service Health & Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🍷 Wine ML Service")
    st.markdown("Production ML Inference Dashboard")
    st.divider()

    st.subheader("⚙️ API Configuration")
    initial_api_url = resolve_default_api_url()
    api_url_input = st.text_input(
        "FastAPI Service URL",
        value=initial_api_url,
        help="Base address of the running FastAPI prediction backend.",
    )
    api_url = api_url_input.strip().rstrip("/")
    st.session_state.api_url = api_url

    client = get_api_client(api_url)

    # Health check probe
    refresh_health = st.button("🔄 Probe Backend Health", use_container_width=True)

    if "health_status" not in st.session_state or refresh_health:
        try:
            health_data = client.health()
            st.session_state.health_status = health_data
            st.session_state.health_error = None
        except ApiConnectionError as exc:
            st.session_state.health_status = None
            st.session_state.health_error = str(exc)
        except ApiClientError as exc:
            st.session_state.health_status = None
            st.session_state.health_error = exc.message

    if st.session_state.get("health_status"):
        h = st.session_state.health_status
        st.success("🟢 **Backend Connected**")
        st.caption(f"Status: `{h.get('status')}` | Model Ready: `{h.get('model_ready')}`")
        if h.get("model_name"):
            st.caption(f"Model: *{h.get('model_name')}*")
    else:
        st.error("🔴 **Backend Offline**")
        if st.session_state.get("health_error"):
            st.caption(st.session_state.health_error)
        st.info("Tip: Start backend with `python run.py serve`")

    st.divider()
    st.subheader("🧭 Navigation")
    view_mode = st.radio(
        "Select Module:",
        options=["Single Prediction", "Batch Prediction (CSV)", "Model Architecture & Metrics"],
        index=0,
    )

    st.divider()
    st.caption("Week 4 AI Project Deployment Capstone\nScikit-learn StandardScaler + SVC Pipeline")


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.title("🍇 Italian Wine Cultivar Classification")
st.markdown(
    "Predict the cultivar origin of Italian wines based on 13 chemical constituents "
    "using an optimized, non-linear **Support Vector Classifier (SVC)**."
)
st.write("")

# -----------------------------------------------------------------------------
# MODULE 1: SINGLE PREDICTION
# -----------------------------------------------------------------------------
if view_mode == "Single Prediction":
    st.header("🧪 Single Wine Sample Inference")
    st.write("Input the chemical analysis values below or select a reference preset to populate the form.")

    # Preset loader
    col_preset_1, col_preset_2 = st.columns([2, 1])
    with col_preset_1:
        selected_preset = st.selectbox("Load Sample Archetype:", options=list(PRESET_SAMPLES.keys()))
    with col_preset_2:
        st.write("")
        st.write("")
        if st.button("Apply Preset", use_container_width=True):
            for k, val in PRESET_SAMPLES[selected_preset].items():
                st.session_state[f"input_{k}"] = val

    st.write("")
    with st.form("single_prediction_form"):
        st.subheader("Chemical Constituents")

        col1, col2, col3 = st.columns(3)

        inputs: dict[str, float] = {}

        # Column 1
        with col1:
            inputs["alcohol"] = st.number_input(
                FEATURE_LABELS["alcohol"],
                value=float(st.session_state.get("input_alcohol", 13.0)),
                step=0.05,
                format="%.2f",
                key="field_alcohol",
            )
            inputs["malic_acid"] = st.number_input(
                FEATURE_LABELS["malic_acid"],
                value=float(st.session_state.get("input_malic_acid", 2.0)),
                step=0.05,
                format="%.2f",
                key="field_malic_acid",
            )
            inputs["ash"] = st.number_input(
                FEATURE_LABELS["ash"],
                value=float(st.session_state.get("input_ash", 2.3)),
                step=0.05,
                format="%.2f",
                key="field_ash",
            )
            inputs["alcalinity_of_ash"] = st.number_input(
                FEATURE_LABELS["alcalinity_of_ash"],
                value=float(st.session_state.get("input_alcalinity_of_ash", 19.0)),
                step=0.5,
                format="%.1f",
                key="field_alcalinity_of_ash",
            )

        # Column 2
        with col2:
            inputs["magnesium"] = st.number_input(
                FEATURE_LABELS["magnesium"],
                value=float(st.session_state.get("input_magnesium", 99.0)),
                step=1.0,
                format="%.1f",
                key="field_magnesium",
            )
            inputs["total_phenols"] = st.number_input(
                FEATURE_LABELS["total_phenols"],
                value=float(st.session_state.get("input_total_phenols", 2.3)),
                step=0.05,
                format="%.2f",
                key="field_total_phenols",
            )
            inputs["flavanoids"] = st.number_input(
                FEATURE_LABELS["flavanoids"],
                value=float(st.session_state.get("input_flavanoids", 2.0)),
                step=0.05,
                format="%.2f",
                key="field_flavanoids",
            )
            inputs["nonflavanoid_phenols"] = st.number_input(
                FEATURE_LABELS["nonflavanoid_phenols"],
                value=float(st.session_state.get("input_nonflavanoid_phenols", 0.35)),
                step=0.02,
                format="%.2f",
                key="field_nonflavanoid_phenols",
            )

        # Column 3
        with col3:
            inputs["proanthocyanins"] = st.number_input(
                FEATURE_LABELS["proanthocyanins"],
                value=float(st.session_state.get("input_proanthocyanins", 1.5)),
                step=0.05,
                format="%.2f",
                key="field_proanthocyanins",
            )
            inputs["color_intensity"] = st.number_input(
                FEATURE_LABELS["color_intensity"],
                value=float(st.session_state.get("input_color_intensity", 5.0)),
                step=0.1,
                format="%.2f",
                key="field_color_intensity",
            )
            inputs["hue"] = st.number_input(
                FEATURE_LABELS["hue"],
                value=float(st.session_state.get("input_hue", 0.95)),
                step=0.02,
                format="%.2f",
                key="field_hue",
            )
            inputs["od280/od315_of_diluted_wines"] = st.number_input(
                FEATURE_LABELS["od280/od315_of_diluted_wines"],
                value=float(st.session_state.get("input_od280/od315_of_diluted_wines", 2.6)),
                step=0.05,
                format="%.2f",
                key="field_od280",
            )
            inputs["proline"] = st.number_input(
                FEATURE_LABELS["proline"],
                value=float(st.session_state.get("input_proline", 750.0)),
                step=10.0,
                format="%.1f",
                key="field_proline",
            )

        submit_single = st.form_submit_button("🍇 Classify Wine Sample", use_container_width=True)

    if submit_single:
        with st.spinner("Submitting sample to FastAPI prediction service..."):
            try:
                result = client.predict(inputs)
                st.success("✅ Prediction Successfully Received")

                pred_class = result.get("predicted_class")
                pred_name = result.get("predicted_class_name", f"class_{pred_class}")
                probs = result.get("probabilities", {})

                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric(
                        label="Predicted Cultivar",
                        value=f"Cultivar {pred_class}",
                        delta=f"Label: {pred_name}",
                    )
                    st.caption(f"Model: {result.get('model_name', 'SVC')}")
                    if result.get("model_version"):
                        st.caption(f"Version: {result.get('model_version')}")

                with res_col2:
                    st.subheader("Estimated Class Probabilities")
                    prob_df = pd.DataFrame(
                        {
                            "Cultivar": list(probs.keys()),
                            "Estimated Probability (%)": [p * 100 for p in probs.values()],
                        }
                    )
                    st.bar_chart(prob_df.set_index("Cultivar"))

                    # Formatted percentage table
                    st.dataframe(
                        prob_df.style.format({"Estimated Probability (%)": "{:.2f}%"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Note: These represent model-calculated class probability estimates produced by "
                        "scikit-learn SVC(probability=True). These values are model estimates and are "
                        "not guaranteed to be empirically calibrated."
                    )

            except ApiClientError as exc:
                st.error(f"❌ Prediction Failed: {exc.message}")
                if exc.detail:
                    st.caption(f"Details: {exc.detail}")


# -----------------------------------------------------------------------------
# MODULE 2: BATCH PREDICTION
# -----------------------------------------------------------------------------
elif view_mode == "Batch Prediction (CSV)":
    st.header("📁 Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV file containing 1 to 100 samples with all 13 chemical constituents.")

    # Downloadable CSV template
    sample_df = pd.DataFrame(list(PRESET_SAMPLES.values()))
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    sample_csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=sample_csv_bytes,
        file_name="wine_batch_sample_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.subheader(f"Uploaded Data Preview ({len(df_upload)} records)")
            st.dataframe(df_upload.head(10), use_container_width=True)

            # Pre-flight schema validation
            missing_cols = [c for c in WINE_FEATURE_NAMES if c not in df_upload.columns]
            extra_cols = [c for c in df_upload.columns if c not in WINE_FEATURE_NAMES]

            validation_errors: list[str] = []
            if missing_cols:
                validation_errors.append(f"Missing required columns: {missing_cols}")
            if extra_cols:
                validation_errors.append(f"Unexpected extra columns found: {extra_cols}")
            if len(df_upload) == 0:
                validation_errors.append("Uploaded CSV contains 0 rows. Minimum 1 sample required.")
            if len(df_upload) > 100:
                validation_errors.append(
                    f"Uploaded CSV contains {len(df_upload)} rows. Maximum allowed batch size is 100."
                )

            # Check numeric types and NaNs
            if not missing_cols and not extra_cols and len(df_upload) > 0:
                for col in WINE_FEATURE_NAMES:
                    if not pd.to_numeric(df_upload[col], errors="coerce").notnull().all():
                        bad_rows = df_upload[pd.to_numeric(df_upload[col], errors="coerce").isnull()].index.tolist()
                        validation_errors.append(
                            f"Column '{col}' contains non-numeric or missing (NaN) values at rows: {bad_rows[:5]}"
                        )

            if validation_errors:
                for err in validation_errors:
                    st.error(f"⚠️ {err}")
            else:
                st.success("✅ File format and feature columns verified!")

                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    samples_payload = df_upload[WINE_FEATURE_NAMES].to_dict(orient="records")

                    with st.spinner(f"Classifying {len(samples_payload)} samples via backend..."):
                        try:
                            batch_response = client.predict_batch(samples_payload)
                            st.success("✅ Batch Inference Complete!")

                            predictions = batch_response.get("predictions", [])
                            total_rec = batch_response.get("total_records", len(predictions))
                            success_rec = batch_response.get("successful_predictions", len(predictions))

                            st.info(
                                f"Processed {total_rec} records | Successfully classified: {success_rec} samples"
                            )

                            # Build combined result DataFrame
                            result_rows = []
                            for p in predictions:
                                s_idx = p.get("sample_index")
                                p_class = p.get("predicted_class")
                                p_name = p.get("predicted_class_name")
                                probs = p.get("probabilities", {})

                                row = {
                                    "sample_index": s_idx,
                                    "predicted_class": p_class,
                                    "predicted_class_name": p_name,
                                }
                                for c_name, c_prob in probs.items():
                                    row[f"prob_{c_name}"] = round(c_prob, 4)
                                result_rows.append(row)

                            results_df = pd.DataFrame(result_rows)
                            combined_df = pd.concat([df_upload.reset_index(drop=True), results_df], axis=1)

                            st.subheader("Classification Results")
                            st.dataframe(combined_df, use_container_width=True)

                            # Download results CSV
                            res_buffer = io.StringIO()
                            combined_df.to_csv(res_buffer, index=False)
                            st.download_button(
                                label="💾 Download Classified Results CSV",
                                data=res_buffer.getvalue().encode("utf-8"),
                                file_name="wine_classification_results.csv",
                                mime="text/csv",
                            )

                        except ApiClientError as exc:
                            st.error(f"❌ Batch Prediction Error: {exc.message}")
                            if exc.detail:
                                st.caption(f"Details: {exc.detail}")

        except (ValueError, pd.errors.ParserError, KeyError, OSError) as exc:
            st.error(f"Failed to parse CSV file: {exc}")


# -----------------------------------------------------------------------------
# MODULE 3: MODEL INFORMATION & METRICS
# -----------------------------------------------------------------------------
elif view_mode == "Model Architecture & Metrics":
    st.header("📊 Model Architecture, Hyperparameters & Metrics")
    st.markdown("Live metadata and benchmark statistics fetched from the FastAPI `/model-info` endpoint.")

    with st.spinner("Fetching model metadata from backend..."):
        try:
            info = client.model_info()

            st.subheader("Model Overview")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Model Architecture", info.get("model_name", "SVC"))
                st.caption(f"Class: `{info.get('model_type', 'sklearn.svm.SVC')}`")
            with m_col2:
                st.metric("Required Features", info.get("feature_count", 13))
                st.caption(f"Pipeline: `{' -> '.join(info.get('pipeline_steps', []))}`")
            with m_col3:
                st.metric("Target Classes", len(info.get("target_class_names", [])))
                st.caption(f"Classes: `{', '.join(info.get('target_class_names', []))}`")

            st.divider()

            # Hyperparameters
            st.subheader("Tuned Champion Hyperparameters")
            hp = info.get("hyperparameters", {})
            hp_df = pd.DataFrame(list(hp.items()), columns=["Hyperparameter", "Value"])
            st.table(hp_df)

            st.divider()

            # Cross-Validation vs Holdout Metrics
            st.subheader("Performance Evaluation Metrics")
            met_col1, met_col2 = st.columns(2)

            with met_col1:
                st.markdown("### 🔄 5-Fold Cross-Validation (Training Partition)")
                cv = info.get("cross_validation_metrics", {})
                st.write(f"**Strategy:** {cv.get('strategy', '5-Fold Stratified')}")
                st.metric("Mean CV Macro F1", f"{cv.get('mean_cv_macro_f1', 0):.4f}")
                st.caption(f"Std Deviation: ±{cv.get('std_cv_macro_f1', 0):.4f}")

            with met_col2:
                st.markdown("### 🎯 Final Holdout Test Evaluation (36 Samples)")
                holdout = info.get("holdout_test_metrics", {})
                st.write(f"**Scope:** {holdout.get('partition_scope', 'Unseen Holdout Test Set')}")
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    st.metric("Holdout Accuracy", f"{holdout.get('accuracy', 0):.4f}")
                    st.metric("Holdout Macro F1", f"{holdout.get('macro_f1', 0):.4f}")
                with h_c2:
                    st.metric("Holdout Precision", f"{holdout.get('macro_precision', 0):.4f}")
                    st.metric("Holdout ROC-AUC (OvR)", f"{holdout.get('roc_auc_ovr_macro', 0):.4f}")

            st.divider()
            st.caption(
                f"Scikit-learn Version: `{info.get('scikit_learn_version')}` | "
                f"Joblib Version: `{info.get('joblib_version')}`"
            )

        except ApiClientError as exc:
            st.error(f"❌ Failed to load model metadata: {exc.message}")
            if exc.detail:
                st.caption(f"Details: {exc.detail}")
