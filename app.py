
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================

def get_plot_colors():
    """Return plot colors that work in both Streamlit light and dark themes."""
    try:
        theme = st.context.theme.type
    except Exception:
        theme = "light"

    if theme == "dark":
        return {
            "text": "#FAFAFA",
            "grid": "#555555",
            "background": "#0E1117"
        }

    return {
        "text": "#262730",
        "grid": "#CCCCCC",
        "background": "#FFFFFF"
    }


st.set_page_config(
    page_title="AI Breast Cancer Diagnostic Support",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================

MODEL_DIR = "models"

lr_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

rf_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

svm_model = joblib.load(
    os.path.join(MODEL_DIR, "svm.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

nn_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "neural_network.keras")
)

FEATURE_NAMES = joblib.load(
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

METRICS = joblib.load(
    os.path.join(MODEL_DIR, "metrics.pkl")
)

X_TEST = joblib.load(
    os.path.join(MODEL_DIR, "X_test.pkl")
)

Y_TEST = joblib.load(
    os.path.join(MODEL_DIR, "y_test.pkl")
)

PREPROCESSING_SUMMARY = {
    "Dataset": "Wisconsin Breast Cancer Diagnostic Dataset",
    "Total Samples": 569,
    "Total Features": 30,
    "Missing Values": 0,
    "Duplicate Samples": 0,
    "Train-Test Split": "80:20",
    "Split Method": "Stratified",
    "Random State": 42,
    "Standardization": "StandardScaler",
    "Class Imbalance Handling": "Balanced Class Weights",
    "Malignant Samples": 212,
    "Benign Samples": 357,
    "Malignant Class Weight": 1.3382,
    "Benign Class Weight": 0.7982
}

FEATURE_IMPORTANCE = joblib.load(
    os.path.join(MODEL_DIR, "feature_importance.pkl")
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🧹 Data Preprocessing",
        "🔬 Prediction",
        "📊 Model Performance",
        "⭐ Feature Importance",
        "ℹ️ About"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🧬 AI-Based Breast Cancer Diagnostic Support")

    st.caption(
        "Machine learning and deep learning approaches for breast tumor "
        "classification using diagnostic features."
    )

    st.divider()

    # ========================================================
    # DATASET
    # ========================================================

    st.subheader("Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Samples", "569")

    with col2:
        st.metric("Features", "30")

    with col3:
        st.metric("Malignant", "212")

    with col4:
        st.metric("Benign", "357")

    st.divider()

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader("About the Dataset")

    left, right = st.columns([1.15, 0.85])

    with left:

        st.markdown("**Wisconsin Breast Cancer Diagnostic Dataset**")

        st.write(
            "The dataset contains numerical measurements describing "
            "characteristics of cell nuclei obtained from breast tissue "
            "samples."
        )

        st.write(
            "The classification task is to distinguish between malignant "
            "and benign tumor samples using these diagnostic measurements."
        )

        st.markdown("**Classes**")

        st.write(
            "• Malignant — 212 samples\n"
            "• Benign — 357 samples"
        )

    with right:

        class_counts = pd.Series(
            {
                "Malignant": 212,
                "Benign": 357
            }
        )

        st.bar_chart(
            class_counts,
            height=280
        )

    st.divider()

    # ========================================================
    # MODELS
    # ========================================================

    st.subheader("Models Used")

    models = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Support Vector Machine",
                "Neural Network"
            ],
            "Approach": [
                "Linear classification",
                "Ensemble classification",
                "Kernel-based classification",
                "Feed-forward deep learning"
            ]
        }
    )

    st.dataframe(
        models,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # WORKFLOW
    # ========================================================

    st.subheader("Workflow")

    st.write(
        "Input Data  →  Preprocessing  →  AI Classification  →  "
        "Model Comparison"
    )

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.markdown("**Input and Preprocessing**")

        st.write(
            "Tumor measurements are provided as model input. "
            "Standardization is applied where required by the trained models."
        )

    with workflow_col2:

        st.markdown("**Classification and Comparison**")

        st.write(
            "Four trained models generate predictions and probability "
            "estimates. Their performance is evaluated using classification "
            "metrics and ROC-AUC."
        )

    st.divider()

    # ========================================================
    # APPLICATION
    # ========================================================

    st.subheader("Application")

    st.write(
        "Use the navigation menu to enter a sample for prediction, "
        "compare model performance, or examine feature importance."
    )

    st.divider()

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.info(
        "**Research & Educational Use Only:** "
        "This application was developed as part of an academic mini-project. "
        "It is intended for demonstration and educational purposes and "
        "should not be used for clinical diagnosis or treatment decisions."
    )

# ============================================================
# DATA PREPROCESSING PAGE
# ============================================================

elif page == "🧹 Data Preprocessing":

    st.header("🧹 Data Preprocessing")

    st.write(
        "This section presents the preprocessing steps applied to the "
        "Wisconsin Breast Cancer Diagnostic Dataset before model training."
    )

    # ---------------------------------------------------------
    # 1. DATASET OVERVIEW
    # ---------------------------------------------------------
    st.subheader("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", PREPROCESSING_SUMMARY["Total Samples"])
    col2.metric("Features", PREPROCESSING_SUMMARY["Total Features"])
    col3.metric("Malignant", PREPROCESSING_SUMMARY["Malignant Samples"])
    col4.metric("Benign", PREPROCESSING_SUMMARY["Benign Samples"])

    st.write("### Data Quality Check")

    quality_df = pd.DataFrame({
        "Check": [
            "Missing Values",
            "Duplicate Samples"
        ],
        "Result": [
            PREPROCESSING_SUMMARY["Missing Values"],
            PREPROCESSING_SUMMARY["Duplicate Samples"]
        ],
        "Status": [
            "✓ No missing values",
            "✓ No duplicate samples"
        ]
    })

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # 2. CLASS DISTRIBUTION
    # ---------------------------------------------------------
    st.subheader("2. Class Distribution")

    st.write(
        "The dataset contains fewer malignant cases than benign cases. "
        "Therefore, class imbalance was considered during model training."
    )

    class_counts = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Samples": [
            PREPROCESSING_SUMMARY["Malignant Samples"],
            PREPROCESSING_SUMMARY["Benign Samples"]
        ]
    })

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        class_counts["Class"],
        class_counts["Samples"]
    )

    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution Before Training")

    for i, value in enumerate(class_counts["Samples"]):
        ax.text(i, value + 5, str(value), ha="center")

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------------
    # 3. TRAIN-TEST SPLIT
    # ---------------------------------------------------------
    st.subheader("3. Train-Test Split")

    split_col1, split_col2 = st.columns(2)

    split_col1.metric(
        "Training Set",
        "80%"
    )

    split_col2.metric(
        "Testing Set",
        "20%"
    )

    st.info(
        "A stratified 80:20 train-test split was used with random state 42. "
        "Stratification preserves the malignant/benign class proportion "
        "in both training and testing sets."
    )

    # ---------------------------------------------------------
    # 4. FEATURE STANDARDIZATION
    # ---------------------------------------------------------
    st.subheader("4. Feature Standardization")

    st.write(
        "StandardScaler was used for models that require feature scaling."
    )

    scaling_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Support Vector Machine",
            "Neural Network",
            "Random Forest"
        ],
        "Scaling": [
            "StandardScaler",
            "StandardScaler",
            "StandardScaler",
            "Not required"
        ]
    })

    st.dataframe(
        scaling_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "The scaler was fitted using the training data only and then "
        "applied to the test data, preventing data leakage."
    )

    # ---------------------------------------------------------
    # 5. CLASS IMBALANCE HANDLING
    # ---------------------------------------------------------
    st.subheader("5. Class Imbalance Handling")

    st.write(
        "Balanced class weights were used during model training. "
        "The minority malignant class receives a higher weight so that "
        "malignant cases are given greater importance during learning."
    )

    weight_df = pd.DataFrame({
        "Class": [
            "Malignant",
            "Benign"
        ],
        "Samples": [
            PREPROCESSING_SUMMARY["Malignant Samples"],
            PREPROCESSING_SUMMARY["Benign Samples"]
        ],
        "Class Weight": [
            PREPROCESSING_SUMMARY["Malignant Class Weight"],
            PREPROCESSING_SUMMARY["Benign Class Weight"]
        ]
    })

    st.dataframe(
        weight_df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # 6. PREPROCESSING SUMMARY
    # ---------------------------------------------------------
    st.subheader("6. Preprocessing Workflow")

    workflow = [
        "Raw Wisconsin Breast Cancer Dataset",
        "↓",
        "Missing-value and duplicate check",
        "↓",
        "Stratified 80:20 train-test split",
        "↓",
        "Standardization for applicable models",
        "↓",
        "Balanced class weights",
        "↓",
        "Training-ready dataset"
    ]

    for step in workflow:
        if step == "↓":
            st.markdown(
                "<div style='text-align:center; font-size:22px;'>↓</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    padding:10px;
                    margin:4px;
                    border:1px solid #cccccc;
                    border-radius:8px;
                    text-align:center;
                ">
                    <b>{step}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.caption(
        "These preprocessing steps were applied before evaluating the "
        "classification models."
    )

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔬 Prediction":

    st.title("🔬 Tumor Classification")

    st.write(
        "Enter tumor characteristics below or load a demonstration sample "
        "to evaluate the trained classification models."
    )

    # --------------------------------------------------------
    # DEMONSTRATION SAMPLES
    # --------------------------------------------------------

    from sklearn.datasets import load_breast_cancer

    demo_data = load_breast_cancer()

    malignant_indices = [
        i for i, label in enumerate(demo_data.target) if label == 0
    ][:10]

    benign_indices = [
        i for i, label in enumerate(demo_data.target) if label == 1
    ][:10]

    demo_indices = malignant_indices + benign_indices

    demo_options = [
        f"Demo {i + 1:02d} — "
        + ("Malignant" if demo_data.target[idx] == 0 else "Benign")
        for i, idx in enumerate(demo_indices)
    ]

    st.subheader("Demonstration Samples")

    st.write(
        "Select a real sample from the Wisconsin Breast Cancer Diagnostic "
        "Dataset to demonstrate classification using the trained models."
    )

    selected_demo = st.selectbox(
        "Select a demonstration sample:",
        demo_options
    )

    selected_index = demo_indices[
        demo_options.index(selected_demo)
    ]

    if st.button("Load Selected Sample"):

        selected_values = demo_data.data[selected_index].tolist()

        st.session_state.demo_values = selected_values

        for i, value in enumerate(selected_values):
            st.session_state[f"feature_{i}"] = float(value)

        st.success(
            f"{selected_demo} loaded successfully."
        )

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    input_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(FEATURE_NAMES):

        with col1 if i < 15 else col2:

            if f"feature_{i}" not in st.session_state:
                st.session_state[f"feature_{i}"] = 0.0

            value = st.number_input(
                feature,
                format="%.6f",
                key=f"feature_{i}"
            )

            input_values.append(value)

    st.markdown("---")

    predict_button = st.button(
        "🔍 Predict Tumor Classification",
        type="primary"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        X_input = np.array(input_values).reshape(1, -1)

        # Scaled input for LR, SVM and Neural Network
        X_input_scaled = scaler.transform(X_input)

        # Logistic Regression
        lr_prediction = lr_model.predict(X_input_scaled)[0]
        lr_probability = lr_model.predict_proba(X_input_scaled)[0]

        # Random Forest
        rf_prediction = rf_model.predict(X_input)[0]
        rf_probability = rf_model.predict_proba(X_input)[0]

        # SVM
        svm_prediction = svm_model.predict(X_input_scaled)[0]
        svm_probability = svm_model.predict_proba(X_input_scaled)[0]

        # Neural Network
        nn_probability = float(
            nn_model.predict(
                X_input_scaled,
                verbose=0
            )[0][0]
        )

        nn_prediction = 1 if nn_probability >= 0.5 else 0

        # 0 = Malignant
        # 1 = Benign

        predictions = [
            lr_prediction,
            rf_prediction,
            svm_prediction,
            nn_prediction
        ]

        model_names = [
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Neural Network"
        ]

        # Probabilities
        malignant_probabilities = [
            (1 - lr_probability[1]) * 100,
            (1 - rf_probability[1]) * 100,
            (1 - svm_probability[1]) * 100,
            (1 - nn_probability) * 100
        ]

        benign_probabilities = [
            lr_probability[1] * 100,
            rf_probability[1] * 100,
            svm_probability[1] * 100,
            nn_probability * 100
        ]

        # Consensus
        malignant_votes = predictions.count(0)
        benign_votes = predictions.count(1)

        if malignant_votes > benign_votes:
            final_prediction = "MALIGNANT"
            consensus_votes = malignant_votes
        else:
            final_prediction = "BENIGN"
            consensus_votes = benign_votes

        average_malignant_probability = (
            sum(malignant_probabilities) / 4
        )

        average_benign_probability = (
            sum(benign_probabilities) / 4
        )

        st.subheader("Prediction Result")

        if final_prediction == "MALIGNANT":
            st.error("⚠️ Model Consensus: MALIGNANT")
        else:
            st.success("✅ Model Consensus: BENIGN")

        st.write(
            f"🧠 **Model Consensus:** "
            f"{consensus_votes}/4 models agree"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Average Malignant Probability",
                f"{average_malignant_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Average Benign Probability",
                f"{average_benign_probability:.2f}%"
            )

        st.subheader("Individual Model Predictions")

        result_df = pd.DataFrame({
            "Model": model_names,
            "Prediction": [
                "Malignant" if p == 0 else "Benign"
                for p in predictions
            ],
            "Malignant Probability (%)": [
                round(x, 2)
                for x in malignant_probabilities
            ],
            "Benign Probability (%)": [
                round(x, 2)
                for x in benign_probabilities
            ]
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "This application is for educational and research purposes "
            "only and must not be used as a medical diagnostic tool."
        )

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.subheader("Classification Performance")

    # Prepare test data
    X_test_scaled_metrics = scaler.transform(X_TEST)

    # Test-set predictions
    lr_pred = lr_model.predict(X_test_scaled_metrics)

    rf_pred = rf_model.predict(X_TEST)

    svm_pred = svm_model.predict(X_test_scaled_metrics)

    nn_pred = (
        nn_model.predict(
            X_test_scaled_metrics,
            verbose=0
        ).ravel() >= 0.5
    ).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    metrics_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Neural Network"
        ],
        "Accuracy": [
            accuracy_score(Y_TEST, lr_pred),
            accuracy_score(Y_TEST, rf_pred),
            accuracy_score(Y_TEST, svm_pred),
            accuracy_score(Y_TEST, nn_pred)
        ],
        "Precision": [
            precision_score(Y_TEST, lr_pred),
            precision_score(Y_TEST, rf_pred),
            precision_score(Y_TEST, svm_pred),
            precision_score(Y_TEST, nn_pred)
        ],
        "Recall": [
            recall_score(Y_TEST, lr_pred),
            recall_score(Y_TEST, rf_pred),
            recall_score(Y_TEST, svm_pred),
            recall_score(Y_TEST, nn_pred)
        ],
        "F1-score": [
            f1_score(Y_TEST, lr_pred),
            f1_score(Y_TEST, rf_pred),
            f1_score(Y_TEST, svm_pred),
            f1_score(Y_TEST, nn_pred)
        ]
    })

    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Confusion Matrices")

    from sklearn.metrics import confusion_matrix

    confusion_data = {
        "Logistic Regression": lr_pred,
        "Random Forest": rf_pred,
        "SVM": svm_pred,
        "Neural Network": nn_pred
    }

    for model_name, predictions in confusion_data.items():

        cm = confusion_matrix(
            Y_TEST,
            predictions
        )

        st.markdown(f"**{model_name}**")

        plot_colors = get_plot_colors()

        fig_cm, ax_cm = plt.subplots(
            figsize=(4.5, 3.5)
        )

        fig_cm.patch.set_facecolor(
            plot_colors["background"]
        )

        ax_cm.set_facecolor(
            plot_colors["background"]
        )

        ax_cm.imshow(cm)

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])

        ax_cm.set_xticklabels(
            ["Malignant", "Benign"],
            color=plot_colors["text"]
        )

        ax_cm.set_yticklabels(
            ["Malignant", "Benign"],
            color=plot_colors["text"]
        )

        ax_cm.set_xlabel(
            "Predicted Label",
            color=plot_colors["text"]
        )

        ax_cm.set_ylabel(
            "True Label",
            color=plot_colors["text"]
        )

        ax_cm.set_title(
            f"{model_name} - Confusion Matrix",
            color=plot_colors["text"]
        )

        for i in range(2):

            for j in range(2):

                ax_cm.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=plot_colors["text"]
                )

        ax_cm.tick_params(
            colors=plot_colors["text"]
        )

        for spine in ax_cm.spines.values():
            spine.set_color(
                plot_colors["text"]
            )

        plt.tight_layout()

        st.pyplot(fig_cm)

        plt.close(fig_cm)

    # ========================================================
    # ROC CURVE COMPARISON
    # ========================================================

    st.subheader("ROC Curve Comparison")

    # Prepare test data
    X_test_scaled = scaler.transform(X_TEST)

    # Predicted probabilities
    lr_prob = lr_model.predict_proba(
        X_test_scaled
    )[:, 1]

    rf_prob = rf_model.predict_proba(
        X_TEST
    )[:, 1]

    svm_prob = svm_model.predict_proba(
        X_test_scaled
    )[:, 1]

    nn_prob = nn_model.predict(
        X_test_scaled,
        verbose=0
    ).ravel()

    # ROC curves
    lr_fpr, lr_tpr, _ = roc_curve(
        Y_TEST,
        lr_prob
    )

    rf_fpr, rf_tpr, _ = roc_curve(
        Y_TEST,
        rf_prob
    )

    svm_fpr, svm_tpr, _ = roc_curve(
        Y_TEST,
        svm_prob
    )

    nn_fpr, nn_tpr, _ = roc_curve(
        Y_TEST,
        nn_prob
    )

    # AUC scores
    lr_auc = roc_auc_score(
        Y_TEST,
        lr_prob
    )

    rf_auc = roc_auc_score(
        Y_TEST,
        rf_prob
    )

    svm_auc = roc_auc_score(
        Y_TEST,
        svm_prob
    )

    nn_auc = roc_auc_score(
        Y_TEST,
        nn_prob
    )

    # ========================================================
    # ROC PLOT
    # ========================================================

    plot_colors = get_plot_colors()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    fig.patch.set_facecolor(
        plot_colors["background"]
    )

    ax.set_facecolor(
        plot_colors["background"]
    )

    ax.plot(
        lr_fpr,
        lr_tpr,
        linewidth=2.5,
        label=f"Logistic Regression (AUC = {lr_auc:.3f})"
    )

    ax.plot(
        rf_fpr,
        rf_tpr,
        linewidth=2.5,
        linestyle="--",
        label=f"Random Forest (AUC = {rf_auc:.3f})"
    )

    ax.plot(
        svm_fpr,
        svm_tpr,
        linewidth=2.5,
        linestyle="-.",
        label=f"SVM (AUC = {svm_auc:.3f})"
    )

    ax.plot(
        nn_fpr,
        nn_tpr,
        linewidth=2.5,
        linestyle=":",
        label=f"Neural Network (AUC = {nn_auc:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier (AUC = 0.500)"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.02
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=12,
        color=plot_colors["text"]
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=12,
        color=plot_colors["text"]
    )

    ax.set_title(
        "ROC Curve Comparison of Classification Models",
        fontsize=14,
        fontweight="bold",
        color=plot_colors["text"]
    )

    ax.tick_params(
        colors=plot_colors["text"]
    )

    for spine in ax.spines.values():
        spine.set_color(
            plot_colors["text"]
        )

    legend = ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True
    )

    legend.get_frame().set_facecolor(
        plot_colors["background"]
    )

    legend.get_frame().set_edgecolor(
        plot_colors["grid"]
    )

    for label in legend.get_texts():
        label.set_color(
            plot_colors["text"]
        )

    ax.grid(
        True,
        linestyle=":",
        alpha=0.5,
        color=plot_colors["grid"]
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ========================================================
    # AUC TABLE
    # ========================================================

    st.subheader("ROC-AUC Scores")

    auc_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Neural Network",
            "Random Classifier"
        ],
        "ROC-AUC": [
            round(lr_auc, 4),
            round(rf_auc, 4),
            round(svm_auc, 4),
            round(nn_auc, 4),
            0.5000
        ]
    })

    st.dataframe(
        auc_df,
        use_container_width=True,
        hide_index=True
    )
    # ========================================================
    # NEURAL NETWORK ARCHITECTURE
    # ========================================================

    st.subheader("Neural Network Architecture")

    st.write(
        "The feed-forward neural network used for breast cancer "
        "classification consists of an input layer representing the "
        "diagnostic features, followed by fully connected layers and "
        "a sigmoid output layer for binary classification."
    )

    # --------------------------------------------------------
    # DISPLAY ACTUAL TRAINED MODEL ARCHITECTURE
    # --------------------------------------------------------

    architecture_data = []

    for i, layer in enumerate(nn_model.layers):

        output_shape = layer.output.shape

        architecture_data.append({
            "Layer": i + 1,
            "Layer Type": layer.__class__.__name__,
            "Output Shape": str(tuple(output_shape)),
            "Parameters": layer.count_params()
        })

    architecture_df = pd.DataFrame(
        architecture_data
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # VISUAL NETWORK STRUCTURE
    # --------------------------------------------------------

    st.markdown("### Network Structure")

    network_layers = []

    for layer in nn_model.layers:

        if isinstance(layer, tf.keras.layers.Dense):

            units = layer.units
            activation = layer.activation.__name__

            network_layers.append(
                f"Dense ({units})\n{activation}"
            )

        else:

            network_layers.append(
                layer.__class__.__name__
            )

    # --------------------------------------------------------
    # CREATE ARCHITECTURE DIAGRAM
    # --------------------------------------------------------

    plot_colors = get_plot_colors()

    fig_nn, ax_nn = plt.subplots(
        figsize=(12, 3.5)
    )

    fig_nn.patch.set_facecolor(
        plot_colors["background"]
    )

    ax_nn.set_facecolor(
        plot_colors["background"]
    )

    ax_nn.axis("off")

    if len(network_layers) == 1:

        x_positions = [0.5]

    else:

        x_positions = np.linspace(
            0.08,
            0.92,
            len(network_layers)
        )

    for i, (x, layer_text) in enumerate(
        zip(x_positions, network_layers)
    ):

        ax_nn.text(
            x,
            0.5,
            layer_text,
            ha="center",
            va="center",
            fontsize=11,
            color=plot_colors["text"],
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=plot_colors["background"],
                edgecolor=plot_colors["text"],
                linewidth=1.5
            )
        )

        # Connect layers with arrows

        if i < len(network_layers) - 1:

            ax_nn.annotate(
                "",
                xy=(
                    x_positions[i + 1] - 0.07,
                    0.5
                ),
                xytext=(
                    x + 0.07,
                    0.5
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.5,
                    color=plot_colors["text"]
                )
            )

    ax_nn.set_xlim(
        0,
        1
    )

    ax_nn.set_ylim(
        0,
        1
    )

    ax_nn.set_title(
        "Feed-Forward Neural Network Architecture",
        fontsize=14,
        fontweight="bold",
        color=plot_colors["text"]
    )

    plt.tight_layout()

    st.pyplot(fig_nn)

    plt.close(fig_nn)

    st.caption(
        "The architecture shown above is generated directly from "
        "the trained TensorFlow/Keras neural network loaded by the application."
    )

# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================

elif page == "⭐ Feature Importance":

    st.header("Feature Importance")

    try:

        importance_df = pd.DataFrame(
            FEATURE_IMPORTANCE
        ).copy()

        # Rank features
        importance_df.insert(
            0,
            "Rank",
            range(
                1,
                len(importance_df) + 1
            )
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            f"Unable to display feature importance: {e}"
        )

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.header("About the Application")

    st.markdown(
        """
        ### Objective

        The objective of this project is to develop and compare
        machine learning and deep learning approaches for breast
        cancer classification.

        ### Dataset

        The application uses the Wisconsin Breast Cancer
        Diagnostic Dataset containing numerical tumor
        characteristics.

        ### Machine Learning

        The application uses:

        - Logistic Regression
        - Random Forest
        - Support Vector Machine

        ### Deep Learning

        A feed-forward neural network developed using
        TensorFlow/Keras is also used.

        ### Clinical Decision Support

        The application demonstrates how artificial intelligence
        can potentially assist classification and decision-support
        workflows.

        However, the model is intended only for academic and
        research demonstration and must not be used as a
        substitute for clinical diagnosis.
        """
    )

    st.markdown("---")

    st.info(
        "Developed as an Application-Oriented Mini-Project "
        "for Artificial Intelligence in Healthcare."
    )
