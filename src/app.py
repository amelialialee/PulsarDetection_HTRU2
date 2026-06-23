# streamlit run src/app.py
import streamlit as st
import joblib, shap, numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Pulsar Detector", page_icon="★", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/pulsar_pipeline.joblib")

model = load_model()


# Feature labels & description
features = [
    ("Mean IP",  "Mean of the Integrated Profile",              "Average signal strength across the integrated pulse profile."),
    ("Std IP",   "Standard Deviation of the Integrated Profile","How much the pulse profile signal varies around its mean."),
    ("Kurt IP",  "Excess Kurtosis of the Integrated Profile",   "Measures sharpness of peaks in the pulse profile. Higher values indicate more pointed, pulsar-like spikes."),
    ("Skew IP",  "Skewness of the Integrated Profile",          "Asymmetry of the pulse profile shape. Pulsars tend to be more skewed than background noise."),
    ("Mean DM",  "Mean of the DM-SNR Curve",                    "Average signal-to-noise ratio across the dispersion measure curve."),
    ("Std DM",   "Standard Deviation of the DM-SNR Curve",      "Variation in the dispersion measure signal-to-noise ratio."),
    ("Kurt DM",  "Excess Kurtosis of the DM-SNR Curve",         "Peakedness of the DM-SNR curve. Sharp peaks are characteristic of real pulsars."),
    ("Skew DM",  "Skewness of the DM-SNR Curve",                "Asymmetry of the DM-SNR curve. Real pulsars often show a distinctive skew pattern."),
]

feature_names  = [f[0] for f in features]
feature_labels = {k: label for k, label, _ in features}
feature_desc   = {k: desc  for k, _, desc  in features}

for key, _, _ in features:
    if key not in st.session_state:
        st.session_state[key] = 50.0



# Sidebar
st.sidebar.title("Pulsar Detector")
st.sidebar.caption("HTRU2 Pulsar Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Signal Classification",
        "Overview",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "Pulsar detection using an XGBoost model trained on the HTRU2 dataset. Differentiating pulsars from background signals through analysis of key signal features derived from radio observations."
)


# Helper: single prediction + SHAP
def run_single_prediction(vals):
    X_input = np.array(vals).reshape(1, -1)
    pred    = model.predict(X_input)[0]
    proba   = model.predict_proba(X_input)[0][1]

    # Result banner
    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.success(f"**Pulsar detected** : {proba*100:.1f}% confidence")
        else:
            st.error(f"**Not a pulsar** : {proba*100:.1f}% pulsar probability")
            
        if pred == 1:
            if proba >= 0.85:
                phrase = "The model is highly confident this looks like a pulsar."
            elif proba >= 0.6:
                phrase = "The model leans toward this being a pulsar, but it is not fully certain."
            else:
                phrase = "The model only just classifies this as a pulsar. This is a borderline case."
        else:
            if proba <= 0.15:
                phrase = "The model is highly confident this is background noise."
            elif proba <= 0.4:
                phrase = "The model leans toward background noise, though the signal is not perfectly clean."
            else:
                phrase = "The model only narrowly rules out a pulsar. This is a borderline case."
        st.caption(phrase)           
            
    with col2:
        st.metric("Pulsar Probability", f"{proba*100:.1f}%")
        st.progress(float(proba))

    

    # ── Input feature table ────────────────────────────────────
    st.markdown("---")
    st.subheader("Feature Values")

    col1, col2 = st.columns([1.5, 1])  # adjust widths if needed

    with col1:
        df_input = pd.DataFrame({
            "Feature": [feature_labels[k] for k in feature_names],
            "Value": [round(v, 4) for v in vals],
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Signal Profile")

        quick_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": vals
        })

        st.bar_chart(quick_df.set_index("Feature"))

    # ── SHAP explanation ───────────────────────────────────────
    st.markdown("---")
    st.subheader("SHAP Explanation")
    st.caption(
        "SHAP (SHapley Additive exPlanations) explains how each feature influences this single prediction.\n "
        "The model starts from a baseline (average prediction) and each feature pushes it higher or lower.\n" 
        "Red bars push toward pulsar while the blue bars push away."
    )

    clf    = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X_input)

    explainer = shap.TreeExplainer(clf)
    sv        = explainer.shap_values(X_scaled)
    sv_array  = np.array(sv)

    # Determine SHAP values and base value
    if isinstance(sv, list):
        # list format: sv[1] is the pulsar class
        shap_vals = np.array(sv[1]).flatten()
        ev = explainer.expected_value
        base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    elif sv_array.ndim == 3:
        # shape (1, n_features, 2)
        shap_vals  = sv_array[0, :, 1]
        ev = explainer.expected_value
        base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    elif sv_array.ndim == 2 and sv_array.shape[0] == 1:
        # shape (1, n_features) — single binary output (newer XGBoost + SHAP)
        shap_vals  = sv_array[0]
        ev = explainer.expected_value
        base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    else:
        # fallback
        shap_vals  = sv_array.flatten()
        ev = explainer.expected_value
        base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    full_labels = [feature_labels[k] for k in feature_names]

    explanation = shap.Explanation(
        values        = shap_vals,
        base_values   = base_value,
        data          = X_scaled[0],
        feature_names = full_labels,
    )
    
    st.subheader("Prediction Breakdown")

    st.caption(
        "Shows how the model builds the final prediction step by step from the baseline.\n"
        "Each feature adds or subtracts from the starting value until the final prediction is reached."
    )

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:  
        fig, ax = plt.subplots(figsize=(7, 2))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # Feature impact bar
    st.markdown("---")
    st.subheader("Feature Influence Summary (Direction and Magnitude)")
    st.caption(
        "Ranks features by how strongly they influenced this prediction.\n"
        "Right (positive) values push toward pulsar, left (negative) values push away."
    )

    impact_df = pd.DataFrame({
        "Feature": full_labels,
        "SHAP value": shap_vals,
    }).sort_values("SHAP value")

    colours = ["#4C9BE8" if v < 0 else "#E8503A" for v in impact_df["SHAP value"]]

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
    
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.barh(impact_df["Feature"], impact_df["SHAP value"], color=colours)
        ax2.axvline(0, color="#888780", linewidth=0.8)
        ax2.set_xlabel("SHAP value (impact on prediction)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=False)
        plt.close(fig2)


# PAGE 1: Classify a Signal 
if page == "Signal Classification":
    st.title("Pulsar Signal Classifier")
    st.caption("This project is to classify radio signal candidates as pulsar or non-pulsar using a trained XGBoost machine learning model. Explore how the model makes its decisions and learn the background behind pulsars and signal features in the other sections.")

    st.markdown("---")
    st.subheader("Input Signal Features")
    st.markdown(
        "Enter values for all 8 features of a radio signal candidate to classify it as a pulsar or background noise."
        "\nHover over the **?** icon beside each field for a description."
    )

    c1, c2 = st.columns(2)
    for i, (key, label, desc) in enumerate(features):
        col = c1 if i < 4 else c2
        st.session_state[key] = col.number_input(
            label,
            min_value = 0.0,
            max_value = 500.0,
            value     = float(st.session_state[key]),
            step      = 0.001,
            format    = "%.4f",
            help      = desc,
            key       = f"num_{key}",
        )

    if st.button("Classify Signal", type="primary"):
        vals = [st.session_state[k] for k in feature_names]
        run_single_prediction(vals)

# PAGE 2: Overview
elif page == "Overview":
    st.title("Overview")

    tab1, tab2, tab3= st.tabs([
        "Pulsar Overview",
        "Model Overview",
        "Signal Feature Guide"
    ])

    with tab1:
        st.markdown("## Pulsar")
        st.markdown(
            "A rapidly rotating **neutron star** (one of the densest objects in the universe). A neutron star is what remains after a massive star explodes in a supernova: a sphere roughly the size of a city, yet "
            "containing more mass than the Sun, compressed so tightly that its protons and electrons are crushed into neutrons."
        )

        st.markdown("#### The lighthouse effect")
        st.markdown(
            "Pulsars emit two narrow beams of radio waves from their magnetic poles, like a lighthouse. As the star spins (some rotate hundreds of times per second), these beams sweep across space."
            "So if Earth happens to lie in the path of one of those beams, a regular, repeating pulse of radio emission is detected each time the beam sweeps past."
        )

        st.markdown("---")
        st.markdown("## What this project is trying to do")
        st.markdown(
            "Radio telescopes surveying the sky record enormous volumes of data. "
            "The HTRU2 dataset was assembled from the High Time Resolution Universe Survey conducted using the Parkes radio telescope which scanned the southern sky for short-duration radio pulses. "
            "So utilising the 8 given statistical properties of a recorded signal, decides whether it came from a pulsar or is background interference."
        )
        steps_proj = [
            ("Receives the 8 statistical features",
             "Mean, standard deviation, excess kurtosis and skewness of both "
             "the integrated pulse profile and the DM-SNR curve."),
            ("Scales them",
             "StandardScaler normalises each feature to the same scale as the "
             "training set, so the model's learned thresholds still apply."),
            ("Classifies via XGBoost",
             "400 gradient-boosted trees vote on whether the pattern matches "
             "known pulsar signatures. A probability is computed."),
            ("Explains the decision via SHAP",
             "Each feature's contribution to that specific prediction is "
             "decomposed and visualised, showing which measurements were most "
             "influential."),
        ]
        for i, (title, detail) in enumerate(steps_proj, 1):
            st.markdown(f"**{i}. {title}** : {detail}")

    # What the model does
    with tab2:
        st.markdown("###XGBoost")
        st.markdown(
            """XGBoost builds decision trees sequentially, where each new tree is trained specifically to correct the mistakes of all previous trees. "
            The final prediction is a weighted sum of all trees' outputs, converted to a probability via the sigmoid function. A key practical challenge for this model is radio frequency interference.
            Earth-based signals from phones, satellites, and power lines can look statistically similar to pulsar pulses, making the classification boundary
            irregular and non-linear. This is one reason why tree-based ensembles like XGBoost outperform linear classifiers on this dataset. 
            """
        )
        # Performance
        st.markdown("### Model performance")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Recall",    "90.5%", help="% of real pulsars correctly detected")
        m2.metric("Precision", "80.9%", help="% of pulsar predictions that are correct")
        m3.metric("F1",        "85.5%", help="Harmonic mean of recall and precision")
        m4.metric("ROC-AUC",   "0.977", help="Area under ROC curve — 1.0 is perfect")
        m5.metric("PR-AUC",    "0.929", help="Area under precision-recall curve")

        st.markdown("#### Full Classification Report")
        report_df = pd.DataFrame({
            "Class":     ["Non-pulsar", "Pulsar", "Accuracy",
                          "Macro average", "Weighted average"],
            "Precision": [0.985, 0.809, "",    0.897, 0.975],
            "Recall":    [0.984, 0.905, 0.974, 0.945, 0.974],
            "F1":        [0.984, 0.855, 0.974, 0.920, 0.974],
            "Support":   [3252,  328,   3580,  3580,  3580],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Tuned Hyperparameters")
        hp_df = pd.DataFrame({
            "Parameter":        ["n_estimators", "max_depth", "learning_rate",
                                 "subsample", "colsample_bytree", "scale_pos_weight"],
            "Value":            [400, 9, 0.01, 0.8, 0.7, 1],
            "What It Is": [
                "Total number of trees in the ensemble",
                "Maximum depth each individual tree is allowed to grow "
                "Deeper trees capture more complex patterns but risk overfitting",
                "Step size applied at each boosting round. "
                "Lower = more conservative learning (typically better generalisation)",
                "Fraction of training rows each tree is trained on (row-level sampling) "
                "Reduces overfitting by introducing randomness",
                "Fraction of features considered when building each tree (column-level "
                "sampling). Also reduces overfitting and speeds up training*",
                "Set to 1 because SMOTE already balances the training classes. "
                "Setting it higher as well would over correct for imbalance."
            ],
        })
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

        st.markdown("### Primary Metric: Recall")
        st.markdown(
            """ There is no setting that maximises both recall and precision at once (catching more pulsars always means accepting more false alarms). Because
            missing a real pulsar is the "costlier" error, recall was the primary metric throughout tuning and model selection. 
            F1 was the secondary metric to ensure precision did not collapse entirely."""
        )
        st.markdown("##### Precision-recall tradeoff")
        st.markdown(
            "The threshold was kept at 0.5 for this project to meets the recall-first objective without making precision impractically low. "
        )
        pr_df = pd.DataFrame({
            "Threshold example": ["0.3 (lower)", "0.5 (current)", "0.7 (higher)"],
            "Effect on recall":  ["↑ Higher: catches more real pulsars",
                                  "90.5%: current setting",
                                  "↓ Lower: misses more real pulsars"],
            "Effect on precision": ["↓ Lower: more false alarms",
                                    "80.9%: current setting",
                                    "↑ Higher: fewer false alarms"],
        })
        st.dataframe(pr_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
    
        #HTRU2 & Pipeline 
        st.markdown("### The HTRU2 dataset")
        st.markdown(
            "HTRU2 (High Time Resolution Universe Survey 2) is a benchmark dataset assembled from the Parkes radio telescope's survey. "
            "Each of its 17,898 rows represents one radio signal candidate that was recorded and processed by the telescope. Each candidate is described by "
            "8 statistical numbers and labelled as either a real pulsar (1) or background noise (0)."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total candidates",  "17,898")
        col2.metric("Real pulsars",       "1,639  (9.2%)")
        col3.metric("Background noise",   "16,259  (90.8%)")

        st.markdown("##### The class imbalance problem")
        st.markdown(
            """Only 1 in every ~10 candidates is a real pulsar. A model that simply predicts 'not a pulsar' for every signal would easily be correct 90.8% of the time. This would not be helpful for actually finding pulsars. 
            Standard accuracy is therefore a misleading metric, which is why recall and F1 were used instead."""
        )

        st.markdown(
            """Addressing this imbalance, SMOTE (Synthetic Minority Oversampling Technique) generates new, synthetic pulsar examples by interpolating between real ones. 
            During training,  this balances the classes so the model has equal exposure to pulsar and non-pulsar patterns."""
        )

         
    # Tab 3: Signal Feature Guide 
    with tab3:
        st.markdown("## Signal Feature Guide")
        st.markdown(
            "Each of the 8 features is a statistical summary of one of two signal "
            "components recorded for every candidate. The same four statistics "
            "(mean, standard deviation, excess kurtosis, skewness) are computed "
            "for each component."
        )

        st.markdown("---")
        st.markdown("### Signal Components")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Integrated Pulse Profile (IP)")
            st.markdown("""
                The average shape of the recorded pulse, folded over many periods.
                A real pulsar produces a sharp, consistent peak (like a narrow
                spike) that appears in exactly the same place each rotation.
                Background noise produces a flat or irregular profile.
                \nFeatures: mean_ip, std_ip, kurt_ip, skew_ip""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("### DM-SNR Curve")
            st.markdown("""
                Radio signals are smeared in time as they travel through the
                interstellar medium. The DM-SNR curve shows how the signal-to-noise
                ratio changes as a function of the assumed dispersion measure (DM).
                A real pulsar shows a sharp peak at the correct DM value.
                RFI typically shows a flat or scattered curve.
                \nFeatures: mean_dm, std_dm, kurt_dm, skew_dm""",
                unsafe_allow_html=True,
            )


        st.markdown("---")
        st.markdown("### Individual feature breakdown")
        st.caption("Click each feature to understand what it means.")

        for key, label, desc in features:
            with st.expander(f"{label}"):
                st.write(desc)

                if "Integrated Profile" in label and "Mean" in label:
                    st.info("""
                        The integrated profile is built by folding the raw signal at the candidate period. The mean tells you the average signal strength 
                        across that folded profile. Real pulsars tend to have a lower, more focused mean because most of the profile is quiet (the pulse energy 
                        is concentrated into a narrow spike rather than spread evenly)."""
                    )
                elif "Integrated Profile" in label and "Standard Deviation" in label:
                    st.info("""
                        The standard deviation of the integrated profile measures how much the signal varies across the pulse bins. A high value means the 
                        profile has strong peaks alongside quiet regions (the hallmark of a well-defined pulse shape). Background noise tends to have a flatter, 
                        more uniform profile with lower standard deviation."""
                    )
                elif "Integrated Profile" in label and "Kurtosis" in label:
                    st.info("""
                        Excess kurtosis measures whether the profile has sharper, more extreme peaks than a normal distribution would. Positive kurtosis means 
                        the distribution is more 'pointed' (typical of real pulsar pulses), which concentrate energy into a very short phase window. Noise tends to be flatter (lower kurtosis).
                        """
                    )
                elif "Integrated Profile" in label and "Skewness" in label:
                    st.info(
                        """Skewness captures the asymmetry of the pulse profile shape. A value near zero means the profile is roughly symmetric. Real pulsars 
                        often have asymmetric profiles with a sharper leading or trailing edge, producing a non-zero skewness. Background interference tends to be more symmetric on average.
                        """
                    )
                elif "DM-SNR" in label and "Mean" in label:
                    st.info(
                        """The DM-SNR curve plots signal-to-noise ratio against dispersion measure trial values. Its mean captures the average SNR across that curve. 
                        Real pulsars produce a sharp SNR peak at the correct dispersion measure, which keeps the overall mean relatively low since most trial DM values 
                        return poor SNR. Noise tends to have a flatter curve with a higher mean."""
                    )
                elif "DM-SNR" in label and "Standard Deviation" in label:
                    st.info(
                        """The standard deviationof the DM-SNR curve reflects how much the SNR varies across dispersion measure trials. A high standard deviation 
                        indicates one region of the curve stands out strongly from the rest (consistent with a real dispersion sweep signature). Noise candidates 
                        "produce a more uniform curve with lower standard deviation."""
                    )
                elif "DM-SNR" in label and "Kurtosis" in label:
                    st.info(
                        """Excess kurtosis of the DM-SNR curve measures how sharply peaked 
                        the SNR response is at the best dispersion measure. A high value means the SNR spikes steeply at one point and drops away quickly (the expected 
                        behaviour for a real dispersed pulsar signal). Noise produces a flatter, broader SNR response with lower kurtosis."""
                    )
                elif "DM-SNR" in label and "Skewness" in label:
                    st.info(
                        """Skewness of the DM-SNR curve captures whether the SNR response is symmetric around its peak or tails off more in one direction. 
                        The dispersion sweep for a real pulsar produces a characteristic asymmetric curve shape, giving a distinct skewness value. 
                        Radio frequency interference and noise tend to produce more symmetric or randomly skewed DM-SNR curves."""
                    )
