import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from matplotlib.patches import Wedge

@st.cache_resource
def load_model():
    with open("CellSight/xgb_model.pkl", "rb") as f:
        return pickle.load(f)
   
model = load_model()

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

# --- Streamlit app configuration ---

st.set_page_config(
    page_title="CellSight",
    layout="wide"
)

# --- Font Awesome icons ---
st.markdown("""
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

#--- Numeric data input step state ---
if "numeric_step" not in st.session_state:
    st.session_state.numeric_step = "data"

def numeric_step_header():
    c1, c2, c3 = st.columns([1, 1, 8])

    with c1:
        if st.session_state.numeric_step == "data":
            st.markdown(
                "<i class='fa-solid fa-file-lines' style='color:#0B3C5D;'></i> "
                "<b>Data Input</b>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<i class='fa-solid fa-file-lines' style='color:#94A3B8;'></i> "
                "Data",
                unsafe_allow_html=True
            )

    # --- Step 2: Prediction ---
    with c2:
        if st.session_state.numeric_step == "prediction":
            st.markdown(
                "<i class='fa-solid fa-chart-line' style='color:#0B3C5D;'></i> "
                "<b>Prediction</b>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<i class='fa-solid fa-chart-line' style='color:#94A3B8;'></i> "
                "Prediction",
                unsafe_allow_html=True
            )

    # --- Progress bar ---
    with c3:
        st.progress(50 if st.session_state.numeric_step == "data" else 100)


# --- Side bar ---
def sidebar_item(icon, label, target_page):
    col_icon, col_button = st.columns([1, 6])

    with col_icon:
        st.markdown(
            f"<div style='padding-top:6px;'>"
            f"<i class='{icon}' style='color:#0B3C5D; font-size:18px;'></i>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_button:
        if st.button(label, key=label):
            st.session_state.page = target_page
            st.rerun()

def sidebar_item_disabled(icon, label):
    col_icon, col_text = st.columns([1, 6])

    with col_icon:
        st.markdown(
            f"<div style='padding-top:6px;'>"
            f"<i class='{icon}' style='color:#0B3C5D; font-size:18px;'></i>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_text:
        st.markdown(
            f"<span style='color:#0B3C5D; font-size:16px;'>{label}</span>",
            unsafe_allow_html=True
        )

with st.sidebar:

    st.markdown("## CellSight")
    st.markdown("---")

    # Home
    sidebar_item("fa-solid fa-house", "Home", "landing")
    st.markdown("---")

    # Data Input section header
    st.markdown(
        "<span style='font-size:15px; color:#0B3C5D;'>DATA INPUT</span>",
        unsafe_allow_html=True
    )

    sidebar_item_disabled("fa-solid fa-inbox", "Numerical data")
    sidebar_item_disabled("fa-solid fa-image", "Image data")
    sidebar_item_disabled("fa-solid fa-brain", "Other")
    st.markdown("---")

    sidebar_item_disabled("fa-solid fa-clock-rotate-left", "History")
    st.markdown("---")
    sidebar_item_disabled("fa-solid fa-comments", "Chat")


# --- Landing page ---

def clickable_card(title, description, enabled, target_page=None, highlight=False):

    border_style = "border-left:6px solid #0B3C5D;" if highlight else ""

    card = f"""
    <div class="landing-card" style="
        background-color:#F1F5F9;
        padding:28px;
        border-radius:18px;
        box-shadow:0 8px 20px rgba(15,23,42,0.08);
        {border_style}
    ">
        <h4 style="color:#0B3C5D; margin-bottom:10px;">{title}</h4>
        <p style="color:#475569;">{description}</p>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

    if enabled:
        if st.button("Open", key=title, use_container_width=True):
            st.session_state.page = target_page
            st.rerun()
    else:
        st.button("Open", disabled=True, key=title, use_container_width=True)

if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Logo (Streamlit-native!) ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("CellSight/Logo4.png", width=6000)
 
    st.markdown(
        "<p style='text-align:center; font-size:20px; color:#475569;'>"
        "Clinical Decision Support for Tumor Assessment"
        "</p>",
        unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        clickable_card(
            title="Numerical cell measurements",
            description="Structured numerical features derived from cell morphology",
            enabled=True,
            target_page="numeric",
            highlight=True 
        )

    with col2:
        clickable_card(
            title="Pathology images",
            description="Histopathology slide-based tumor assessment",
            enabled=False,
            highlight=True 
        )

    with col3:
        clickable_card(
            title="Multimodal Data",
            description="Combined numerical and image-based decision support",
            enabled=False,
            highlight=True 
        )

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
    "<p style='text-align:center; color:#64748B; font-size:13px; margin-top:1px;'>"
    "Disclaimer: This tool is a research prototype and does not provide a medical diagnosis. "
    "It must not be used for clinical decision-making."
    "</p>",
    unsafe_allow_html=True
    )  
    
# --- Numerical data page ---

# =========================================================
# Numerical data page
# =========================================================
elif st.session_state.page == "numeric":
    
    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------
    top_left, top_right = st.columns([6, 1])

    with top_left:
        st.markdown(
            "<h1 style='color:#0B3C5D; font-size:42px; margin-bottom:4px;'>"
            "Numerical Cell Assessment"
            "</h1>",
            unsafe_allow_html=True
        )
    with top_right:
        st.image("CellSight/Logo4.png", width=200)

    st.caption("Risk estimation based on numerical cell morphology measurements")
    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Step header
    # -----------------------------------------------------
    numeric_step_header()
    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # STEP 2 — PREDICTION DASHBOARD (Step 1 is below)
    # =====================================================  
    if st.session_state.numeric_step == "prediction":

        st.divider()

        # --- Back to data ---
        back_left, _ = st.columns([6, 1])
        with back_left:
            if st.button("← Edit data"):
                st.session_state.numeric_step = "data"
                st.rerun()

        # --- Load stored results ---
        patient_data = st.session_state.get("last_patient_data")
        prediction = st.session_state.get("last_pred")
        probability = st.session_state.get("last_prob")
        shap_values = st.session_state.get("last_shap")

        if patient_data is None or shap_values is None:
            st.warning("No assessment found yet. Please enter data and run the assessment.")
            st.session_state.numeric_step = "data"
            st.rerun()

        # -----------------------------
        # KPI CARDS
        # -----------------------------
        st.markdown(
            "<h2 style='color:#0B3C5D;'>Assessment Results</h2>",
            unsafe_allow_html=True
        )
        
        if probability >= 0.75:
            risk_pattern = "High-risk pattern"
            risk_color = "#991B1B"   # red
        elif probability >= 0.4:
            risk_pattern = "Intermediate-risk pattern"
            risk_color = "#EA580C"   # orange
        else:
            risk_pattern = "Low-risk pattern"
            risk_color = "#065F46"   # green

        k1, k2= st.columns(2)

        with k1:
            with st.container(border=True):
                st.markdown("##### Classification")
                if prediction == 1:
                    st.error("Malignant")
                else:
                    st.success("Benign")

        with k2:
            with st.container(border=True):
                st.markdown("##### Risk Assessment")

                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown(
                        f"<span style='color:{risk_color}; "
                        f"font-weight:600; font-size:1.1em;'>"
                        f"{risk_pattern}</span>",
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"<span style='font-size:2.1em; font-weight:600;'>"
                        f"{probability:.1%}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<span style='color:#9CA3AF; font-size:0.75em; "
                        "margin-top:-10px; display:block;'>"
                        "Relative similarity to malignant cases"
                        "</span>",
                        unsafe_allow_html=True
                    )

    

        FEATURE_LABELS = {
            "radius_mean": "cell radius (mean)",
            "perimeter_mean": "cell perimeter (mean)",
            "area_mean": "cell area (mean)",
            "compactness_mean": "cell compactness (mean)",
            "concavity_mean": "cell concavity (mean)",
            "concave_points_mean": "number of concave points (mean)",
            "symmetry_mean": "cell symmetry (mean)",
            "fractal_dimension_mean": "boundary complexity (mean)",

            "radius_worst": "cell radius (worst)",
            "perimeter_worst": "cell perimeter (worst)",
            "area_worst": "cell area (worst)",
            "compactness_worst": "cell compactness (worst)",
            "concavity_worst": "cell concavity (worst)",
            "concave_points_worst": "number of concave points (worst)",
            "symmetry_worst": "cell symmetry (worst)",
            "fractal_dimension_worst": "boundary complexity (worst)",
        }
    

        # -----------------------------
        # EXPLAINABILITY
        # -----------------------------
        # --- Baseline & deviation ---
        base_value = explainer.expected_value
        if isinstance(base_value, (list, tuple)) or hasattr(base_value, "__len__"):
            base_value = base_value[1]

        local_output = base_value + shap_values[0].values.sum()
        delta = local_output - base_value

        delta_clipped = max(min(delta, 1.0), -1.0)

        def baseline_gauge(delta):
            fig, ax = plt.subplots(figsize=(18, 6))
            
            # Halbkreis-Bereiche
            ranges = [
                (-1.0, -0.2, "#1D4ED8"),  # below average (blue)
                (-0.2,  0.2, "#CBD5E1"),  # baseline (gray)
                ( 0.2,  1.0, "#B91C1C"),  # above average (red)
            ]

            for start, end, color in ranges:
                theta1 = 180 * (1 - (start + 1) / 2)
                theta2 = 180 * (1 - (end + 1) / 2)
                ax.add_patch(
                    Wedge((0, 0), 1, theta2, theta1, facecolor=color, edgecolor="white")
                )

            # Patient marker
            angle = np.pi * (1 - (delta + 1) / 2)
            ax.plot(
                [0, 0.85 * np.cos(angle)],
                [0, 0.85 * np.sin(angle)],
                linewidth=3,
                color="#0B3C5D"
            )
            ax.plot(0, 0, "o", color="#0B3C5D")

            # Labels
            ax.text(-1.0, -0.15, "Below average", ha="left", va="center", fontsize=10)
            ax.text( 0.0, -0.15, "Baseline", ha="center", va="center", fontsize=10)
            ax.text( 1.0, -0.15, "Above average", ha="right", va="center", fontsize=10)

            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)

            return fig
        
        # ----- Explainability section -----
        with st.container(border=True):
                st.markdown("##### Clinical Interpretation")

                top_features = shap_values[0].abs.values.argsort()[-3:][::-1]
                features_readable = [
                    FEATURE_LABELS.get(patient_data.columns[i], patient_data.columns[i].replace("_", " "))
                    for i in top_features
                ]

                if probability >= 0.7:
                    risk_sentence = (
                        "The overall pattern identified by the model is indicative of a high-risk morphological profile."
                    )
                elif probability >= 0.4:
                    risk_sentence = (
                        "The model identifies a mixed morphological pattern corresponding to an intermediate-risk profile."
                    )
                else:
                    risk_sentence = (
                        "The observed morphological pattern is more consistent with a low-risk profile."
                    )

                st.markdown(
                    f"""
                    {risk_sentence}<br>
                    In this case, the assessment is primarily driven by alterations in 
                    **{features_readable[0]}**, **{features_readable[1]}** and **{features_readable[2]}**.
                    """,
                    unsafe_allow_html=True
                )

        col_left, col_right = st.columns(2)

        with col_right:
            with st.container(border=True):
                st.markdown("##### Baseline comparison")

                fig_gauge = baseline_gauge(delta_clipped)
                st.pyplot(fig_gauge, use_container_width=False)

                if delta > 0.2:
                    txt = "This case deviates clearly above the model’s baseline risk profile."
                elif delta < -0.2:
                    txt = "This case lies below the model’s baseline risk profile."
                else:
                    txt = "This case lies close to the model’s baseline expectation."

                st.caption(txt)

        with col_left:
            with st.container(border=True):
                st.markdown("##### Feature Contributions")

                # Prepare data
                shap_df = pd.DataFrame({
                    "feature": patient_data.columns,
                    "contribution": shap_values[0].values
                })

                shap_df["abs"] = shap_df["contribution"].abs()
                shap_df = shap_df.sort_values("abs", ascending=False).head(10)

                # Plot
                fig, ax = plt.subplots(figsize=(5, 3.2))
                colors = ["#B91C1C" if v > 0 else "#15803D" for v in shap_df["contribution"]]

                ax.barh(
                    shap_df["feature"].str.replace("_", " "),
                    shap_df["contribution"],
                    color=colors
                )

                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Contribution to malignancy risk")
                ax.invert_yaxis()

                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)


        # -----------------------------
        # Disclaimer
        # -----------------------------
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
        "<p style='text-align:center; color:#64748B; font-size:13px; margin-top:1px;'>"
        "Disclaimer: This tool is a research prototype and does not provide a medical diagnosis. "
        "It must not be used for clinical decision-making."
        "</p>",
        unsafe_allow_html=True
        )  

        # Demo Buttons 
        PRESET_BENIGN = {
            "radius_mean": 12.3, "texture_mean": 17.0, "perimeter_mean": 78.0, "area_mean": 460.0, "smoothness_mean": 0.085,
            "compactness_mean": 0.060, "concavity_mean": 0.030, "concave_points_mean": 0.020, "symmetry_mean": 0.175, "fractal_dimension_mean": 0.060,

            "radius_se": 0.28, "texture_se": 1.20, "perimeter_se": 1.90, "area_se": 22.0, "smoothness_se": 0.006,
            "compactness_se": 0.012, "concavity_se": 0.018, "concave_points_se": 0.010, "symmetry_se": 0.020, "fractal_dimension_se": 0.003,

            "radius_worst": 13.5, "texture_worst": 21.0, "perimeter_worst": 87.0, "area_worst": 560.0, "smoothness_worst": 0.115,
            "compactness_worst": 0.135, "concavity_worst": 0.120, "concave_points_worst": 0.065, "symmetry_worst": 0.250, "fractal_dimension_worst": 0.075,
        }
        PRESET_BORDERLINE = {
            "radius_mean": 13.0, "texture_mean": 19.5, "perimeter_mean": 88.0, "area_mean": 520.0, "smoothness_mean": 0.095,
            "compactness_mean": 0.10, "concavity_mean": 0.040, "concave_points_mean": 0.035, "symmetry_mean": 0.18, "fractal_dimension_mean": 0.062,

            "radius_se": 0.40, "texture_se": 1.45, "perimeter_se": 2.70, "area_se": 35.0, "smoothness_se": 0.007,
            "compactness_se": 0.020, "concavity_se": 0.030, "concave_points_se": 0.015, "symmetry_se": 0.025, "fractal_dimension_se": 0.004,

            "radius_worst": 17.2, "texture_worst": 25.5, "perimeter_worst": 112.0, "area_worst": 930.0, "smoothness_worst": 0.130,
            "compactness_worst": 0.210, "concavity_worst": 0.220, "concave_points_worst": 0.110, "symmetry_worst": 0.290, "fractal_dimension_worst": 0.082,
        }
        PRESET_MALIGNANT = {
            "radius_mean": 14.5, "texture_mean": 20.3, "perimeter_mean": 95.2, "area_mean": 700.0, "smoothness_mean": 0.10, 
            "compactness_mean": 0.12, "concavity_mean": 0.08, "concave_points_mean": 0.05, "symmetry_mean": 0.18, "fractal_dimension_mean": 0.06,
            
            "radius_se": 0.40, "texture_se": 1.20, "perimeter_se": 3.0, "area_se": 40.0, "smoothness_se": 0.007, 
            "compactness_se": 0.020, "concavity_se": 0.030, "concave_points_se": 0.015, "symmetry_se": 0.020, "fractal_dimension_se": 0.003,
            
            "radius_worst": 16.0, "texture_worst": 25.0, "perimeter_worst": 110.0, "area_worst": 900.0, "smoothness_worst": 0.14,
            "compactness_worst": 0.30, "concavity_worst": 0.35, "concave_points_worst": 0.15, "symmetry_worst": 0.30, "fractal_dimension_worst": 0.09,
        }
        
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("🟢",use_container_width=True):
                for k, v in PRESET_BENIGN.items():
                    st.session_state[k] = v
                st.session_state.numeric_step = "data"
                st.rerun()

        with b2:
            if st.button("🟡", use_container_width=True):
                for k, v in PRESET_BORDERLINE.items():
                    st.session_state[k] = v
                st.session_state.numeric_step = "data"
                st.rerun()

        with b3:
            if st.button("🔴", use_container_width=True):
                for k, v in PRESET_MALIGNANT.items():
                    st.session_state[k] = v
                st.session_state.numeric_step = "data"
                st.rerun()


    # =====================================================
    # STEP 1 — DATA INPUT
    # =====================================================
    else:

        # -----------------------------
        # Ensure default values exist
        # -----------------------------
        DEFAULT_INPUTS = {
            "radius_mean": 14.2,
            "texture_mean": 20.4,
            "perimeter_mean": 92.1,
            "area_mean": 654.9,
            "smoothness_mean": 0.096,
            "compactness_mean": 0.104,
            "concavity_mean": 0.089,
            "concave_points_mean": 0.048,
            "symmetry_mean": 0.181,
            "fractal_dimension_mean": 0.062,

            "radius_se": 0.45,
            "texture_se": 1.2,
            "perimeter_se": 2.8,
            "area_se": 40.5,
            "smoothness_se": 0.006,
            "compactness_se": 0.025,
            "concavity_se": 0.031,
            "concave_points_se": 0.012,
            "symmetry_se": 0.021,
            "fractal_dimension_se": 0.004,

            "radius_worst": 16.5,
            "texture_worst": 28.3,
            "perimeter_worst": 108.2,
            "area_worst": 880.6,
            "smoothness_worst": 0.132,
            "compactness_worst": 0.254,
            "concavity_worst": 0.312,
            "concave_points_worst": 0.135,
            "symmetry_worst": 0.295,
            "fractal_dimension_worst": 0.085,
        }
        
        for key, value in DEFAULT_INPUTS.items():
            if key not in st.session_state:
                st.session_state[key] = value

        ALL_INPUT_KEYS = list(DEFAULT_INPUTS.keys())

        # -----------------------------
        # Clear inputs
        # -----------------------------
        _, clear_col = st.columns([10, 1])
        with clear_col:
            if st.button("Clear inputs"):
                for key in ALL_INPUT_KEYS:
                    st.session_state[key] = 0.0
                st.rerun()

        # -----------------------------
        # Helper
        # -----------------------------
        def num_input(label, key):
            return st.number_input(
                label,
                value=float(st.session_state[key]),
                key=key,
                format="%.2f"
            )

        # -----------------------------
        # INPUT GROUPS
        # -----------------------------
        # A) Shape & Size
        with st.expander("Shape & Size", expanded=True):
            t1, t2, t3 = st.tabs(["Mean", "SE", "Worst"])

            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    num_input("Radius (mean)", "radius_mean")
                    num_input("Perimeter (mean)", "perimeter_mean")
                with c2:
                    num_input("Area (mean)", "area_mean")

            with t2:
                c1, c2 = st.columns(2)
                with c1:
                    num_input("Radius (SE)", "radius_se")
                    num_input("Perimeter (SE)", "perimeter_se")
                with c2:
                    num_input("Area (SE)", "area_se")

            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    num_input("Radius (worst)", "radius_worst")
                    num_input("Perimeter (worst)", "perimeter_worst")
                with c2:
                    num_input("Area (worst)", "area_worst")

        # B) Texture & Smoothness
        with st.expander("Texture & Smoothness", expanded=False):
            t1, t2, t3 = st.tabs(["Mean", "SE", "Worst"])

            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    texture_mean = num_input("Texture (mean)", "texture_mean")
                    smoothness_mean = num_input("Smoothness (mean)", "smoothness_mean")
                with c2:
                    fractal_dimension_mean = num_input("Fractal dimension (mean)", "fractal_dimension_mean")

            with t2:
                c1, c2 = st.columns(2)
                with c1:
                    texture_se = num_input("Texture (SE)", "texture_se")
                    smoothness_se = num_input("Smoothness (SE)", "smoothness_se")
                with c2:
                    fractal_dimension_se = num_input("Fractal dimension (SE)", "fractal_dimension_se")

            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    texture_worst = num_input("Texture (worst)", "texture_worst")
                    smoothness_worst = num_input("Smoothness (worst)", "smoothness_worst")
                with c2:
                    fractal_dimension_worst = num_input("Fractal dimension (worst)", "fractal_dimension_worst")

        # C) Boundary / Irregularity
        with st.expander("Boundary irregularity", expanded=False):
            t1, t2, t3 = st.tabs(["Mean", "SE", "Worst"])

            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    compactness_mean = num_input("Compactness (mean)", "compactness_mean")
                    concavity_mean = num_input("Concavity (mean)", "concavity_mean")
                with c2:
                    concave_points_mean = num_input("Concave points (mean)", "concave_points_mean")

            with t2:
                c1, c2 = st.columns(2)
                with c1:
                    compactness_se = num_input("Compactness (SE)", "compactness_se")
                    concavity_se = num_input("Concavity (SE)", "concavity_se")
                with c2:
                    concave_points_se = num_input("Concave points (SE)", "concave_points_se")

            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    compactness_worst = num_input("Compactness (worst)", "compactness_worst")
                    concavity_worst = num_input("Concavity (worst)", "concavity_worst")
                with c2:
                    concave_points_worst = num_input("Concave points (worst)", "concave_points_worst")

        # D) Symmetry
        with st.expander("Symmetry", expanded=False):
            t1, t2, t3 = st.tabs(["Mean", "SE", "Worst"])

            with t1:
                symmetry_mean = num_input("Symmetry (mean)", "symmetry_mean")

            with t2:
                symmetry_se = num_input("Symmetry (SE)", "symmetry_se")

            with t3:
                symmetry_worst = num_input("Symmetry (worst)", "symmetry_worst")


        # -----------------------------
        # Collect patient data
        # -----------------------------
        patient_data = pd.DataFrame(
            {k: [st.session_state[k]] for k in ALL_INPUT_KEYS}
        )

        # -----------------------------
        # Run assessment
        # -----------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        center = st.columns([1, 2, 1])
        with center[1]:
            run_clicked = st.button("Run Assessment", type="primary", use_container_width=True)

        if run_clicked:
            # Prediction
            pred = model.predict(patient_data)[0]
            prob = model.predict_proba(patient_data)[0, 1]

            # SHAP explanation
            shap_vals = explainer(patient_data)

            # save results for dashboard
            st.session_state["last_patient_data"] = patient_data
            st.session_state["last_pred"] = int(pred)
            st.session_state["last_prob"] = float(prob)
            st.session_state["last_shap"] = shap_vals

            # Go to prediction step
            st.session_state.numeric_step = "prediction"
            st.rerun()

        # -----------------------------
        # Disclaimer
        # -----------------------------
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
        "<p style='text-align:center; color:#64748B; font-size:13px; margin-top:1px;'>"
        "Disclaimer: This tool is a research prototype and does not provide a medical diagnosis. "
        "It must not be used for clinical decision-making."
        "</p>",
        unsafe_allow_html=True
        )  