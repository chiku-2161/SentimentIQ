"""
SentimentIQ – AI-Powered Sentiment Analysis and Customer Feedback Intelligence Dashboard
Streamlit Web Application featuring multi-page analytics, single-text inference,
bulk CSV batch processing, model evaluation metrics, and linguistic feedback insights.
"""

import sys
import json
import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    DATASET_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    SENTIMENT_CLASSES,
    CLASS_POSITIVE,
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    SENTIMENT_COLORS,
    MAX_FILE_SIZE_MB,
)
from src.preprocessing import clean_text
from src.predict import SentimentPredictor
from src.evaluation import load_metrics
from src.analytics import (
    compute_sentiment_distribution,
    generate_feedback_insights,
)
from src.utils import load_and_validate_csv, detect_text_column, FileValidationError

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SentimentIQ - Customer Feedback Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main container and font styles */
    .main {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .app-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .app-header h1 {
        color: #F8FAFC !important;
        margin: 0 0 8px 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94A3B8;
        margin: 0;
        font-size: 1.05rem;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
    }
    .kpi-subtitle {
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Sentiment Badges */
    .sentiment-badge-pos {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .sentiment-badge-neg {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .sentiment-badge-neu {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #E5E7EB;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.88rem;
        color: #78350F;
    }
</style>
""", unsafe_allow_html=True)


# --- RESOURCE CACHING ---
@st.cache_resource(show_spinner=False)
def load_cached_predictor() -> SentimentPredictor:
    """Loads and caches the model inference engine."""
    return SentimentPredictor(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)


@st.cache_data(show_spinner=False)
def load_cached_dataset() -> pd.DataFrame:
    """Loads training/reference dataset from disk."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame(columns=["review_text", "sentiment"])


@st.cache_data(show_spinner=False)
def load_cached_metrics():
    """Loads saved model evaluation metrics."""
    if METRICS_PATH.exists():
        return load_metrics(METRICS_PATH)
    return None


# --- APPLICATION INITIALIZATION ---
try:
    predictor = load_cached_predictor()
except Exception as e:
    predictor = None
    st.error(f"Error loading machine learning models: {e}. Please ensure train_model.py has been executed.")

metrics_data = load_cached_metrics()
dataset_df = load_cached_dataset()


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("## 🧠 **SentimentIQ**")
    st.caption("AI-Powered Feedback Intelligence")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🔍 Single Text Analysis",
            "📁 Bulk CSV Batch Analysis",
            "🎯 Model Performance",
            "💡 Feedback Intelligence",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### **System Status**")
    if predictor is not None:
        st.success("🟢 Inference Engine: **Online**")
    else:
        st.error("🔴 Inference Engine: **Offline**")

    if metrics_data:
        acc = metrics_data["overall"]["accuracy"] * 100
        st.info(f"Model Accuracy: **{acc:.1f}%**")
        st.caption(f"Evaluated on {metrics_data['overall']['total_test_samples']:,} test records")
    
    st.markdown("---")
    st.caption("SentimentIQ v1.0 • Scikit-Learn • TF-IDF • Logistic Regression")


# --- PAGE 1: EXECUTIVE OVERVIEW ---
if selected_page == "📊 Executive Overview":
    st.markdown("""
    <div class="app-header">
        <h1>📊 Feedback Intelligence Overview</h1>
        <p>Real-time analytics and volume breakdown across customer sentiment classes.</p>
    </div>
    """, unsafe_allow_html=True)

    if dataset_df.empty:
        st.warning("Reference dataset not found at `dataset/reviews.csv`. Run `python dataset/download_dataset.py` to populate data.")
    else:
        dist_data = compute_sentiment_distribution(dataset_df, sentiment_column="sentiment")
        dist = dist_data["distribution"]
        total = dist_data["total_reviews"]

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Customer Reviews</div>
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-subtitle">Curated feedback records</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <div class="kpi-title">Positive Reviews</div>
                <div class="kpi-value" style="color: #10B981;">{dist['positive']['count']:,}</div>
                <div class="kpi-subtitle">{dist['positive']['percentage']}% of total volume</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #6B7280;">
                <div class="kpi-title">Neutral Reviews</div>
                <div class="kpi-value" style="color: #4B5563;">{dist['neutral']['count']:,}</div>
                <div class="kpi-subtitle">{dist['neutral']['percentage']}% of total volume</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #EF4444;">
                <div class="kpi-title">Negative Reviews</div>
                <div class="kpi-value" style="color: #EF4444;">{dist['negative']['count']:,}</div>
                <div class="kpi-subtitle">{dist['negative']['percentage']}% of total volume</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("### **Sentiment Distribution Ratio**")
            donut_fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[dist['positive']['count'], dist['neutral']['count'], dist['negative']['count']],
                hole=0.55,
                marker=dict(colors=[SENTIMENT_COLORS["positive"], SENTIMENT_COLORS["neutral"], SENTIMENT_COLORS["negative"]]),
                textinfo="label+percent",
                hoverinfo="value+percent",
            )])
            donut_fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            )
            st.plotly_chart(donut_fig, use_container_width=True)

        with chart_col2:
            st.markdown("### **Sentiment Review Counts**")
            bar_df = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [dist['positive']['count'], dist['neutral']['count'], dist['negative']['count']],
                "Color": [SENTIMENT_COLORS["positive"], SENTIMENT_COLORS["neutral"], SENTIMENT_COLORS["negative"]]
            })
            bar_fig = px.bar(
                bar_df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive": SENTIMENT_COLORS["positive"],
                    "Neutral": SENTIMENT_COLORS["neutral"],
                    "Negative": SENTIMENT_COLORS["negative"]
                },
                text="Count"
            )
            bar_fig.update_traces(textposition='outside')
            bar_fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                showlegend=False,
                xaxis_title="",
                yaxis_title="Total Feedback Volume"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # Dataset Explorer with search & filter
        st.markdown("### **Customer Feedback Dataset Explorer**")
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment Class:",
                ["All Classes", "Positive", "Neutral", "Negative"],
                index=0
            )
        with filter_col2:
            search_query = st.text_input("Search keywords in customer feedback:", placeholder="e.g., flight, delay, baggage, crew...")

        filtered_df = dataset_df.copy()
        if sentiment_filter != "All Classes":
            filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter.lower()]
        if search_query:
            filtered_df = filtered_df[filtered_df["review_text"].astype(str).str.contains(search_query, case=False, na=False)]

        st.caption(f"Displaying {len(filtered_df):,} matching customer reviews (showing first 50 rows)")
        st.dataframe(
            filtered_df[["review_text", "sentiment"]].head(50),
            column_config={
                "review_text": st.column_config.TextColumn("Customer Review Text", width="large"),
                "sentiment": st.column_config.TextColumn("Sentiment Label", width="small"),
            },
            use_container_width=True,
            hide_index=True
        )


# --- PAGE 2: SINGLE TEXT ANALYSIS ---
elif selected_page == "🔍 Single Text Analysis":
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Single Review Sentiment Inference</h1>
        <p>Classify text feedback in real time with probability distributions and negation preservation.</p>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("Model predictor is not initialized. Please train the model first.")
    else:
        st.markdown("#### **Quick Preset Examples**")
        pcol1, pcol2, pcol3 = st.columns(3)
        sample_text = ""
        
        # Session state management for text input
        if "input_text_val" not in st.session_state:
            st.session_state["input_text_val"] = ""

        with pcol1:
            if st.button("🌟 Positive Flight Experience", use_container_width=True):
                st.session_state["input_text_val"] = "Outstanding customer service! The flight attendants were polite, attentive, and made our trip wonderful."
        with pcol2:
            if st.button("⚠️ Frustrated Delay Complaint", use_container_width=True):
                st.session_state["input_text_val"] = "Worst airline ever! 6 hour delay, lost my baggage, and customer service was completely rude and unhelpful."
        with pcol3:
            if st.button("ℹ️ Neutral Flight Inquiry", use_container_width=True):
                st.session_state["input_text_val"] = "What time does flight 420 from Boston arrive in Chicago? Need gate information."

        review_input = st.text_area(
            "Enter customer review or feedback:",
            value=st.session_state["input_text_val"],
            placeholder="Type or paste customer feedback here...",
            height=130,
            key="review_text_area",
        )

        btn_col1, btn_col2, _ = st.columns([1, 1, 3])
        with btn_col1:
            analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🔄 Clear Text", use_container_width=True):
                st.session_state["input_text_val"] = ""
                st.rerun()

        if analyze_btn or review_input.strip():
            if not review_input.strip():
                st.info("Please enter some text or click one of the preset example buttons above.")
            else:
                with st.spinner("Analyzing text sentiment..."):
                    result = predictor.predict_single(review_input)

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                probs = result["probabilities"]
                cleaned = result["cleaned_text"]

                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.markdown("### **Prediction Result**")
                    if sentiment == CLASS_POSITIVE:
                        badge_html = f'<div class="sentiment-badge-pos">😄 POSITIVE</div>'
                    elif sentiment == CLASS_NEGATIVE:
                        badge_html = f'<div class="sentiment-badge-neg">😡 NEGATIVE</div>'
                    else:
                        badge_html = f'<div class="sentiment-badge-neu">😐 NEUTRAL</div>'

                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** `{confidence * 100:.1f}%`")
                    st.progress(float(confidence))

                    st.markdown("#### **Preprocessed Normalized Text:**")
                    st.code(cleaned if cleaned else "(Empty after normalization)", language="text")

                    st.markdown("""
                    <div class="disclaimer-box">
                        <strong>Model Notice:</strong> Predictions are statistical estimates generated by a trained Logistic Regression model and should be interpreted as probabilistic indicators.
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown("### **Class Probability Distribution**")
                    prob_df = pd.DataFrame({
                        "Sentiment": ["Positive", "Neutral", "Negative"],
                        "Probability": [probs[CLASS_POSITIVE], probs[CLASS_NEUTRAL], probs[CLASS_NEGATIVE]],
                        "Color": [SENTIMENT_COLORS[CLASS_POSITIVE], SENTIMENT_COLORS[CLASS_NEUTRAL], SENTIMENT_COLORS[CLASS_NEGATIVE]],
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x="Probability",
                        y="Sentiment",
                        orientation="h",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive": SENTIMENT_COLORS[CLASS_POSITIVE],
                            "Neutral": SENTIMENT_COLORS[CLASS_NEUTRAL],
                            "Negative": SENTIMENT_COLORS[CLASS_NEGATIVE],
                        },
                        text=[f"{p * 100:.1f}%" for p in prob_df["Probability"]],
                        range_x=[0, 1.05]
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        margin=dict(t=20, b=20, l=20, r=20),
                        height=260,
                        showlegend=False,
                        xaxis_title="Predicted Probability",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)


# --- PAGE 3: BULK CSV ANALYSIS ---
elif selected_page == "📁 Bulk CSV Batch Analysis":
    st.markdown("""
    <div class="app-header">
        <h1>📁 Bulk CSV Feedback Batch Analysis</h1>
        <p>Upload a CSV file of customer reviews, automatically process sentiment at scale, and export intelligence results.</p>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("Model predictor is not initialized. Please train the model first.")
    else:
        # Sample CSV generation for quick user testing
        st.markdown("#### **Upload or Test with Sample Data**")
        sample_csv_data = (
            "review_id,feedback_text\n"
            "1,\"The crew was courteous and the flight was perfectly on time!\"\n"
            "2,\"Sitting on tarmac for 3 hours with no air conditioning or water. Terrible!\"\n"
            "3,\"Can I check my carry-on bag at the gate for tomorrow's departure?\"\n"
            "4,\"Not good at all. Extremely disappointing experience with customer service.\"\n"
            "5,\"Loved the extra legroom and complimentary snacks. Will fly again!\"\n"
            "6,\"Flight delayed by 40 minutes due to weather in Dallas.\"\n"
            "7,\"Never had a worse customer care experience. Absolutely unacceptable!\"\n"
            "8,\"Quick check-in process and friendly staff at the ticket counter.\"\n"
        )
        
        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            uploaded_file = st.file_uploader(
                "Upload customer reviews CSV file (max 10 MB):",
                type=["csv"],
                help="File must contain at least one column with customer feedback text."
            )
        with up_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_csv_data,
                file_name="sentiment_sample_reviews.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download a ready-to-test 8-row sample CSV file."
            )

        if uploaded_file is not None:
            try:
                # Validate and load CSV safely
                file_bytes = uploaded_file.getvalue()
                df_upload = load_and_validate_csv(file_bytes, max_mb=MAX_FILE_SIZE_MB)
                
                st.success(f"Successfully loaded CSV with **{len(df_upload):,} rows** and **{len(df_upload.columns)} columns**.")

                # Suggest text column
                suggested_col = detect_text_column(df_upload)
                col_options = list(df_upload.columns)
                default_idx = col_options.index(suggested_col) if suggested_col in col_options else 0

                text_column = st.selectbox(
                    "Select the column containing review text:",
                    options=col_options,
                    index=default_idx
                )

                if st.button("⚡ Process All Reviews", type="primary"):
                    with st.spinner(f"Analyzing {len(df_upload):,} reviews via TF-IDF Logistic Regression..."):
                        # Run batch prediction
                        pred_df = predictor.predict_batch(df_upload[text_column], include_probabilities=True)
                        
                        # Merge predictions with original DataFrame
                        merged_df = df_upload.copy()
                        merged_df["predicted_sentiment"] = pred_df["predicted_sentiment"]
                        merged_df["confidence"] = pred_df["confidence"]
                        merged_df["prob_positive"] = pred_df["prob_positive"]
                        merged_df["prob_neutral"] = pred_df["prob_neutral"]
                        merged_df["prob_negative"] = pred_df["prob_negative"]
                        
                        st.session_state["bulk_results"] = merged_df

                # Display Results if available in session state
                if "bulk_results" in st.session_state:
                    res_df = st.session_state["bulk_results"]
                    st.divider()
                    st.markdown("### **Batch Analysis Summary**")

                    bulk_counts = res_df["predicted_sentiment"].value_counts().to_dict()
                    pos_c = bulk_counts.get(CLASS_POSITIVE, 0)
                    neu_c = bulk_counts.get(CLASS_NEUTRAL, 0)
                    neg_c = bulk_counts.get(CLASS_NEGATIVE, 0)
                    total_c = len(res_df)

                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    with b_col1:
                        st.metric("Total Analyzed", f"{total_c:,}")
                    with b_col2:
                        st.metric("Positive", f"{pos_c:,}", f"{pos_c/total_c*100:.1f}%")
                    with b_col3:
                        st.metric("Neutral", f"{neu_c:,}", f"{neu_c/total_c*100:.1f}%")
                    with b_col4:
                        st.metric("Negative", f"{neg_c:,}", f"{neg_c/total_c*100:.1f}%")

                    # Charts
                    b_chart1, b_chart2 = st.columns([1, 1])
                    with b_chart1:
                        pie_fig = px.pie(
                            values=[pos_c, neu_c, neg_c],
                            names=["Positive", "Neutral", "Negative"],
                            color=["Positive", "Neutral", "Negative"],
                            color_discrete_map={
                                "Positive": SENTIMENT_COLORS[CLASS_POSITIVE],
                                "Neutral": SENTIMENT_COLORS[CLASS_NEUTRAL],
                                "Negative": SENTIMENT_COLORS[CLASS_NEGATIVE],
                            },
                            hole=0.45,
                            title="Batch Sentiment Proportion"
                        )
                        pie_fig.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
                        st.plotly_chart(pie_fig, use_container_width=True)

                    with b_chart2:
                        conf_fig = px.histogram(
                            res_df,
                            x="confidence",
                            color="predicted_sentiment",
                            color_discrete_map={
                                "positive": SENTIMENT_COLORS[CLASS_POSITIVE],
                                "neutral": SENTIMENT_COLORS[CLASS_NEUTRAL],
                                "negative": SENTIMENT_COLORS[CLASS_NEGATIVE],
                            },
                            nbins=20,
                            title="Prediction Confidence Distribution"
                        )
                        conf_fig.update_layout(
                            height=280,
                            margin=dict(t=30, b=10, l=10, r=10),
                            xaxis_title="Confidence Score",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(conf_fig, use_container_width=True)

                    # Export Download Button
                    st.markdown("### **Export Intelligence Results**")
                    csv_buffer = io.StringIO()
                    res_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Analyzed CSV with Predictions",
                        data=csv_buffer.getvalue(),
                        file_name="sentimentiq_analyzed_results.csv",
                        mime="text/csv",
                        type="primary"
                    )

                    # Table view with sentiment filter
                    st.markdown("### **Detailed Results Table**")
                    s_filter = st.multiselect(
                        "Filter by Sentiment Class:",
                        options=["positive", "neutral", "negative"],
                        default=["positive", "neutral", "negative"]
                    )
                    display_sub = res_df[res_df["predicted_sentiment"].isin(s_filter)]
                    st.dataframe(display_sub, use_container_width=True)

            except FileValidationError as fve:
                st.error(f"Validation Error: {fve}")
            except Exception as e:
                st.error(f"Unexpected error processing CSV: {e}")


# --- PAGE 4: MODEL PERFORMANCE ---
elif selected_page == "🎯 Model Performance":
    st.markdown("""
    <div class="app-header">
        <h1>🎯 Real Model Performance & Evaluation</h1>
        <p>Empirical metrics evaluated on 20% strictly held-out test data (2,886 records). No fabricated results.</p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics_data:
        st.warning("Model evaluation metrics not found. Please run `python train_model.py` to generate metrics.")
    else:
        ov = metrics_data["overall"]
        labels = metrics_data["labels"]
        cm = metrics_data["confusion_matrix"]
        per_class = metrics_data["per_class"]

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #3B82F6;">
                <div class="kpi-title">Accuracy</div>
                <div class="kpi-value" style="color: #1D4ED8;">{ov['accuracy']*100:.2f}%</div>
                <div class="kpi-subtitle">Overall test accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <div class="kpi-title">Precision (Macro)</div>
                <div class="kpi-value" style="color: #047857;">{ov['precision_macro']*100:.2f}%</div>
                <div class="kpi-subtitle">Unweighted mean precision</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                <div class="kpi-title">Recall (Macro)</div>
                <div class="kpi-value" style="color: #B45309;">{ov['recall_macro']*100:.2f}%</div>
                <div class="kpi-subtitle">Unweighted mean recall</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #8B5CF6;">
                <div class="kpi-title">F1-Score (Macro)</div>
                <div class="kpi-value" style="color: #6D28D9;">{ov['f1_macro']*100:.2f}%</div>
                <div class="kpi-subtitle">Harmonic mean precision-recall</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m_col1, m_col2 = st.columns([1, 1])

        with m_col1:
            st.markdown("### **Confusion Matrix Heatmap**")
            # Format matrix for Plotly heatmap
            display_labels = [cls.capitalize() for cls in labels]
            z_vals = np.array(cm)
            text_vals = [[f"{val:,}" for val in row] for row in z_vals]

            cm_fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=display_labels,
                y=display_labels,
                colorscale="Blues",
                text=text_vals,
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                hoverongaps=False,
            ))
            cm_fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Ground Truth",
                margin=dict(t=20, b=20, l=20, r=20),
                height=340,
                yaxis_autorange='reversed'
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        with m_col2:
            st.markdown("### **Per-Class Metrics Comparison**")
            class_df = pd.DataFrame([
                {
                    "Class": cls.capitalize(),
                    "Precision": per_class[cls]["precision"],
                    "Recall": per_class[cls]["recall"],
                    "F1-Score": per_class[cls]["f1_score"],
                    "Test Samples": per_class[cls]["support"],
                }
                for cls in labels
            ])
            
            bar_metrics = go.Figure()
            bar_metrics.add_trace(go.Bar(name='Precision', x=class_df['Class'], y=class_df['Precision'], marker_color='#3B82F6'))
            bar_metrics.add_trace(go.Bar(name='Recall', x=class_df['Class'], y=class_df['Recall'], marker_color='#10B981'))
            bar_metrics.add_trace(go.Bar(name='F1-Score', x=class_df['Class'], y=class_df['F1-Score'], marker_color='#8B5CF6'))
            bar_metrics.update_layout(
                barmode='group',
                yaxis=dict(range=[0, 1.05]),
                yaxis_title="Score (0.0 - 1.0)",
                margin=dict(t=20, b=20, l=20, r=20),
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(bar_metrics, use_container_width=True)

        # Full Classification Report Table
        st.markdown("### **Full Classification Report**")
        st.dataframe(
            class_df,
            column_config={
                "Class": st.column_config.TextColumn("Sentiment Category", width="medium"),
                "Precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                "Recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                "F1-Score": st.column_config.ProgressColumn("F1-Score", min_value=0.0, max_value=1.0, format="%.3f"),
                "Test Samples": st.column_config.NumberColumn("Held-out Test N", format="%d"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Architecture and Training Specifications
        st.markdown("### **Model Pipeline Specifications**")
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.markdown("""
            - **Model Algorithm:** Logistic Regression (`class_weight='balanced'`)
            - **Solver:** L-BFGS (`max_iter=1000`)
            - **Feature Extraction:** TF-IDF Vectorizer
            - **N-Gram Range:** `(1, 2)` (Unigrams + Bigrams)
            - **Max Features:** `10,000` features
            """)
        with spec_col2:
            st.markdown(f"""
            - **Train / Test Split:** `80%` Training / `20%` Held-out Test
            - **Split Strategy:** Stratified by class labels
            - **Random Seed:** `42` (Strictly reproducible)
            - **Total Evaluated Records:** `{ov['total_test_samples']:,}`
            - **Weighted F1-Score:** `{ov['f1_weighted']*100:.2f}%`
            """)


# --- PAGE 5: INSIGHTS AND REPORTS ---
elif selected_page == "💡 Feedback Intelligence":
    st.markdown("""
    <div class="app-header">
        <h1>💡 Feedback Insights & Keyword Intelligence</h1>
        <p>Linguistic pattern discovery, frequent descriptive n-grams, and exportable business intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    if dataset_df.empty:
        st.warning("Reference dataset not found. Please ensure `dataset/reviews.csv` exists.")
    else:
        with st.spinner("Extracting descriptive linguistic patterns across customer feedback..."):
            insights = generate_feedback_insights(dataset_df, text_column="review_text", sentiment_column="sentiment", top_n=12)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>Descriptive Pattern Disclaimer:</strong> The terms and phrases below represent observed co-occurrence frequencies in customer feedback. They are descriptive indicators of sentiment context, not confirmed causal drivers of customer decisions.
        </div>
        """, unsafe_allow_html=True)

        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("### 🌟 **Top Terms in Positive Feedback**")
            pos_terms = insights["descriptive_patterns"]["top_positive_terms"]
            if pos_terms:
                pos_df = pd.DataFrame(pos_terms)
                pos_fig = px.bar(
                    pos_df,
                    x="frequency",
                    y="term",
                    orientation="h",
                    color_discrete_sequence=[SENTIMENT_COLORS[CLASS_POSITIVE]],
                    text="frequency"
                )
                pos_fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Term Co-occurrence Count",
                    yaxis_title="",
                    height=380,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(pos_fig, use_container_width=True)
            else:
                st.info("No frequent terms found for positive feedback.")

        with col_neg:
            st.markdown("### ⚠️ **Top Terms in Negative Feedback**")
            neg_terms = insights["descriptive_patterns"]["top_negative_terms"]
            if neg_terms:
                neg_df = pd.DataFrame(neg_terms)
                neg_fig = px.bar(
                    neg_df,
                    x="frequency",
                    y="term",
                    orientation="h",
                    color_discrete_sequence=[SENTIMENT_COLORS[CLASS_NEGATIVE]],
                    text="frequency"
                )
                neg_fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Term Co-occurrence Count",
                    yaxis_title="",
                    height=380,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(neg_fig, use_container_width=True)
            else:
                st.info("No frequent terms found for negative feedback.")

        # Text length distributions
        st.markdown("### **Review Length & Verbiage Breakdown**")
        stats = insights["text_statistics"]
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric("Avg Review Length", f"{stats['avg_char_length']} chars")
        with s_col2:
            st.metric("Avg Negative Words", f"{stats['negative_avg_words']} words")
        with s_col3:
            st.metric("Avg Neutral Words", f"{stats['neutral_avg_words']} words")
        with s_col4:
            st.metric("Avg Positive Words", f"{stats['positive_avg_words']} words")

        # Exportable summary report
        st.markdown("### **Export Intelligence Report**")
        rep_json = json.dumps(insights, indent=4)
        st.download_button(
            label="📥 Download Full Intelligence Summary (JSON)",
            data=rep_json,
            file_name="sentimentiq_intelligence_report.json",
            mime="application/json",
            type="primary"
        )
