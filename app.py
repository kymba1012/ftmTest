"""
Dual-Purpose Streamlit Dashboard:
1. Executive Overview: Insights from historical data (20251128_FtData.csv).
2. Ticket Workbench: Upload tool for Managers to validate new field tickets.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Import core logic from refactored model_train
from model_train import (
    DATA_PATH,
    MODEL_PATH,
    NUMERIC_FIELDS,
    NaiveBayes,
    build_dataset,
    load_model,
    load_table,
)

st.set_page_config(
    page_title="Field Ticket AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Shared Utilities ------------------------------------------------------- #

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Optional[str]]]:
    """Convert dataframe rows to dictionaries (handling NaNs as None)."""
    # Ensure string conversion for categorical logic
    df_str = df.astype(object).where(pd.notnull(df), None)
    data = []
    for _, row in df_str.iterrows():
        record = {str(k): (str(v) if v is not None else None) for k, v in row.items()}
        data.append(record)
    return data


def _normalize_ticket_id(val: Optional[str]) -> str:
    if val is None:
        return "UNKNOWN"
    norm = str(val).strip()
    return norm if norm else "UNKNOWN"


@st.cache_resource(show_spinner=False)
def load_and_train_model(split: float = 0.8):
    """
    Loads bundled data, trains the model, and returns:
    1. The trained Model
    2. The raw DataFrame (for dashboarding)
    3. Evaluation metrics (for transparency)
    """
    if not DATA_PATH.exists():
        return None, None, None

    # Load data via model_train logic (now using Pandas)
    headers, body = load_table(DATA_PATH)
    data, labels = build_dataset(headers, body)

    # Create a DataFrame for the dashboard visualizations
    df_historical = pd.DataFrame(data)
    df_historical['error_label'] = labels 

    cols_to_fill = ['division', 'manager', 'customer', 'job']
    for col in cols_to_fill:
        if col in df_historical.columns:
            df_historical[col] = df_historical[col].fillna('Unassigned')

    # Prefer loading a persisted model to avoid retraining
    if MODEL_PATH.exists():
        try:
            model = load_model(MODEL_PATH)
            return model, df_historical, None
        except Exception:
            pass

    # Train Model
    total = len(labels)
    idx = list(range(total))
    random.seed(42)
    random.shuffle(idx)
    split_at = int(split * total)
    
    train_idx = idx[:split_at]
    test_idx = idx[split_at:]
    
    train_rows = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_rows = [data[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    model = NaiveBayes(NUMERIC_FIELDS)
    model.fit(train_rows, train_labels)

    # Calc accuracy on holdout
    probs = [model.predict_proba(r) for r in test_rows]
    preds = [1 if p >= 0.9 else 0 for p in probs]
    correct = sum(1 for p, y in zip(preds, test_labels) if p == y)
    accuracy = correct / len(test_labels) if test_labels else 0.0

    return model, df_historical, accuracy


# --- View: Executive Dashboard ---------------------------------------------- #
def render_executive_dashboard(df: pd.DataFrame, accuracy: float):
    st.title("📊 Executive Risk Overview")
    st.markdown("Insights derived from historical dataset: `20251128_FtData.csv`")

    # Key Metrics Row
    total_rows = len(df)
    total_errors = df['error_label'].sum()
    error_rate = (total_errors / total_rows) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Line Items", f"{total_rows:,}")
    c2.metric("Flagged Errors", f"{total_errors:,}")
    c3.metric("Overall Error Rate", f"{error_rate:.1f}%", delta_color="inverse")
    acc_val = accuracy if accuracy is not None else 0.0
    acc_label = f"{acc_val:.1%}" if accuracy is not None else "N/A"
    c4.metric("Model Confidence", acc_label, help="Accuracy on internal validation set")

    st.divider()

    # Deep Dive Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⚠️ Top High-Risk Managers")
        if 'manager' in df.columns:
            # Group by manager, calc mean error rate, filter for significant volume
            mgr_stats = df.groupby('manager').agg(
                count=('error_label', 'count'),
                errors=('error_label', 'sum')
            ).reset_index()
            mgr_stats['rate'] = mgr_stats['errors'] / mgr_stats['count']
            
            # Filter: only managers with > 10 tickets
            mgr_stats = mgr_stats[mgr_stats['count'] > 10].sort_values('rate', ascending=False).head(10)
            
            # Use st.table (static) instead of chart to avoid scrolling
            mgr_disp = mgr_stats[['manager', 'count', 'errors', 'rate']].copy()
            mgr_disp['rate'] = (mgr_disp['rate'] * 100).map('{:.1f}%'.format)
            mgr_disp.columns = ['Manager', 'Total Items', 'Error Count', 'Error Rate']
            st.table(mgr_disp.set_index('Manager'))
            
            st.caption("Managers with highest error rate (min. 10 items).")
        else:
            st.warning("Manager column not found in data.")

    with col_right:
        st.subheader("🏢 Risk by Division")
        if 'division' in df.columns:
            # Group by division, calc count and sum, filter for significant volume
            div_stats = df.groupby('division').agg(
                count=('error_label', 'count'),
                errors=('error_label', 'sum')
            ).reset_index()
            div_stats['rate'] = div_stats['errors'] / div_stats['count']
            
            # Filter: only divisions with > 10 tickets (removes noise)
            div_stats = div_stats[div_stats['count'] > 10].sort_values('rate', ascending=False).head(10)
            
            # Use st.table (static) instead of chart to avoid scrolling
            div_disp = div_stats[['division', 'count', 'errors', 'rate']].copy()
            div_disp['rate'] = (div_disp['rate'] * 100).map('{:.1f}%'.format)
            div_disp.columns = ['Division', 'Total Items', 'Error Count', 'Error Rate']
            st.table(div_disp.set_index('Division'))
            
            st.caption("Divisions sorted by error rate (min. 10 items).")
        else:
            st.warning("Division column not found in data.")

    # Recent Trends (Simulated if no date)
    st.subheader("📋 Raw Data Explorer")
    with st.expander("View Historical Data"):
        st.dataframe(df)


# --- View: Manager Workbench ------------------------------------------------ #
def render_manager_workbench(model: NaiveBayes):
    st.title("🛠️ Ticket Workbench")
    st.markdown("Upload current batch file to validate tickets against the AI model.")

    # Upload Section
    uploaded = st.file_uploader("Upload CSV Batch", type=["csv"], help="Must contain a ticket ID column (e.g. 'title')")
    
    if uploaded is None:
        st.info("👋 Waiting for file upload...")
        return

    try:
        df_upload = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    if df_upload.empty:
        st.warning("Uploaded file is empty.")
        return

    # Configuration
    with st.expander("⚙️ Analysis Settings", expanded=False):
        ticket_col = st.selectbox(
            "Ticket ID Column", 
            options=df_upload.columns, 
            index=(list(df_upload.columns).index("title") if "title" in df_upload.columns else 0)
        )
        threshold = st.slider("Strictness Threshold", 0.5, 0.99, 0.90, 0.01, help="Higher = Fewer rejections (only rejects very obvious errors).")

    # Processing
    if st.button("Analyze Batch", type="primary"):
        with st.spinner("Analyzing rows..."):
            data_records = df_to_records(df_upload)
            
            # Predict
            probs = [model.predict_proba(r) for r in data_records]
            preds = [1 if p >= threshold else 0 for p in probs]

            # Aggregate by Ticket
            ticket_results = defaultdict(lambda: {"rows": 0, "max_prob": 0.0, "errors": 0})
            
            for i, (row, prob, pred) in enumerate(zip(data_records, probs, preds)):
                tid = _normalize_ticket_id(row.get(ticket_col))
                ticket_results[tid]["rows"] += 1
                ticket_results[tid]["max_prob"] = max(ticket_results[tid]["max_prob"], prob)
                if pred == 1:
                    ticket_results[tid]["errors"] += 1

            # Decision Logic
            approved = {}
            rejected = {}
            for tid, stats in ticket_results.items():
                if stats["errors"] > 0:
                    rejected[tid] = stats
                else:
                    approved[tid] = stats

        # Results Display
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tickets", len(ticket_results))
        m2.metric("✅ Approved", len(approved))
        m3.metric("❌ Rejected", len(rejected), delta_color="inverse")

        # Rejected Tickets (Action Items)
        if rejected:
            st.error(f"Action Required: {len(rejected)} Tickets Rejected")
            
            # Create detail dataframe for rejected
            rej_data = []
            for tid, stats in rejected.items():
                rej_data.append({
                    "Ticket ID": tid,
                    "Row Count": stats["rows"],
                    "Bad Rows": stats["errors"],
                    "Max Risk Score": f"{stats['max_prob']:.2f}"
                })
            
            st.dataframe(
                pd.DataFrame(rej_data).sort_values("Max Risk Score", ascending=False),
                width=True,
                hide_index=True
            )
        else:
            st.success("All tickets passed validation!")

        # Download Report
        df_out = df_upload.copy()
        df_out['AI_Risk_Score'] = probs
        df_out['AI_Flagged'] = preds
        
        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "Download Annotated Report",
            data=csv_buffer.getvalue(),
            file_name="ai_ticket_analysis.csv",
            mime="text/csv"
        )


# --- Main Application ------------------------------------------------------- #
def main():
    # Load Model (Singleton)
    model, df_hist, accuracy = load_and_train_model()
    
    if model is None:
        st.error(f"❌ Missing Training Data: Could not find {DATA_PATH}")
        st.stop()

    # Sidebar Info (Navigation removed)
    st.sidebar.title("System Status")
    st.sidebar.success(f"Model Ready\n\nAccuracy: {accuracy:.1%}")
    st.sidebar.markdown(f"**Training Data:** {len(df_hist)} rows")
    st.sidebar.divider()
    st.sidebar.markdown("### Guide\n1. Review **Executive Risk Overview** for historical trends.\n2. Scroll down to **Ticket Workbench** to validate new files.")

    # 1. Executive Section
    render_executive_dashboard(df_hist, accuracy)
    
    st.markdown("---")
    
    # 2. Workbench Section
    render_manager_workbench(model)


if __name__ == "__main__":
    main()
