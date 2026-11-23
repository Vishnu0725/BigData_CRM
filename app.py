import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
import os

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(
    page_title="CRM Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------
# CUSTOM CSS
# -------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
[data-testid="stMetric"] {
    background-color: #162033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263650;
}
[data-testid="stMetricLabel"] {
    color: #a8b6ca;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# LOAD DATA AND MODEL
# -------------------------------------------
DATA_PATH = "data/customers.csv"
MODEL_PATH = "models/churn_model.pkl"
METRICS_PATH = "models/metrics.json"

if not os.path.exists(DATA_PATH):
    st.error("Customer dataset not found. Run python generate_data.py first.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run python train_model.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

# -------------------------------------------
# LOAD HDFS SIMULATION REPORT
# -------------------------------------------
HDFS_REPORT_PATH = "hdfs_simulation/storage_report.json"

if os.path.exists(HDFS_REPORT_PATH):
    with open(HDFS_REPORT_PATH, "r") as file:
        hdfs_report = json.load(file)
else:
    hdfs_report = None

# Predict probability for all customers
features = df.drop(columns=["CustomerID", "Churn"])
df["ChurnProbability"] = model.predict_proba(features)[:, 1]
df["PredictedChurn"] = model.predict(features)

df["RiskLevel"] = pd.cut(
    df["ChurnProbability"],
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low", "Medium", "High"]
)

# -------------------------------------------
# HEADER
# -------------------------------------------
st.title("📊 CRM Churn Intelligence")
st.markdown(
    "### Big Data Classification & Prediction for Customer Relationship Management"
)
st.caption(
    "Customer analytics • Machine learning • Churn prediction • Retention intelligence"
)

st.divider()

# -------------------------------------------
# SIDEBAR
# -------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

risk_filter = st.sidebar.multiselect(
    "Filter by Risk Level",
    ["Low", "Medium", "High"],
    default=["Low", "Medium", "High"]
)

contract_filter = st.sidebar.multiselect(
    "Filter by Contract Type",
    df["ContractType"].unique().tolist(),
    default=df["ContractType"].unique().tolist()
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Storage Status")
st.sidebar.success("Customer CSV loaded")
st.sidebar.info("HDFS: Simulation mode")
st.sidebar.caption(
    "This prototype uses local CSV storage. "
    "A real HDFS cluster is not connected."
)

# -------------------------------------------
# KPI METRICS
# -------------------------------------------
total_customers = len(df)
actual_churn = int(df["Churn"].sum())
actual_retained = total_customers - actual_churn
actual_churn_rate = actual_churn / total_customers * 100

high_risk_count = int((df["RiskLevel"] == "High").sum())
predicted_churn_count = int(df["PredictedChurn"].sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Customers", f"{total_customers:,}")
col2.metric("🚨 Actual Churned", f"{actual_churn:,}", f"{actual_churn_rate:.1f}% churn")
col3.metric("💚 Retained Customers", f"{actual_retained:,}")
col4.metric("⚠️ High Risk Customers", f"{high_risk_count:,}")

st.divider()

# -------------------------------------------
# TABS
# -------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "👥 Customer Analytics",
    "🔮 Churn Prediction",
    "🤖 Model Performance",
    "🗄️ Big Data Storage"
])

# -------------------------------------------
# TAB 1 - OVERVIEW
# -------------------------------------------
with tab1:
    st.subheader("Business Overview")

    left, right = st.columns(2)

    with left:
        churn_data = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [actual_retained, actual_churn]
        })

        fig = px.pie(
            churn_data,
            names="Status",
            values="Customers",
            hole=0.55,
            title="Actual Customer Churn Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_data = df["RiskLevel"].value_counts().reset_index()
        risk_data.columns = ["Risk Level", "Customers"]

        fig = px.bar(
            risk_data,
            x="Risk Level",
            y="Customers",
            color="Risk Level",
            title="Customer Risk Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Customers at Risk")

    risky_customers = df.sort_values(
        "ChurnProbability",
        ascending=False
    ).head(10)

    st.dataframe(
        risky_customers[
            [
                "CustomerID",
                "Tenure",
                "MonthlyCharges",
                "Complaints",
                "SupportCalls",
                "ContractType",
                "ChurnProbability",
                "RiskLevel"
            ]
        ],
        use_container_width=True
    )

# -------------------------------------------
# TAB 2 - CUSTOMER ANALYTICS
# -------------------------------------------
with tab2:
    st.subheader("Customer Analytics")

    filtered_df = df[
        df["RiskLevel"].astype(str).isin(risk_filter)
        & df["ContractType"].isin(contract_filter)
    ]

    st.write(f"Showing {len(filtered_df)} customers")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered_df,
            x="Tenure",
            color="RiskLevel",
            title="Customer Tenure Distribution",
            nbins=20
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered_df,
            x="ContractType",
            y="MonthlyCharges",
            color="RiskLevel",
            title="Monthly Charges by Contract Type"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Data Explorer")

    search_id = st.text_input("Search Customer ID", "")

    if search_id:
        filtered_df = filtered_df[
            filtered_df["CustomerID"].str.contains(
                search_id, case=False, na=False
            )
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download filtered customer data",
        data=csv_data,
        file_name="crm_filtered_customers.csv",
        mime="text/csv"
    )

# -------------------------------------------
# TAB 3 - PREDICTION
# -------------------------------------------
with tab3:
    st.subheader("Predict Customer Churn")
    st.write("Enter customer information to estimate churn probability.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age_input = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=30
            )

            tenure_input = st.number_input(
                "Tenure (months)",
                min_value=1,
                max_value=120,
                value=12
            )

            monthly_input = st.number_input(
                "Monthly Charges (₹)",
                min_value=100,
                max_value=10000,
                value=799
            )

        with c2:
            complaints_input = st.number_input(
                "Number of Complaints",
                min_value=0,
                max_value=30,
                value=1
            )

            support_input = st.number_input(
                "Support Calls",
                min_value=0,
                max_value=30,
                value=2
            )

            contract_input = st.selectbox(
                "Contract Type",
                ["Monthly", "Yearly", "Two-Year"]
            )

        with c3:
            payment_input = st.selectbox(
                "Payment Method",
                ["UPI", "Credit Card", "Debit Card", "Cash"]
            )

            internet_input = st.number_input(
                "Internet Usage (GB)",
                min_value=0,
                max_value=1000,
                value=50
            )

        submitted = st.form_submit_button("🔍 Predict Churn")

    if submitted:
        new_customer = pd.DataFrame([{
            "Age": age_input,
            "Tenure": tenure_input,
            "MonthlyCharges": monthly_input,
            "Complaints": complaints_input,
            "SupportCalls": support_input,
            "ContractType": contract_input,
            "PaymentMethod": payment_input,
            "InternetUsageGB": internet_input
        }])

        probability = model.predict_proba(new_customer)[0][1]
        prediction = model.predict(new_customer)[0]

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        if prediction == 1:
            col2.error("⚠️ Prediction: Likely to Churn")
        else:
            col2.success("✅ Prediction: Likely to Stay")

        if probability >= 0.60:
            st.error("🔴 HIGH RISK — Immediate retention action recommended.")
            recommendation = (
                "Contact the customer immediately, investigate complaints, "
                "and offer a suitable retention plan."
            )
        elif probability >= 0.30:
            st.warning("🟠 MEDIUM RISK — Customer needs monitoring.")
            recommendation = (
                "Monitor customer activity and consider a loyalty offer "
                "or service improvement."
            )
        else:
            st.success("🟢 LOW RISK — Customer appears relatively stable.")
            recommendation = (
                "Continue normal engagement and maintain service quality."
            )

        st.info(f"💡 Recommended Action: {recommendation}")

        st.progress(float(probability))

# -------------------------------------------
# TAB 4 - MODEL PERFORMANCE
# -------------------------------------------
with tab4:
    st.subheader("Machine Learning Model Evaluation")
    st.caption("Metrics calculated using the held-out test dataset.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    c4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")

    st.markdown("### Training Information")

    c1, c2 = st.columns(2)

    c1.metric("Training Samples", metrics["train_samples"])
    c2.metric("Testing Samples", metrics["test_samples"])

    st.markdown("### Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])

    fig = px.imshow(
        cm,
        text_auto=True,
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count"
        },
        x=["Stay (0)", "Churn (1)"],
        y=["Stay (0)", "Churn (1)"],
        title="Confusion Matrix",
        color_continuous_scale="Blues"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Metric explanations:**

    - **Accuracy:** Overall percentage of correct predictions.
    - **Precision:** Among customers predicted to churn, how many actually churned?
    - **Recall:** Among customers who actually churned, how many were detected?
    - **F1 Score:** Balance between precision and recall.
    """)

# -------------------------------------------
# FOOTER
# -------------------------------------------
st.divider()

st.caption(
    "CRM Churn Intelligence Prototype | "
    "Machine Learning Classification | "
    "HDFS Storage Simulation | Academic Project"
)
# -------------------------------------------
# TAB 5 - BIG DATA STORAGE
# -------------------------------------------
with tab5:
    st.subheader("Hadoop HDFS Storage Architecture")
    st.caption(
        "Local HDFS concept simulation — "
        "not connected to a real Hadoop cluster"
    )

    if hdfs_report is None:
        st.error("Storage report not found.")
        st.info("Run python hdfs_simulation.py first.")

    else:
        # Storage summary metrics
        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Storage Status",
            hdfs_report["storage_status"]
        )

        s2.metric(
            "Total Records",
            f"{hdfs_report['total_records']:,}"
        )

        s3.metric(
            "Data Blocks",
            hdfs_report["total_blocks"]
        )

        s4.metric(
            "Replication Factor",
            hdfs_report["replication_factor"]
        )

        st.divider()

        # Additional storage information
        s1, s2 = st.columns(2)

        s1.metric(
            "Replica Copies",
            hdfs_report["simulated_replica_copies"]
        )

        s2.metric(
            "Block Size",
            f"{hdfs_report['block_size_rows']} rows"
        )

        # Block chart
        block_df = pd.DataFrame(hdfs_report["blocks"])

        st.subheader("Data Block Distribution")

        if not block_df.empty:
            fig = px.bar(
                block_df,
                x="block_name",
                y="rows",
                title="Customer Records Across Data Blocks",
                labels={
                    "block_name": "Data Block",
                    "rows": "Number of Records"
                },
                text="rows"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Block table
        st.subheader("Block Information")

        if not block_df.empty:
            st.dataframe(
                block_df[[
                    "block_name",
                    "rows",
                    "original_location"
                ]],
                use_container_width=True
            )

        # Architecture
        st.subheader("Storage Architecture")

        st.markdown("""
        **Dataset → Block Division → Replication → Storage Nodes**

        - **Dataset:** Customer records in CSV format.
        - **Blocks:** Dataset divided into smaller sections.
        - **Replication:** Copies of each block are created.
        - **Storage nodes:** Replica folders simulate different nodes.
        - **Fault tolerance concept:** Multiple copies provide redundancy.

        ⚠️ This prototype simulates HDFS concepts using local folders
        and file copies. It does not implement real Hadoop NameNode,
        DataNode, distributed networking, or HDFS fault tolerance.
        """)

        st.success(
            "Storage simulation report successfully loaded."
        )