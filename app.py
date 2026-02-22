import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Crime Hotspot Dashboard",
    layout="wide"
)

st.title("🚔 Crime Hotspot Analysis Dashboard")

# ================= LOAD DATA =================

df = pd.read_csv("crime_hotspots_dbscan_sample.csv")

# ================= SIDEBAR MENU =================

st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "📁 Project Info",
        "🧠 Model & Code Explanation"
    ]
)

if menu == "📊 Dashboard":

    # ================= KPI SECTION =================

    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Crimes", len(df))
    col2.metric("Total Clusters", df['Cluster'].nunique())
    col3.metric(
        "Most Dangerous Cluster",
        df['Cluster'].value_counts().idxmax()
    )


    # ================= SIDEBAR FILTERS =================

    st.sidebar.header("🔎 Filters")

    # Crime Type Filter
    crime_type = st.sidebar.selectbox(
        "Select Crime Type",
        sorted(df['Primary Type'].dropna().unique())
    )

    filtered_df = df[df['Primary Type'] == crime_type]


    # ================= TIME FILTER =================

    st.sidebar.header("⏰ Time Filters")

    selected_hour = st.sidebar.slider(
        "Select Hour Range",
        0, 23, (0, 23)
    )

    selected_month = st.sidebar.slider(
        "Select Month Range",
        1, 12, (1, 12)
    )

    time_filtered_df = filtered_df[
        (filtered_df['Hour'] >= selected_hour[0]) &
        (filtered_df['Hour'] <= selected_hour[1]) &
        (filtered_df['Month'] >= selected_month[0]) &
        (filtered_df['Month'] <= selected_month[1])
    ]


    # ================= CRIME DISTRIBUTION =================

    st.subheader("📊 Crime Distribution by Cluster")

    crime_counts = time_filtered_df['Cluster'].value_counts()

    st.bar_chart(crime_counts)


    # ================= DANGER ZONE RANKING =================

    st.subheader("🚨 Danger Zone Ranking")

    cluster_counts = (
        df['Cluster']
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = ['Cluster', 'Crime_Count']

    st.dataframe(cluster_counts)


    # ================= CLUSTER COLOR MAP =================

    st.subheader("🗺 Crime Hotspot Cluster Map")

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        time_filtered_df['Longitude'],
        time_filtered_df['Latitude'],
        c=time_filtered_df['Cluster'],
        cmap='tab10',
        s=5
    )

    legend1 = ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    ax.add_artist(legend1)

    ax.set_title("Crime Clusters")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    st.pyplot(fig)


    # ================= CRIME TREND BY HOUR =================

    st.subheader("📈 Crime Trend by Hour")

    hour_counts = (
        time_filtered_df['Hour']
        .value_counts()
        .sort_index()
    )

    fig2, ax2 = plt.subplots()

    ax2.plot(
        hour_counts.index,
        hour_counts.values
    )

    ax2.set_title("Crime Trend by Hour")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Crime Count")

    st.pyplot(fig2)


    # ================= DATA TABLE =================

    st.subheader("📄 Filtered Data Preview")

    st.dataframe(time_filtered_df.head(100))


# ================= PROJECT INFO =================

elif menu == "📁 Project Info":

    st.header("📊 Project Overview")

    # =========================
    # Problem Statement
    # =========================
    st.subheader("🧩 Problem Statement")

    st.info("""
    This project identifies **crime hotspot zones**
    using clustering techniques based on:

    • Geographic location  
    • Crime type  
    • Time patterns  
    • Arrest & domestic cases  
    """)

    st.divider()

    # =========================
    # Dataset Info
    # =========================
    st.subheader("🗂️ Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", "5 Lakh+")
    col2.metric("Total Features", "25")
    col3.metric("Dataset Type", "Crime Records")

    st.write("""
    Dataset includes:

    • Latitude & Longitude  
    • Crime category  
    • Arrest status  
    • Domestic crimes  
    • District & ward  
    • Time features (Hour / Day / Month)  
    """)

    st.divider()

    # =========================
    # Data Cleaning
    # =========================
    st.subheader("🧹 Data Cleaning Steps")

    with st.expander("View Preprocessing Steps"):

        st.success("""
        • Removed irrelevant columns  
        • Handled missing values  
        • Encoded categorical features  
        • Extracted date-time features  
        • Created weekend flag  
        • Feature scaling applied  
        """)

    st.divider()

    # =========================
    # Models Used
    # =========================
    st.subheader("🤖 Clustering Models Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### KMeans")
        st.write("""
        • Used for initial clustering  
        • Elbow method applied  
        • Silhouette score evaluated  
        """)

    with col2:
        st.markdown("### DBSCAN")
        st.write("""
        • Density-based clustering  
        • Detects crime hotspots  
        • Handles noise points  
        """)

    with col3:
        st.markdown("### Hierarchical")
        st.write("""
        • Dendrogram visualization  
        • Cluster hierarchy analysis  
        """)

    st.divider()

    # =========================
    # Best Model
    # =========================
    st.subheader("🏆 Final Model Selected")

    st.success("""
    **DBSCAN** selected as final model because:

    • Identifies dense crime zones  
    • Detects real hotspot clusters  
    • Handles outliers effectively  
    • Works best for geo-spatial data  
    """)

    st.divider()

    # =========================
    # Document Link
    # =========================
    st.subheader("📥 Project Documentation")

    st.link_button(
        "📄 Open Project Document",
        "https://docs.google.com/document/d/1GfUJ4xUR1J8fB9JoAzshBDGsioSy4UkOlGo_b5oqRM4/edit?tab=t.0"
    )

# ================= MODEL EXPLANATION =================

elif menu == "🧠 Model & Code Explanation":

    st.header("💻 About This Project Code")

    # =========================
    # Technologies Used
    # =========================
    st.subheader("🛠️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        • Python  
        • Pandas  
        • NumPy  
        """)

    with col2:
        st.info("""
        • Scikit-learn  
        • Matplotlib  
        • Streamlit   
        """)

    st.divider()

    # =========================
    # Workflow
    # =========================
    st.subheader("⚙️ Project Workflow")

    st.success("""
    1️⃣ Data Collection  
    2️⃣ Data Cleaning  
    3️⃣ Feature Engineering  
    4️⃣ Encoding  
    5️⃣ Feature Scaling  
    6️⃣ Clustering Model Training  
    7️⃣ Model Evaluation  
    8️⃣ Hotspot Detection  
    9️⃣ Dashboard Development  
    """)

    st.divider()

    # =========================
    # Model Evaluation
    # =========================
    st.subheader("📈 Model Evaluation")

    st.write("""
    • Elbow Method → Optimal K detection  
    • Silhouette Score → Cluster quality  
    • KNN Distance Graph → EPS selection  
    """)

    st.divider()

    # =========================
    # Deployment
    # =========================
    st.subheader("🚀 Deployment")

    st.write("""
    Clustering outputs were exported and
    integrated into the Streamlit dashboard
    for real-time hotspot visualization.
    """)

    st.divider()

    # =========================
    # Purpose
    # =========================
    st.subheader("🎯 Purpose of Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Analysis Type", "Clustering")

    with col2:
        st.metric("Goal", "Crime Hotspot Detection")

    st.warning("""
    • Identifies high-risk crime zones  
    • Helps police resource allocation  
    • Supports safety planning  
    """)

    # =========================
    # Dashboard Graph Explanation
    # =========================

    st.header("📊 Dashboard Graph Explanations")

    st.divider()

    # -------------------------
    # Crime Distribution
    # -------------------------
    st.subheader("###  Crime Distribution by Cluster")

    st.info("""
    This bar chart shows the number of crimes
    occurring in each identified cluster.

    Insights:

    • Helps compare crime volume across zones  
    • Identifies which clusters are most active  
    • Supports hotspot prioritization  
    """)

    st.divider()

    # -------------------------
    # Danger Zone Ranking
    # -------------------------
    st.markdown("### 🚨 Danger Zone Ranking")

    st.info("""
    Ranks clusters based on crime frequency.

    Insights:

    • Highlights top high-risk zones  
    • Useful for law enforcement planning  
    • Enables resource allocation decisions  
    """)

    st.divider()

    # -------------------------
    # Hotspot Map
    # -------------------------
    st.markdown("### 🗺 Crime Hotspot Cluster Map")

    st.info("""
    Geospatial visualization of crime locations
    colored by cluster.

    Insights:

    • Shows real crime concentration areas  
    • Detects dense hotspot regions  
    • Helps visualize spatial crime patterns  
    """)

    st.divider()

    # -------------------------
    # Time Trend
    # -------------------------
    st.markdown("### ⏰ Crime Trend by Hour")

    st.info("""
    Displays crime occurrence across different hours.

    Insights:

    • Identifies peak crime timings  
    • Helps night patrol planning  
    • Supports time-based risk analysis  
    """)

    st.divider()

    # -------------------------
    # Filtered Data
    # -------------------------
    st.markdown("### 📄 Filtered Data Preview")

    st.info("""
    Shows dataset after applying sidebar filters.

    Insights:

    • Enables detailed record inspection  
    • Supports cluster verification  
    • Helps analysts explore raw data  
    """)