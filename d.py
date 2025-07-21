import sys
import datetime as dt
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
import streamlit as st
import sqlalchemy as sa
from sqlalchemy.engine.base import Engine
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt
import tempfile
import xlsxwriter
from fpdf import FPDF
from functools import lru_cache
# Make full background white and fix text/input color
st.markdown("""
    <style>
        /* Set full background to white */
        .stApp {
            background-color: white;
        }

        /* Force sidebar also white */
        section[data-testid="stSidebar"] {
            background-color: white !important;
        }

        /* Set inputs, text, button color */
        .stTextInput>div>div>input,
        .stTextArea>div>textarea,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>div,
        .stButton>button {
            background-color: white !important;
            color: black !important;
        }

        /* Make labels black for better visibility */
        label, .stTextInput label, .stNumberInput label, .stSelectbox label {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* Set background and text color */
        body, .main, section[data-testid="stSidebar"] {
            background-color: white !important;
            color: black !important;
        }

        /* Ensure all text in sidebar is visible */
        section[data-testid="stSidebar"] * {
            color: black !important;
        }

        /* Button styling */
        button {
            color: black !important;
            background-color: #f0f0f0 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
        }

        /* Force default text color for Streamlit text blocks */
        .css-1cpxqw2, .css-10trblm {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        /* Main background and text */
        body {
            background-color: white;
            color: black;
        }

        /* Streamlit main content background */
        .main {
            background-color: white;
            color: black;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: white;
            color: black;
        }

        /* Change default text color for all elements inside sidebar */
        section[data-testid="stSidebar"] * {
            color: black !important;
        }

        /* Adjust button text and background */
        button {
            color: black !important;
        }

        /* Ensure markdown text is visible */
        .css-1cpxqw2, .css-10trblm {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

CSV_TABLES = [
    "users", "orders", "order_details", "products", "payments",
    "leads", "deals", "activities"
]
csv_data: Dict[str, pd.DataFrame] = {}
_DB_KEY = "db_engine"
st.set_page_config(
    page_title="📊 SME Sales Management Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* Global font and colors */
    .css-1d391kg {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
    }
    h1, h2, h3 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Advanced Sales Management Dashboard for SMEs")
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("---")

st.sidebar.header("Select Data Source")
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Database", "CSV only"),
    horizontal=True)

if data_source == "Database":
    st.sidebar.header("Database Settings")
    db_type = st.sidebar.selectbox("Database Type", ["MySQL", "SQLite"], key="mysql_host_3")
    st.sidebar.markdown(" ")
    if db_type == "MySQL":
        mysql_host = st.sidebar.text_input("MySQL Host", "localhost")
        mysql_db = st.sidebar.text_input("MySQL Database", "sales")
        mysql_user = st.sidebar.text_input("MySQL User", "root")
        mysql_pass = st.sidebar.text_input("MySQL Password", type="password")
    else:
        db_path = st.sidebar.text_input("SQLite DB Path", "sales.db")
    st.sidebar.markdown("---")
    if st.sidebar.button("Connect to Database"):
        # call your connect function here
        pass

elif data_source == "CSV only":
    st.sidebar.header("Upload CSV files")
    for tbl in CSV_TABLES:
        st.sidebar.file_uploader(f"Upload {tbl}.csv", type="csv")

if data_source == "Database":
    st.sidebar.header("1. Database Selection")
    db_type = st.sidebar.selectbox("Select Database Type", ["SQLite", "MySQL"], key="mysql_host_2")
    
    if db_type == "SQLite":
        db_path = st.sidebar.text_input("SQLite DB Path", "sales.db", key="sqlite_db_path")
        db_uri = f"sqlite:///{db_path}"
    else:
        mysql_user = st.sidebar.text_input("MySQL Username", "root", key="mysql_user")
        mysql_pass = st.sidebar.text_input("MySQL Password", type="password", key="mysql_pass")
        mysql_host = st.sidebar.text_input("MySQL Host", "localhost", key="mysql_host")
        mysql_port = st.sidebar.text_input("MySQL Port", "3306", key="mysql_port")
        mysql_db   = st.sidebar.text_input("MySQL Database", "sales", key="mysql_db_name")
        db_uri = f"mysql+pymysql://{mysql_user}:{mysql_pass}@{mysql_host}:{mysql_port}/{mysql_db}"
    connect_clicked = st.sidebar.button("Connect to Database", key="connect_db_btn")
    disconnect_clicked = st.sidebar.button("Disconnect DB", key="disconnect_db_btn")
else:
    st.sidebar.header("2. Upload Your CSVs")
    connect_clicked = disconnect_clicked = False
    db_uri = None
    db_type = None
def get_engine() -> Optional[Engine]:
    return st.session_state.get(_DB_KEY)
def set_engine(engine: Engine):
    st.session_state[_DB_KEY] = engine
def disconnect_db():
    if _DB_KEY in st.session_state:
        del st.session_state[_DB_KEY]
        st.success("✅ Disconnected from database.")
def test_and_connect_db(uri: str) -> Optional[Engine]:
    """Attempt to connect to DB. Returns engine instance if successful."""
    try:
        engine = sa.create_engine(uri, pool_pre_ping=True, pool_recycle=280)
        conn = engine.connect()
        conn.close()
        set_engine(engine)
        st.success("✅ Connected to DB.")
        return engine
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        return None
if connect_clicked and db_uri:
    test_and_connect_db(db_uri)
elif disconnect_clicked:
    disconnect_db()
if data_source == "CSV only":
    for tbl in CSV_TABLES:
        up = st.sidebar.file_uploader(f"Upload {tbl}.csv", type="csv", key=f"csv_{tbl}")
        if up:
            try:
                df = pd.read_csv(up)
                csv_data[tbl] = df
                st.sidebar.success(f"{tbl}.csv loaded ({len(df)} rows)")
            except Exception as e:
                st.sidebar.error(f"Error reading {tbl}.csv: {e}")
users = products = orders = order_details = payments = leads = deals = activities = customer_payments = pd.DataFrame()
if data_source == "Database" and get_engine():
    @st.cache_data(show_spinner="Loading all tables from database...")
    def load_data_from_db(_engine):
        tables = ["users", "orders", "order_details", "products", "payments", "leads", "deals", "activities", "customer_payments_view"]
        data = {}
        for t in tables:
            try:
                if sa.inspect(_engine).has_table(t):
                    data[t] = pd.read_sql(f"SELECT * FROM {t}", _engine)
                else:
                    data[t] = pd.DataFrame()
            except Exception as e:
                st.warning(f"Could not load '{t}': {e}")
                data[t] = pd.DataFrame()
        return data
    loaded_data = load_data_from_db(get_engine())
    users = loaded_data.get("users", pd.DataFrame())
    products = loaded_data.get("products", pd.DataFrame())
    orders = loaded_data.get("orders", pd.DataFrame())
    order_details = loaded_data.get("order_details", pd.DataFrame())
    payments = loaded_data.get("payments", pd.DataFrame())
    leads = loaded_data.get("leads", pd.DataFrame())
    deals = loaded_data.get("deals", pd.DataFrame())
    activities = loaded_data.get("activities", pd.DataFrame())
    customer_payments = loaded_data.get("customer_payments_view", pd.DataFrame())
    csv_data = {}
elif data_source == "CSV only":
    users = csv_data.get("users", pd.DataFrame())
    products = csv_data.get("products", pd.DataFrame())
    orders = csv_data.get("orders", pd.DataFrame())
    order_details = csv_data.get("order_details", pd.DataFrame())
    payments = csv_data.get("payments", pd.DataFrame())
    leads = csv_data.get("leads", pd.DataFrame())
    deals = csv_data.get("deals", pd.DataFrame())
    activities = csv_data.get("activities", pd.DataFrame())
    customer_payments = csv_data.get("customer_payments", pd.DataFrame())
@lru_cache(maxsize=None)
def get_engine(db_type, db_path=None, mysql_user=None, mysql_pass=None, mysql_host=None, mysql_db=None):
    """
    Build and cache a DB engine using inputs from the Streamlit sidebar.
    This function is cached to avoid reconnecting on every script rerun.
    """
    uri = None
    if db_type == "SQLite":
        if not db_path:
            st.sidebar.warning("Please provide a path for the SQLite DB.")
            return None
        uri = f"sqlite:///{db_path}"
    elif db_type == "MySQL":
        if not all([mysql_user, mysql_host, mysql_db]):
            st.sidebar.warning("Please fill in all MySQL connection details.")
            return None
        uri = f"mysql+pymysql://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}"
    if uri:
        try:
            engine = sa.create_engine(uri, pool_pre_ping=True, pool_recycle=280)
            with engine.connect() as connection:
                st.sidebar.success(f"✅ Connected to {db_type} DB!")
            return engine
        except Exception as e:
            st.sidebar.error(f"❌ {db_type} connection failed: {e}")
            return None
    return None
def merge_price_into_order_details(od: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """Ensure order_details has a 'price_each' column by merging with products."""
    if od.empty:
        return od
    if 'price_each' in od.columns:
        return od
    possible_price_cols = ['price', 'unit_price', 'unitPrice', 'unitprice', 'price_each']
    price_col = next((col for col in possible_price_cols if col in prod.columns), None)
    if not price_col:
        st.warning("⚠️ Could not find a price column in 'products'. Revenue cannot be calculated.")
        od['price_each'] = np.nan
        return od
    if 'product_id' in od.columns and 'product_id' in prod.columns:
        # Ensure keys are the same type for merging
        od['product_id'] = od['product_id'].astype(str)
        prod['product_id'] = prod['product_id'].astype(str)
        od = pd.merge(
            od,
            prod[['product_id', price_col]].rename(columns={price_col: 'price_each'}),
            on='product_id',
            how='left')
        st.write("✓ 'price_each' successfully merged into order details.")
    else:
        st.warning("⚠️ 'product_id' missing in 'order_details' or 'products'. Cannot merge price.")
        od['price_each'] = np.nan
    return od
def _coerce_key_cols(df_left, df_right, left_key='customer_id', right_key='user_id', dtype='str'):
    """Ensures key columns have the same dtype before a merge."""
    if left_key in df_left.columns and right_key in df_right.columns:
        df_left[left_key] = df_left[left_key].astype(dtype)
        df_right[right_key] = df_right[right_key].astype(dtype)
    return df_left, df_right
def safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Safely convert a column to datetime, coercing errors."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df
def kpi_leads_vs_target(leads_df, annual_target: int):
    """Calculate lead count, target, and achievement percentage for the current year."""
    if 'created_date' not in leads_df.columns or leads_df.empty:
        return 0, annual_target, 0
    leads_df['created_date'] = pd.to_datetime(leads_df['created_date'], errors='coerce')
    this_year = pd.Timestamp("now").year
    actual = leads_df[leads_df['created_date'].dt.year == this_year].shape[0]
    pct = (actual / annual_target) * 100 if annual_target > 0 else 0
    return actual, annual_target, pct
def kpi_card(value, label, delta=None, prefix="", help_text=""):
    """Render a single KPI metric card."""
    st.metric(label, f"{prefix}{value:,.0f}", delta=delta, help=help_text)
st.sidebar.title("⚙️ Configuration")
data_source = st.sidebar.radio(
    "Select Data Source",
    ("Database", "CSV only"),
    horizontal=True,
    key="input_mode")
engine = None
if data_source == "Database":
    st.sidebar.header("Database Connection")
    db_type = st.sidebar.selectbox("Database Type", ["MySQL", "SQLite"], key="mysql_host_1")


    if db_type == "MySQL":
        mysql_host = st.sidebar.text_input("MySQL Host", "localhost", key="mysql_host_5")
        mysql_db = st.sidebar.text_input("MySQL Database Name", "sales", key="mysql_host_6")
        mysql_user = st.sidebar.text_input("MySQL Username", "root", key="mysql_host_7")
        mysql_pass = st.sidebar.text_input("MySQL Password", type="password", key="mysql_host_4")
        engine = get_engine("MySQL", mysql_host=mysql_host, mysql_db=mysql_db, mysql_user=mysql_user, mysql_pass=mysql_pass)
    elif db_type == "SQLite":
        db_path = st.sidebar.text_input("SQLite DB Path", "sales.db")
        engine = get_engine("SQLite", db_path=db_path)
elif data_source == "CSV only":
    st.sidebar.header("Upload CSV Files")
    for tbl in CSV_TABLES:
        up = st.sidebar.file_uploader(f"Upload {tbl}.csv", type="csv", key=f"csv_{tbl}")
        if up:
            try:
                csv_data[tbl] = pd.read_csv(up)
                st.sidebar.success(f"✓ Loaded {tbl}.csv ({len(csv_data[tbl])} rows)")
            except Exception as e:
                st.sidebar.error(f"Error reading {tbl}.csv: {e}")
users = pd.DataFrame()
products = pd.DataFrame()
orders = pd.DataFrame()
order_details = pd.DataFrame()
payments = pd.DataFrame()
leads = pd.DataFrame()
deals = pd.DataFrame()
activities = pd.DataFrame()
customer_payments = pd.DataFrame()
if data_source == "Database" and engine:
    @st.cache_data
    def load_data_from_db(_engine):
        data = {}
        inspector = sa.inspect(_engine)
        table_names = inspector.get_table_names()
        all_entities = table_names + ["customer_payments_view"]
        for table in all_entities:
            try:
                data[table] = pd.read_sql(f"SELECT * FROM {table}", _engine)
            except Exception:
                data[table] = pd.DataFrame()
        return data
    loaded_data = load_data_from_db(engine)
    payments = loaded_data.get("payments", pd.DataFrame())
    users = loaded_data.get("users", pd.DataFrame())
    orders = loaded_data.get("orders", pd.DataFrame())
    order_details = loaded_data.get("order_details", pd.DataFrame())
    products = loaded_data.get("products", pd.DataFrame())
    leads = loaded_data.get("leads", pd.DataFrame())
    deals = loaded_data.get("deals", pd.DataFrame())
    activities = loaded_data.get("activities", pd.DataFrame())
    customer_payments = loaded_data.get("customer_payments_view", pd.DataFrame())
elif data_source == "CSV only":
    users = csv_data.get("users", pd.DataFrame())
    orders = csv_data.get("orders", pd.DataFrame())
    order_details = csv_data.get("order_details", pd.DataFrame())
    products = csv_data.get("products", pd.DataFrame())
    payments = csv_data.get("payments", pd.DataFrame())
    leads = csv_data.get("leads", pd.DataFrame())
    deals = csv_data.get("deals", pd.DataFrame())
    activities = csv_data.get("activities", pd.DataFrame())
    if not payments.empty and not users.empty:
         customer_payments = pd.merge(payments, users, left_on='customer_id', right_on='user_id', how='left')
if all(df.empty for df in [users, orders, payments, products]):
    st.info("👋 Welcome! Please connect to a database or upload CSV files via the sidebar to begin.")
    st.stop()
st.header("🔄 Data Pre-processing Status")
with st.expander("Show Details"):
    st.write("Ensuring data types are correct for analysis and visualization.")
    payments = safe_to_datetime(payments, 'payment_date')
    orders = safe_to_datetime(orders, 'order_date')
    leads = safe_to_datetime(leads, 'created_date')
    deals = safe_to_datetime(deals, 'expected_close_date')
    activities = safe_to_datetime(activities, 'activity_date')
    if not customer_payments.empty:
        customer_payments = safe_to_datetime(customer_payments, 'payment_date')
    st.write("✓ Date columns converted.")
    for df, name in zip([orders, order_details, products, users, payments, customer_payments],
                        ["orders", "order_details", "products", "users", "payments", "customer_payments"]):
        for col in ['order_id', 'user_id', 'product_id', 'customer_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    st.write("✓ Key ID columns standardized to string type.")
    order_details = merge_price_into_order_details(order_details, products)
    if 'quantity' in order_details.columns and 'price_each' in order_details.columns:
        order_details['quantity'] = pd.to_numeric(order_details['quantity'], errors='coerce').fillna(0)
        order_details['price_each'] = pd.to_numeric(order_details['price_each'], errors='coerce').fillna(0)
        order_details['item_revenue'] = order_details['quantity'] * order_details['price_each']
        st.write("✓ 'item_revenue' calculated for order details.")
    else:
        order_details['item_revenue'] = 0.0
st.markdown("---")
st.header("🏆 Key Performance Indicators")
total_revenue = payments['amount'].sum() if 'amount' in payments.columns else 0
total_orders = orders['order_id'].nunique() if 'order_id' in orders.columns else 0
total_customers = users['user_id'].nunique() if 'user_id' in users.columns else 0
revenue_this_month, rev_delta = (0, None)
if 'payment_date' in payments.columns and not payments.empty:
    current_month_mask = payments['payment_date'].dt.to_period('M') == pd.Timestamp.today().to_period('M')
    last_month_mask = payments['payment_date'].dt.to_period('M') == (pd.Timestamp.today() - pd.DateOffset(months=1)).to_period('M')
    revenue_this_month = payments.loc[current_month_mask, 'amount'].sum()
    revenue_last_month = payments.loc[last_month_mask, 'amount'].sum()
    if revenue_last_month > 0:
        rev_delta = f"{((revenue_this_month - revenue_last_month) / revenue_last_month * 100):.0f}%"
lead_this_month, lead_delta = (0, None)
if 'created_date' in leads.columns and not leads.empty:
    current_month_mask = leads['created_date'].dt.to_period('M') == pd.Timestamp.today().to_period('M')
    last_month_mask = leads['created_date'].dt.to_period('M') == (pd.Timestamp.today() - pd.DateOffset(months=1)).to_period('M')
    lead_this_month = current_month_mask.sum()
    lead_last_month = last_month_mask.sum()
    if lead_last_month > 0:
        lead_delta = f"{(lead_this_month - lead_last_month) / lead_last_month * 100:.0f}% vs last month"
pipeline_deals = deals[deals['stage'] != 'Closed Won'].shape[0] if 'stage' in deals.columns else 0
kpi_cols = st.columns(4)

def colored_metric(col, label, value, delta=None, color="#2ECC71"):
    col.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;">
            <h3>{label}</h3>
            <p style='font-size:24px'>{value}</p>
            {f"<small>Δ {delta}</small>" if delta else ""}
        </div>
        """, unsafe_allow_html=True)

colored_metric(kpi_cols[0], "Leads This Month", f"{lead_this_month:,}", lead_delta, "#2ECC71")
colored_metric(kpi_cols[1], "Revenue This Month", f"MMK {revenue_this_month:,.0f}", rev_delta, "#5DADE2")
colored_metric(kpi_cols[2], "Deals in Pipeline", f"{pipeline_deals:,}", color="#F39C12")
colored_metric(kpi_cols[3], "Active Customers", f"{total_customers:,}", color="#9B59B6")

st.write("")
glob_cols = st.columns(4)
glob_cols[0].metric("💰 Total Lifetime Revenue", f"MMK {total_revenue:,.0f}")
glob_cols[1].metric("🛒 Total Orders", f"{total_orders:,}")
glob_cols[2].metric("📦 Total Products", f"{products['product_id'].nunique() if not products.empty else 0:,}")
glob_cols[3].metric("👥 Total Customers", f"{total_customers:,}")
st.markdown("---")
st.header("📈 Revenue Analytics")
if 'payment_date' in payments.columns and 'amount' in payments.columns and not payments.empty:
    payments['month'] = payments['payment_date'].dt.to_period('M').astype(str)
    monthly_rev = payments.groupby('month')['amount'].sum().reset_index()
    monthly_rev = monthly_rev.sort_values('month')
    tab_trend, tab_forecast = st.tabs(["Monthly Trend", "3-Month Forecast"])
    with tab_trend:
        fig_month = px.line(monthly_rev, x='month', y='amount', title="Actual Monthly Revenue",
                            markers=True, labels={'amount': 'Revenue (MMK)'})
        st.plotly_chart(fig_month, use_container_width=True)
    with tab_forecast:
        if len(monthly_rev) >= 3:
            monthly_rev['idx'] = np.arange(len(monthly_rev))
            model = LinearRegression().fit(monthly_rev[['idx']], monthly_rev['amount'])

            future_idx = range(len(monthly_rev), len(monthly_rev) + 3)
            future_amount = model.predict(np.array(future_idx).reshape(-1, 1))
            last_month = pd.to_datetime(monthly_rev['month'].iloc[-1])
            future_months = [(last_month + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(3)]
            future_df = pd.DataFrame({'month': future_months, 'amount': future_amount})
            combined = pd.concat([monthly_rev[['month', 'amount']], future_df])
            fig_fore = px.line(combined, x='month', y='amount', title="Actual + 3-Month Forecast", markers=True,
                               labels={'amount': 'Revenue (MMK)'})
            fig_fore.add_vline(x=monthly_rev['month'].iloc[-1], line_width=2, line_dash="dash", line_color="green")
            st.plotly_chart(fig_fore, use_container_width=True)
        else:
            st.info("📈 Need at least 3 months of revenue data for forecasting.")
else:
 st.info("📈 `payments` table with `payment_date` and `amount` is required for revenue analytics.")
st.markdown("---")
st.header("🎯 Target vs Actual")
cols_targets = st.columns(2)
with cols_targets[0]:
    if not leads.empty:
        target_leads = st.number_input("Annual Lead Target", min_value=1, value=5000, step=100)
        actual, target, pct = kpi_leads_vs_target(leads, annual_target=target_leads)    
        fig_lead = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=actual,
            title={'text': 'Lead Generation (YTD)'},
            delta={'reference': target},
            gauge={'axis': {'range': [None, target]}, 'bar': {'color': '#2ECC71'}}))
        st.plotly_chart(fig_lead, use_container_width=True)
    else:
       st.info("🎯 `leads` table is required for this chart.")
with cols_targets[1]:
    if not payments.empty:
        target_rev = st.number_input("Annual Revenue Target (MMK)", min_value=1000, value=50_000_000, step=100_000)
        rev_ytd = payments[payments['payment_date'].dt.year == dt.datetime.today().year]['amount'].sum()
        fig_bullet = go.Figure(go.Indicator(
            mode="number+gauge+delta", value=rev_ytd,
            title={'text': "Revenue (YTD)"},
            gauge={'shape': "bullet", 'axis': {'range': [None, target_rev]}, 'bar': {'color': '#5DADE2'}},
            delta={'reference': target_rev, 'position': "bottom"}))
        st.plotly_chart(fig_bullet, use_container_width=True)
    else:
        st.info("🎯 `payments` table is required for this chart.")
st.markdown("---")
st.header("🤖 Customer Segmentation (RFM Lite)")
if not customer_payments.empty and all(c in customer_payments.columns for c in ['customer_id', 'payment_date', 'amount']):
    if 'payment_id' not in customer_payments.columns:
        customer_payments['payment_id'] = customer_payments.index
    now = dt.datetime.now(customer_payments['payment_date'].dt.tz) if customer_payments['payment_date'].dt.tz else dt.datetime.now()
    seg_data = customer_payments.groupby('customer_id').agg(
        recency_days=('payment_date', lambda x: (now - x.max()).days),
        frequency=('payment_id', 'count'),
        monetary=('amount', 'sum')
    ).reset_index()
    use_cols = st.multiselect("Select features for clustering",
                              ['recency_days', 'frequency', 'monetary'],
                              default=['frequency', 'monetary'])
    if len(use_cols) >= 2:
        X_scaled = StandardScaler().fit_transform(seg_data[use_cols])
        k = st.slider("Number of clusters (K)", 2, 6, 3, key="kmeans_k")
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        seg_data['segment'] = km.fit_predict(X_scaled)
        if not users.empty:
            seg_data, users = _coerce_key_cols(seg_data, users, left_key='customer_id', right_key='user_id')
            seg_data = seg_data.merge(
                users[['user_id', 'name']],
                left_on='customer_id', right_on='user_id',
                how='left'
            ).rename(columns={'name': 'customer_name'})
        fig_seg = px.scatter(
            seg_data, x=use_cols[0], y=use_cols[1],
            color=seg_data['segment'].astype(str),
            size='monetary',
            hover_data=['customer_name', 'recency_days', 'frequency', 'monetary'],
            title="Customer Segments")
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info("ℹ️ Select at least 2 dimensions for clustering.")
else:
    st.info("🤖 `payments` and `users` data are required for segmentation.")
st.markdown("---")
st.header("🏅 Product Performance")
if not order_details.empty and not products.empty and 'item_revenue' in order_details.columns:
    prod_perf = order_details.groupby('product_id').agg(
        qty_sold=('quantity', 'sum'),
        revenue=('item_revenue', 'sum')
    ).reset_index()
    prod_perf = prod_perf.merge(products[['product_id', 'name']], on='product_id', how='left')
    n_top = st.slider("Show Top N Products", 3, min(20, len(prod_perf) or 21), 10, key="top_n_prod")
    tab_qty, tab_rev = st.tabs(["By Quantity Sold", "By Revenue Generated"])
    with tab_qty:
        top_qty = prod_perf.sort_values('qty_sold', ascending=False).head(n_top)
        fig_q = px.bar(top_qty, x='name', y='qty_sold', color='qty_sold',
                       color_continuous_scale='Plasma', title=f"Top {n_top} Products by Quantity")
        st.plotly_chart(fig_q, use_container_width=True)
    with tab_rev:
        top_rev = prod_perf.sort_values('revenue', ascending=False).head(n_top)
        fig_r = px.bar(top_rev, x='name', y='revenue', color='revenue',
                       color_continuous_scale='Viridis', title=f"Top {n_top} Products by Revenue")
        st.plotly_chart(fig_r, use_container_width=True)
else:
    st.info("🏅 `order_details` and `products` tables are required for product performance.")
st.markdown("---")
st.header("🔻 Sales Funnel")
if not deals.empty and 'stage' in deals.columns:
    # Use a predefined funnel order, but only include stages present in the data
    funnel_order = ["Prospect", "Qualified", "Proposal", "Negotiation", "Won", "Lost"]
    existing_stages = [s for s in funnel_order if s in deals['stage'].unique()]
    stage_counts = deals['stage'].value_counts().reindex(existing_stages).reset_index()
    stage_counts.columns = ['stage', 'count']
    fig_funnel = go.Figure(go.Funnel(
        y=stage_counts['stage'],
        x=stage_counts['count'],
        textinfo="value+percent initial"))
    fig_funnel.update_layout(title_text="Deal Stage Funnel")
    st.plotly_chart(fig_funnel, use_container_width=True)
else:
    st.info("🔻 `deals` table with a `stage` column is required for funnel visualization.")
st.markdown("---")
st.header("🛠️ Advanced Data Tools")
def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data
col1, col2 = st.columns(2)
with col1:
    with st.expander("Export Raw Data to Excel"):
        export_choice = st.selectbox("Select data to export", options=CSV_TABLES)
        if export_choice:
            df_to_export = locals().get(export_choice)
            if df_to_export is not None and not df_to_export.empty:
                st.download_button(
                    f"Download {export_choice}.xlsx",
                    to_excel(df_to_export),
                    file_name=f"{export_choice}.xlsx")
            else:
                st.warning(f"No data available for '{export_choice}' to export.")
from fpdf import FPDF
def generate_pdf_report(summary: str, chart_fig=None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.set_text_color(0)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Sales Report Summary', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for line in summary.strip().split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.ln(5)
    if chart_fig:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            chart_fig.write_image(tmpfile.name)  # Plotly
            chart_path = tmpfile.name
        pdf.image(chart_path, w=180)
        os.remove(chart_path)
    return pdf.output(dest='S').encode('latin1')
st.markdown("<br>", unsafe_allow_html=True)
if st.button("📄 Generate Sales PDF Report"):

    summary = """Sales Report Summary
Total Lifetime Revenue: MMK 13,980
Total Orders: 26
Total Customers: 26"""
    df_grouped = df.groupby("payment_method")["amount"].sum().reset_index()
    fig = px.bar(
    df_grouped,
    x="payment_method",
    y="amount",
    title="Revenue by Payment Method",
    color="payment_method",  # add color by category
    color_discrete_sequence=px.colors.qualitative.Pastel  # pastel palette
     )
    fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family="Arial", size=14, color="black"),
    title_font=dict(size=20, family="Arial", color="darkblue"),
    legend_title_text="Payment Method"
    )
    st.plotly_chart(fig, use_container_width=True)

    pdf_bytes = generate_pdf_report(summary, chart_fig=fig)
    st.download_button(
        label="📄 Download Sales Report PDF",
        data=pdf_bytes,
        file_name="sales_report.pdf",
        mime="application/pdf")
with col2:
    with st.expander("Export Summary as PDF"):
        if st.button("📄 Generate PDF Report"):
            summary = f"Sales Report Summary\n\n"
            summary += f"Total Lifetime Revenue: MMK {total_revenue:,.0f}\n"
            summary += f"Total Orders: {total_orders:,}\n"
            summary += f"Total Customers: {total_customers:,}\n"
            fig = px.bar(df, x="payment_method", y="amount", title="Revenue by Payment Method")
            pdf_bytes = generate_pdf_report(summary, chart_fig=fig)
            st.download_button(
                "📄 Download PDF Report",
                pdf_bytes,
                file_name="summary_report.pdf",
                mime="application/pdf" )
