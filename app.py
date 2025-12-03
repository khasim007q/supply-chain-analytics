import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📦 Supply Chain Optimization Dashboard")
st.markdown("### Real-Time Analytics with Predictive Intelligence")
st.markdown("---")

# Load data function with caching
@st.cache_data
def load_data():
    try:
        kpi_summary = pd.read_csv('data/dashboards/kpi_summary.csv')
        product_performance = pd.read_csv('data/dashboards/product_performance.csv')
        supplier_performance = pd.read_csv('data/dashboards/supplier_performance.csv')
        warehouse_performance = pd.read_csv('data/dashboards/warehouse_performance.csv')
        monthly_trends = pd.read_csv('data/dashboards/monthly_trends.csv')
        risk_alerts = pd.read_csv('data/dashboards/risk_alerts.csv')
        action_items = pd.read_csv('data/dashboards/action_items.csv')
        
        return {
            'kpi': kpi_summary,
            'products': product_performance,
            'suppliers': supplier_performance,
            'warehouses': warehouse_performance,
            'trends': monthly_trends,
            'risks': risk_alerts,
            'actions': action_items
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load all data
data = load_data()

if data is None:
    st.error("⚠️ Unable to load dashboard data. Please check if CSV files exist in data/dashboards/")
    st.stop()

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["📊 Executive Overview", "📦 Product Analysis", "🏭 Warehouse Operations", 
     "🚚 Supplier Performance", "⚠️ Risk & Alerts", "📋 Action Items"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Project Info")
st.sidebar.info(f"""
**Project:** Supply Chain Optimization  
**Records Processed:** 43,780+  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}  
**Status:** ✅ Operational
""")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📊 Executive Overview":
    st.header("Executive KPI Dashboard")
    
    # Extract key metrics
    kpi_df = data['kpi']
    
    # Create 4 columns for top KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue = kpi_df[kpi_df['KPI_Name'] == 'Total Revenue']['Current_Value'].values[0]
        st.metric("💰 Total Revenue", revenue, delta="Growing")
    
    with col2:
        profit = kpi_df[kpi_df['KPI_Name'] == 'Total Profit']['Current_Value'].values[0]
        st.metric("💵 Total Profit", profit, delta="Growing")
    
    with col3:
        fulfillment = kpi_df[kpi_df['KPI_Name'] == 'Fulfillment Rate %']['Current_Value'].values[0]
        st.metric("✅ Fulfillment Rate", fulfillment, delta="Above Target")
    
    with col4:
        stockout = kpi_df[kpi_df['KPI_Name'] == 'Stockout Rate %']['Current_Value'].values[0]
        st.metric("⚠️ Stockout Rate", stockout, delta="-0.5%", delta_color="inverse")
    
    st.markdown("---")
    
    # KPI Categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Revenue & Operations KPIs")
        revenue_ops = kpi_df[kpi_df['KPI_Category'].isin(['Revenue', 'Operations'])]
        st.dataframe(
            revenue_ops[['KPI_Name', 'Current_Value', 'Target', 'Status']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📦 Inventory & Supplier KPIs")
        inv_supp = kpi_df[kpi_df['KPI_Category'].isin(['Inventory', 'Supplier'])]
        st.dataframe(
            inv_supp[['KPI_Name', 'Current_Value', 'Target', 'Status']],
            use_container_width=True,
            hide_index=True
        )
    
    # Monthly trends chart
    st.markdown("---")
    st.subheader("📈 Revenue Trend Analysis")
    
    trends_df = data['trends']
    trends_df['date'] = pd.to_datetime(trends_df[['year', 'month']].assign(day=1))
    
    fig = px.line(
        trends_df, 
        x='date', 
        y='revenue',
        title='Monthly Revenue Trend',
        labels={'revenue': 'Revenue ($)', 'date': 'Month'}
    )
    fig.update_traces(line_color='#1f77b4', line_width=3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: PRODUCT ANALYSIS
# ==========================================
elif page == "📦 Product Analysis":
    st.header("Product Performance Analysis")
    
    products_df = data['products']
    
    # Debug: Show column names
    # st.write("Available columns:", products_df.columns.tolist())
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(products_df))
    with col2:
        st.metric("Total Revenue", f"${products_df['total_revenue'].sum():,.2f}")
    with col3:
        st.metric("Total Profit", f"${products_df['total_profit'].sum():,.2f}")
    with col4:
        avg_margin = (products_df['total_profit'].sum() / products_df['total_revenue'].sum() * 100)
        st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
    
    st.markdown("---")
    
    # Category filter - FIX: Handle case-insensitive column names
    try:
        # Try to find the category column (case-insensitive)
        category_col = None
        for col in products_df.columns:
            if col.lower() == 'category':
                category_col = col
                break
        
        if category_col:
            categories = ['All'] + list(products_df[category_col].unique())
            selected_category = st.selectbox("Filter by Category", categories)
            
            if selected_category != 'All':
                filtered_products = products_df[products_df[category_col] == selected_category]
            else:
                filtered_products = products_df
        else:
            st.warning("⚠️ Category column not found. Showing all products.")
            filtered_products = products_df
            category_col = None
    except Exception as e:
        st.error(f"Error with category filter: {e}")
        filtered_products = products_df
        category_col = None
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Products by Revenue")
        top_products = filtered_products.nlargest(10, 'total_revenue')
        
        fig = px.bar(
            top_products,
            x='total_revenue',
            y='product_name',
            orientation='h',
            title='Revenue by Product',
            labels={'total_revenue': 'Revenue ($)', 'product_name': 'Product'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Revenue by Category")
        
        if category_col:
            category_revenue = products_df.groupby(category_col)['total_revenue'].sum().reset_index()
            
            fig = px.pie(
                category_revenue,
                values='total_revenue',
                names=category_col,
                title='Category Distribution'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category breakdown not available")
    
    # Product table
    st.markdown("---")
    st.subheader("📋 Detailed Product Performance")
    
    # Determine which columns to display
    base_cols = ['product_name', 'total_revenue', 'total_profit', 'stockout_rate']
    
    if category_col:
        display_cols = ['product_name', category_col, 'total_revenue', 'total_profit', 
                       'stockout_rate', 'risk_category']
    else:
        display_cols = ['product_name', 'total_revenue', 'total_profit', 
                       'stockout_rate', 'risk_category']
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in filtered_products.columns]
    
    display_df = filtered_products[display_cols].sort_values('total_revenue', ascending=False)
    
    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 3: WAREHOUSE OPERATIONS
# ==========================================
elif page == "🏭 Warehouse Operations":
    st.header("Warehouse Performance Dashboard")
    
    warehouse_df = data['warehouses']
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Warehouses", len(warehouse_df))
    with col2:
        st.metric("Total Revenue", f"${warehouse_df['total_revenue'].sum():,.2f}")
    with col3:
        avg_util = warehouse_df['capacity_utilization'].mean()
        st.metric("Avg Capacity Utilization", f"{avg_util:.1f}%")
    with col4:
        avg_stockout = warehouse_df['stockout_rate'].mean()
        st.metric("Avg Stockout Rate", f"{avg_stockout:.1%}")
    
    st.markdown("---")
    
    # Warehouse comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Revenue by Warehouse")
        fig = px.bar(
            warehouse_df.sort_values('total_revenue', ascending=False),
            x='warehouse_id',
            y='total_revenue',
            color='location',
            title='Warehouse Revenue Comparison'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Stockout Rate by Location")
        fig = px.scatter(
            warehouse_df,
            x='capacity_utilization',
            y='stockout_rate',
            size='total_revenue',
            color='location',
            hover_data=['warehouse_id'],
            title='Utilization vs Stockout Rate'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Warehouse table
    st.markdown("---")
    st.subheader("📋 Warehouse Details")
    st.dataframe(
        warehouse_df[[
            'warehouse_id', 'location', 'total_revenue', 'total_fulfilled',
            'capacity_utilization', 'stockout_rate'
        ]],
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 4: SUPPLIER PERFORMANCE
# ==========================================
elif page == "🚚 Supplier Performance":
    st.header("Supplier Performance Scorecard")
    
    supplier_df = data['suppliers']
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Suppliers", len(supplier_df))
    with col2:
        preferred = len(supplier_df[supplier_df['recommendation'] == 'Preferred'])
        st.metric("Preferred Suppliers", preferred)
    with col3:
        avg_otd = supplier_df['on_time_rate'].mean()
        st.metric("Avg On-Time Delivery", f"{avg_otd:.1%}")
    with col4:
        review = len(supplier_df[supplier_df['recommendation'] == 'Review Required'])
        st.metric("Suppliers Under Review", review)
    
    st.markdown("---")
    
    # Supplier performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Suppliers by Performance Score")
        top_suppliers = supplier_df.nlargest(10, 'performance_score')
        
        fig = px.bar(
            top_suppliers,
            x='performance_score',
            y='supplier_name',
            orientation='h',
            color='recommendation',
            title='Supplier Rankings'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 On-Time Delivery vs Reliability")
        fig = px.scatter(
            supplier_df,
            x='on_time_rate',
            y='reliability_score',
            size='total_value',
            color='recommendation',
            hover_data=['supplier_name'],
            title='Supplier Performance Matrix'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Supplier table
    st.markdown("---")
    st.subheader("📋 Supplier Details")
    
    # Filter by recommendation
    rec_filter = st.selectbox(
        "Filter by Recommendation",
        ['All', 'Preferred', 'Approved', 'Review Required']
    )
    
    if rec_filter != 'All':
        filtered_suppliers = supplier_df[supplier_df['recommendation'] == rec_filter]
    else:
        filtered_suppliers = supplier_df
    
    st.dataframe(
        filtered_suppliers[[
            'supplier_name', 'country', 'on_time_rate', 'reliability_score',
            'performance_score', 'recommendation'
        ]].sort_values('performance_score', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# PAGE 5: RISK & ALERTS
# ==========================================
elif page == "⚠️ Risk & Alerts":
    st.header("Risk Monitoring & Alerts")
    
    risk_df = data['risks']
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical = len(risk_df[risk_df['status'] == 'Critical'])
        st.metric("🔴 Critical", critical)
    with col2:
        warning = len(risk_df[risk_df['status'] == 'Warning'])
        st.metric("🟡 Warning", warning)
    with col3:
        normal = len(risk_df[risk_df['status'] == 'Normal'])
        st.metric("🟢 Normal", normal)
    with col4:
        high_risk = len(risk_df[risk_df['risk_category'] == 'High Risk'])
        st.metric("⚠️ High Risk Products", high_risk)
    
    st.markdown("---")
    
    # Risk visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Distribution")
        risk_counts = risk_df['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['risk_category', 'count']
        
        fig = px.pie(
            risk_counts,
            values='count',
            names='risk_category',
            title='Products by Risk Category',
            color='risk_category',
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'orange',
                'High Risk': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Risk Score Distribution")
        fig = px.histogram(
            risk_df,
            x='stockout_risk_score',
            nbins=20,
            title='Stockout Risk Score Distribution',
            labels={'stockout_risk_score': 'Risk Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # High risk products
    st.markdown("---")
    st.subheader("🔴 Critical & High Risk Products")
    
    high_risk_products = risk_df[
        (risk_df['risk_category'] == 'High Risk') | (risk_df['status'] == 'Critical')
    ].sort_values('stockout_risk_score', ascending=False)
    
    if len(high_risk_products) > 0:
        st.dataframe(
            high_risk_products[[
                'product_name', 'category', 'current_stock', 'days_of_stock',
                'stockout_risk_score', 'risk_category', 'status'
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ No critical risk products detected!")

# ==========================================
# PAGE 6: ACTION ITEMS
# ==========================================
elif page == "📋 Action Items":
    st.header("Priority Action Items")
    
    actions_df = data['actions']
    
    # Priority metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        critical_count = len(actions_df[actions_df['priority'] == 'CRITICAL'])
        st.metric("🔴 Critical Actions", critical_count)
    with col2:
        high_count = len(actions_df[actions_df['priority'] == 'HIGH'])
        st.metric("🟡 High Priority", high_count)
    with col3:
        medium_count = len(actions_df[actions_df['priority'] == 'MEDIUM'])
        st.metric("🟢 Medium Priority", medium_count)
    
    st.markdown("---")
    
    # Priority filter
    priority_filter = st.selectbox(
        "Filter by Priority",
        ['All', 'CRITICAL', 'HIGH', 'MEDIUM']
    )
    
    if priority_filter != 'All':
        filtered_actions = actions_df[actions_df['priority'] == priority_filter]
    else:
        filtered_actions = actions_df
    
    # Category filter
    categories = ['All'] + list(actions_df['category'].unique())
    category_filter = st.selectbox("Filter by Category", categories)
    
    if category_filter != 'All':
        filtered_actions = filtered_actions[filtered_actions['category'] == category_filter]
    
    # Action items display
    st.subheader(f"📋 Action Items ({len(filtered_actions)})")
    
    for idx, row in filtered_actions.iterrows():
        priority_color = {
            'CRITICAL': '🔴',
            'HIGH': '🟡',
            'MEDIUM': '🟢'
        }
        
        with st.expander(f"{priority_color[row['priority']]} {row['product_name']} - {row['category']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Action Required:** {row['action']}")
                st.markdown(f"**Impact:** {row['estimated_impact']}")
            
            with col2:
                st.metric("Priority", row['priority'])
                st.metric("Risk Score", f"{row['risk_score']:.2f}")
    
    st.markdown("---")
    st.info("💡 **Tip:** Address CRITICAL priority items immediately to prevent stockouts and revenue loss.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Supply Chain Analytics Dashboard</strong> | Built with Streamlit | 
        Data Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)
