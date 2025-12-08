import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="International Expansion Tool",
    page_icon="🌍",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Read CSV with proper multi-level headers
    df = pd.read_csv('Global_Regions.csv', header=[0, 1, 2])
    
    # Flatten the multi-level columns
    df.columns = ['_'.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip('_') for col in df.columns]
    
    # Clean up extra spaces in column names
    df.columns = df.columns.str.replace('  ', ' ').str.strip()
    
    # Rename columns for easier access
    rename_dict = {}
    for col in df.columns:
        # Market Segmentation columns
        if 'Market Segmentation' in col and 'Business Region' in col:
            rename_dict[col] = 'Business_Region'
        elif col == 'Market Region':
            rename_dict[col] = 'Market_Region'
        elif col == 'ISO2':
            rename_dict[col] = 'ISO2'
        elif col == 'Country':
            rename_dict[col] = 'Country'
        # Shopper metrics
        elif 'Online Shopping' in col and 'Number of People' in col:
            rename_dict[col] = 'Shoppers_Millions'
        # Spend metrics
        elif 'Total B2C Consumer Goods Spend' in col:
            rename_dict[col] = 'Spend_Billions'
        # Revenue per shopper
        elif 'Avg Annual Revenue Per Consumer Goods Ecommerce Shopper' in col:
            rename_dict[col] = 'Revenue_Per_Shopper'
        # Penetration
        elif 'Online Purchases vs Total Consumer Goods' in col and 'Percent' in col:
            rename_dict[col] = 'Penetration_Percent'
        # Product categories
        elif 'Estimated Annual Spend In Each Consumer Goods Ecommerce Category' in col and 'Fashion' in col:
            rename_dict[col] = 'Fashion'
        # Payment methods
        elif 'Share of 2024 B2C Ecommerce Transaction Value' in col and 'Mobile & Digital Wallets' in col:
            rename_dict[col] = 'Payment_Mobile_Wallets'
        elif col == 'Debit & Credit Cards':
            rename_dict[col] = 'Payment_Cards'
        elif col == 'Account-To-Account Transfers (A2A)':
            rename_dict[col] = 'Payment_A2A'
        elif col == 'Buy Now Pay Later Services':
            rename_dict[col] = 'Payment_BNPL'
        elif col == 'Other Payment Methods':
            rename_dict[col] = 'Payment_Other'
        # Purchase motivators
        elif 'Percentage of Internet Users Aged 16+' in col and 'Free Delivery' in col:
            rename_dict[col] = 'Motivator_Free_Delivery'
        elif col == 'Next Day Delivery':
            rename_dict[col] = 'Motivator_Next_Day'
        elif col == 'Coupons & Discounts':
            rename_dict[col] = 'Motivator_Coupons'
        elif col == 'Simple Online Checkout':
            rename_dict[col] = 'Motivator_Simple_Checkout'
        elif col == 'Customer Reviews':
            rename_dict[col] = 'Motivator_Reviews'
        elif col == 'Easy Returns Policy':
            rename_dict[col] = 'Motivator_Returns'
        elif col == 'Loyalty Points':
            rename_dict[col] = 'Motivator_Loyalty'
        elif col == 'Interest Free Payments':
            rename_dict[col] = 'Motivator_Interest_Free'
        elif col == 'Eco-Friendly Credentials':
            rename_dict[col] = 'Motivator_Eco_Friendly'
        elif col == 'Guest Checkout':
            rename_dict[col] = 'Motivator_Guest_Checkout'
        elif col == 'Click and Collect':
            rename_dict[col] = 'Motivator_Click_Collect'
    
    df.rename(columns=rename_dict, inplace=True)
    
    # Handle YoY Increase columns - they appear multiple times
    yoy_count = 0
    yoy_mapping = ['Shoppers_YoY', 'Spend_YoY', 'Revenue_YoY', 'Penetration_YoY']
    
    for i, col in enumerate(df.columns):
        if col == 'YoY Increase':
            if yoy_count < len(yoy_mapping):
                df.columns.values[i] = yoy_mapping[yoy_count]
                yoy_count += 1
    
    # Clean all numeric columns - remove $ signs and convert to float
    numeric_columns = ['Shoppers_Millions', 'Shoppers_YoY', 'Spend_Billions', 'Spend_YoY',
                      'Revenue_Per_Shopper', 'Revenue_YoY', 'Penetration_Percent', 'Penetration_YoY',
                      'Fashion', 'Food', 'Furniture', 'Electronics', 'Beverages', 
                      'Beauty & Personal Care', 'DIY & Hardware', 'Toys & Hobby', 
                      'Tobacco', 'Household Essentials', 'OTC Pharmaceuticals', 
                      'Eyewear', 'Physical Media',
                      'Payment_Mobile_Wallets', 'Payment_Cards', 'Payment_A2A', 'Payment_BNPL', 'Payment_Other',
                      'Motivator_Free_Delivery', 'Motivator_Next_Day', 'Motivator_Coupons',
                      'Motivator_Simple_Checkout', 'Motivator_Reviews', 'Motivator_Returns',
                      'Motivator_Loyalty', 'Motivator_Interest_Free', 'Motivator_Eco_Friendly',
                      'Motivator_Guest_Checkout', 'Motivator_Click_Collect']
    
    for col in numeric_columns:
        if col in df.columns:
            try:
                if isinstance(df[col], pd.Series):
                    df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                pass
    
    return df

df = load_data()

# Title
st.title("🌍 International E-commerce Market Expansion Tool")
st.markdown("**Analyze global e-commerce markets to identify expansion opportunities**")

# Sidebar filters
st.sidebar.title("🔍 Filters")

# Region filter
regions = ['All'] + sorted(df['Business_Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Business Region", regions)

# Country filter
if selected_region != 'All':
    available_countries = sorted(df[df['Business_Region'] == selected_region]['Country'].unique().tolist())
else:
    available_countries = sorted(df['Country'].unique().tolist())

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    available_countries,
    default=available_countries  # Show all countries by default
)

# Filter dataframe
if selected_region != 'All':
    df_filtered = df[df['Business_Region'] == selected_region].copy()
else:
    df_filtered = df.copy()

if selected_countries:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]

# Product category columns
category_cols = ['Fashion', 'Food', 'Furniture', 'Electronics', 'Beverages', 
                'Beauty & Personal Care', 'DIY & Hardware', 'Toys & Hobby', 
                'Tobacco', 'Household Essentials', 'OTC Pharmaceuticals', 
                'Eyewear', 'Physical Media']

# Payment method columns
payment_cols = ['Payment_Mobile_Wallets', 'Payment_Cards', 'Payment_A2A', 'Payment_BNPL', 'Payment_Other']

# Motivator columns
motivator_cols = ['Motivator_Free_Delivery', 'Motivator_Next_Day', 'Motivator_Coupons',
                 'Motivator_Simple_Checkout', 'Motivator_Reviews', 'Motivator_Returns',
                 'Motivator_Loyalty', 'Motivator_Interest_Free', 'Motivator_Eco_Friendly',
                 'Motivator_Guest_Checkout', 'Motivator_Click_Collect']

# Create tabs based on Row 2 categories
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Summary", 
    "🌐 Market Segmentation",
    "👥 Shopper Growth", 
    "💰 Revenue Metrics",
    "🛍️ Product Categories",
    "💳 Payment Methods",
    "⭐ Purchase Motivators"
])

# TAB 1: SUMMARY
with tab1:
    st.header("Market Overview Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_shoppers = df_filtered['Shoppers_Millions'].sum()
        st.metric("Total Shoppers", f"{total_shoppers:.1f}M")
    
    with col2:
        total_spend = df_filtered['Spend_Billions'].sum()
        st.metric("Total Market Size", f"${total_spend:.1f}B")
    
    with col3:
        avg_revenue = df_filtered['Revenue_Per_Shopper'].mean()
        st.metric("Avg Revenue/Shopper", f"${avg_revenue:.0f}")
    
    with col4:
        avg_growth = df_filtered['Spend_YoY'].mean()
        st.metric("Avg Growth Rate", f"{avg_growth:.1f}%")
    
    st.markdown("---")
    
    # Top markets table
    st.subheader("Top Markets by Total Spend")
    
    top_markets = df_filtered.nlargest(10, 'Spend_Billions')[['Country', 'Spend_Billions', 'Shoppers_Millions', 'Revenue_Per_Shopper', 'Spend_YoY']]
    top_markets_display = top_markets.copy()
    top_markets_display['Spend_Billions'] = top_markets_display['Spend_Billions'].apply(lambda x: f"${x:.1f}B")
    top_markets_display['Shoppers_Millions'] = top_markets_display['Shoppers_Millions'].apply(lambda x: f"{x:.1f}M")
    top_markets_display['Revenue_Per_Shopper'] = top_markets_display['Revenue_Per_Shopper'].apply(lambda x: f"${x:.0f}")
    top_markets_display['Spend_YoY'] = top_markets_display['Spend_YoY'].apply(lambda x: f"{x:.1f}%")
    top_markets_display.columns = ['Country', 'Market Size', 'Shoppers', 'AOV', 'Growth']
    
    st.dataframe(top_markets_display, use_container_width=True, hide_index=True)

# TAB 2: MARKET SEGMENTATION
with tab2:
    st.header("Market Segmentation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # By Business Region
        region_summary = df_filtered.groupby('Business_Region').agg({
            'Spend_Billions': 'sum',
            'Shoppers_Millions': 'sum',
            'Country': 'count'
        }).reset_index()
        region_summary.columns = ['Business Region', 'Total Spend ($B)', 'Total Shoppers (M)', 'Number of Countries']
        
        fig_region = px.bar(
            region_summary,
            x='Business Region',
            y='Total Spend ($B)',
            color='Business Region',
            text='Total Spend ($B)',
            title='Market Size by Business Region'
        )
        fig_region.update_traces(texttemplate='$%{text:.1f}B', textposition='outside')
        fig_region.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        # By Market Region
        if 'Market_Region' in df_filtered.columns:
            market_summary = df_filtered.groupby('Market_Region').agg({
                'Spend_Billions': 'sum',
                'Shoppers_Millions': 'sum'
            }).reset_index()
            
            fig_market = px.pie(
                market_summary,
                values='Spend_Billions',
                names='Market_Region',
                title='Market Share by Market Region'
            )
            fig_market.update_layout(height=400)
            st.plotly_chart(fig_market, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed country breakdown
    st.subheader("Country-Level Details")
    country_details = df_filtered[['Country', 'Business_Region', 'Market_Region', 'Spend_Billions', 'Shoppers_Millions', 'Revenue_Per_Shopper']].copy()
    country_details = country_details.sort_values('Spend_Billions', ascending=False)
    st.dataframe(country_details, use_container_width=True, hide_index=True)

# TAB 3: SHOPPER GROWTH
with tab3:
    st.header("E-commerce Shopper Growth Analysis")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_shoppers = df_filtered['Shoppers_Millions'].sum()
        st.metric("Total Shoppers", f"{total_shoppers:.1f}M")
    
    with col2:
        avg_growth = df_filtered['Shoppers_YoY'].mean()
        st.metric("Avg Shopper Growth", f"{avg_growth:.1f}%")
    
    with col3:
        fastest_growth = df_filtered.nlargest(1, 'Shoppers_YoY')
        if not fastest_growth.empty:
            st.metric("Fastest Growing", f"{fastest_growth.iloc[0]['Country']}", f"{fastest_growth.iloc[0]['Shoppers_YoY']:.1f}%")
    
    st.markdown("---")
    
    # Shopper base vs growth chart
    st.subheader("Shopper Base vs Growth Rate")
    
    fig_scatter = px.scatter(
        df_filtered,
        x='Shoppers_Millions',
        y='Shoppers_YoY',
        size='Spend_Billions',
        color='Business_Region',
        hover_name='Country',
        labels={
            'Shoppers_Millions': 'Current Shoppers (Millions)',
            'Shoppers_YoY': 'YoY Growth (%)',
            'Spend_Billions': 'Market Size ($B)'
        },
        title='Shopper Growth Opportunity Matrix'
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Top growth markets
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Markets by Shopper Base")
        top_shoppers = df_filtered.nlargest(10, 'Shoppers_Millions')[['Country', 'Shoppers_Millions', 'Shoppers_YoY']]
        
        fig_shoppers = px.bar(
            top_shoppers,
            x='Shoppers_Millions',
            y='Country',
            orientation='h',
            text='Shoppers_Millions',
            color='Shoppers_YoY',
            color_continuous_scale='RdYlGn',
            labels={'Shoppers_Millions': 'Shoppers (M)', 'Shoppers_YoY': 'Growth (%)'}
        )
        fig_shoppers.update_traces(texttemplate='%{text:.1f}M', textposition='outside')
        fig_shoppers.update_layout(height=400)
        st.plotly_chart(fig_shoppers, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Markets by Growth Rate")
        top_growth = df_filtered.nlargest(10, 'Shoppers_YoY')[['Country', 'Shoppers_YoY', 'Shoppers_Millions']]
        
        fig_growth = px.bar(
            top_growth,
            x='Shoppers_YoY',
            y='Country',
            orientation='h',
            text='Shoppers_YoY',
            color='Shoppers_Millions',
            color_continuous_scale='Blues',
            labels={'Shoppers_YoY': 'Growth (%)', 'Shoppers_Millions': 'Shoppers (M)'}
        )
        fig_growth.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_growth.update_layout(height=400)
        st.plotly_chart(fig_growth, use_container_width=True)

# TAB 4: REVENUE METRICS
with tab4:
    st.header("Revenue & Spend Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_spend = df_filtered['Spend_Billions'].sum()
        st.metric("Total Market Size", f"${total_spend:.1f}B")
    
    with col2:
        avg_spend_growth = df_filtered['Spend_YoY'].mean()
        st.metric("Avg Spend Growth", f"{avg_spend_growth:.1f}%")
    
    with col3:
        avg_aov = df_filtered['Revenue_Per_Shopper'].mean()
        st.metric("Avg AOV", f"${avg_aov:.0f}")
    
    with col4:
        avg_aov_growth = df_filtered['Revenue_YoY'].mean()
        st.metric("Avg AOV Growth", f"{avg_aov_growth:.1f}%")
    
    st.markdown("---")
    
    # Dual axis chart: Spend vs AOV
    st.subheader("Market Size & Average Order Value Comparison")
    
    df_sorted = df_filtered.nlargest(15, 'Spend_Billions')
    
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_dual.add_trace(
        go.Bar(
            name="Market Size",
            x=df_sorted['Country'],
            y=df_sorted['Spend_Billions'],
            marker_color='lightblue'
        ),
        secondary_y=False
    )
    
    fig_dual.add_trace(
        go.Scatter(
            name="AOV",
            x=df_sorted['Country'],
            y=df_sorted['Revenue_Per_Shopper'],
            mode='lines+markers',
            marker=dict(size=10, color='orange'),
            line=dict(width=3, color='orange')
        ),
        secondary_y=True
    )
    
    fig_dual.update_xaxes(title_text="Country")
    fig_dual.update_yaxes(title_text="Market Size ($B)", secondary_y=False)
    fig_dual.update_yaxes(title_text="AOV ($)", secondary_y=True)
    fig_dual.update_layout(height=500, title_text="Top 15 Markets: Size vs AOV")
    
    st.plotly_chart(fig_dual, use_container_width=True)
    
    st.markdown("---")
    
    # Growth comparison
    st.subheader("Growth Rate Comparison")
    
    df_growth = df_filtered[['Country', 'Spend_YoY', 'Revenue_YoY', 'Shoppers_YoY']].copy()
    df_growth = df_growth.nlargest(15, 'Spend_YoY')
    
    fig_growth_compare = go.Figure()
    
    fig_growth_compare.add_trace(go.Bar(
        name='Spend Growth',
        x=df_growth['Country'],
        y=df_growth['Spend_YoY'],
        marker_color='steelblue'
    ))
    
    fig_growth_compare.add_trace(go.Bar(
        name='AOV Growth',
        x=df_growth['Country'],
        y=df_growth['Revenue_YoY'],
        marker_color='lightgreen'
    ))
    
    fig_growth_compare.add_trace(go.Bar(
        name='Shopper Growth',
        x=df_growth['Country'],
        y=df_growth['Shoppers_YoY'],
        marker_color='coral'
    ))
    
    fig_growth_compare.update_layout(
        barmode='group',
        height=500,
        title='YoY Growth Comparison (Top 15 by Spend Growth)',
        yaxis_title='Growth Rate (%)',
        xaxis_title='Country'
    )
    
    st.plotly_chart(fig_growth_compare, use_container_width=True)
    
    st.markdown("---")
    
    # Penetration analysis
    st.subheader("E-commerce Penetration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top penetration markets
        top_penetration = df_filtered.nlargest(10, 'Penetration_Percent')[['Country', 'Penetration_Percent', 'Penetration_YoY']]
        
        fig_pen = px.bar(
            top_penetration,
            x='Penetration_Percent',
            y='Country',
            orientation='h',
            text='Penetration_Percent',
            color='Penetration_YoY',
            color_continuous_scale='RdYlGn',
            title='Top 10 Markets by E-commerce Penetration',
            labels={'Penetration_Percent': 'Penetration (%)', 'Penetration_YoY': 'Growth (%)'}
        )
        fig_pen.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_pen.update_layout(height=400)
        st.plotly_chart(fig_pen, use_container_width=True)
    
    with col2:
        # Penetration vs market size
        fig_pen_scatter = px.scatter(
            df_filtered,
            x='Penetration_Percent',
            y='Spend_Billions',
            size='Shoppers_Millions',
            color='Business_Region',
            hover_name='Country',
            title='Penetration vs Market Size',
            labels={
                'Penetration_Percent': 'E-commerce Penetration (%)',
                'Spend_Billions': 'Market Size ($B)',
                'Shoppers_Millions': 'Shoppers (M)'
            }
        )
        fig_pen_scatter.update_layout(height=400)
        st.plotly_chart(fig_pen_scatter, use_container_width=True)

# TAB 5: PRODUCT CATEGORIES
with tab5:
    st.header("Product Category Analysis")
    
    # Category selector
    selected_category = st.selectbox("Select Category to Analyze", category_cols)
    
    st.markdown("---")
    
    # Category overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Top Markets for {selected_category}")
        
        top_category = df_filtered.nlargest(15, selected_category)[['Country', selected_category, 'Spend_Billions']]
        
        fig_cat = px.bar(
            top_category,
            x=selected_category,
            y='Country',
            orientation='h',
            text=selected_category,
            color='Spend_Billions',
            color_continuous_scale='Viridis',
            labels={selected_category: f'{selected_category} Spend ($B)', 'Spend_Billions': 'Total Market Size ($B)'}
        )
        fig_cat.update_traces(texttemplate='$%{text:.1f}B', textposition='outside')
        fig_cat.update_layout(height=500)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("Category Stats")
        
        total_category_spend = df_filtered[selected_category].sum()
        st.metric("Total Category Spend", f"${total_category_spend:.1f}B")
        
        avg_category_spend = df_filtered[selected_category].mean()
        st.metric("Avg per Country", f"${avg_category_spend:.1f}B")
        
        top_market = df_filtered.nlargest(1, selected_category)
        if not top_market.empty:
            st.metric("Top Market", top_market.iloc[0]['Country'], f"${top_market.iloc[0][selected_category]:.1f}B")
    
    st.markdown("---")
    
    # All categories heatmap
    st.subheader("Category Comparison Across Selected Markets")
    
    if len(selected_countries) > 0:
        # Create heatmap data
        heatmap_data = df_filtered[df_filtered['Country'].isin(selected_countries)][['Country'] + category_cols].set_index('Country')
        
        fig_heatmap = px.imshow(
            heatmap_data.T,
            labels=dict(x="Country", y="Category", color="Spend ($B)"),
            x=heatmap_data.index,
            y=heatmap_data.columns,
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # Category mix for selected country
        if len(selected_countries) > 0:
            selected_for_mix = st.selectbox("Select Country for Category Mix", selected_countries)
            
            country_data = df_filtered[df_filtered['Country'] == selected_for_mix][category_cols].T
            country_data.columns = ['Spend']
            country_data = country_data.reset_index()
            country_data.columns = ['Category', 'Spend']
            country_data = country_data.sort_values('Spend', ascending=False)
            
            fig_pie = px.pie(
                country_data,
                values='Spend',
                names='Category',
                title=f'Category Mix for {selected_for_mix}'
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

# TAB 6: PAYMENT METHODS
with tab6:
    st.header("Payment Method Preferences")
    
    st.markdown("Understanding payment preferences is critical for checkout optimization and reducing cart abandonment.")
    
    st.markdown("---")
    
    # Payment method distribution
    st.subheader("Payment Method Distribution by Country")
    
    # Prepare data for stacked bar chart
    payment_data = df_filtered[['Country'] + payment_cols].copy()
    payment_data = payment_data.set_index('Country')
    
    # Rename columns for display
    payment_display_names = {
        'Payment_Mobile_Wallets': 'Mobile Wallets',
        'Payment_Cards': 'Cards',
        'Payment_A2A': 'A2A Transfers',
        'Payment_BNPL': 'BNPL',
        'Payment_Other': 'Other'
    }
    payment_data.columns = [payment_display_names.get(col, col) for col in payment_data.columns]
    
    fig_payment = px.bar(
        payment_data.head(15),
        barmode='stack',
        labels={'value': 'Share (%)', 'variable': 'Payment Method'},
        title='Payment Method Distribution (Top 15 Markets)',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    )
    fig_payment.update_layout(height=500, xaxis_title='Country', yaxis_title='Share (%)')
    st.plotly_chart(fig_payment, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Top BNPL Markets")
        top_bnpl = df_filtered.nlargest(10, 'Payment_BNPL')[['Country', 'Payment_BNPL']]
        top_bnpl_display = top_bnpl.copy()
        top_bnpl_display['Payment_BNPL'] = top_bnpl_display['Payment_BNPL'].apply(lambda x: f"{x:.1f}%")
        top_bnpl_display.columns = ['Country', 'BNPL Share']
        st.dataframe(top_bnpl_display, hide_index=True, use_container_width=True)
        
        st.info("💡 **Key Insight**: These markets require BNPL integration to maximize conversion.")
    
    with col2:
        st.subheader("📱 Top Mobile Wallet Markets")
        top_mobile = df_filtered.nlargest(10, 'Payment_Mobile_Wallets')[['Country', 'Payment_Mobile_Wallets']]
        top_mobile_display = top_mobile.copy()
        top_mobile_display['Payment_Mobile_Wallets'] = top_mobile_display['Payment_Mobile_Wallets'].apply(lambda x: f"{x:.1f}%")
        top_mobile_display.columns = ['Country', 'Mobile Wallet Share']
        st.dataframe(top_mobile_display, hide_index=True, use_container_width=True)
        
        st.info("💡 **Key Insight**: Mobile wallet support is critical in these markets.")
    
    st.markdown("---")
    
    # A2A Transfer warning
    st.subheader("⚠️ Account-to-Account (A2A) Transfer Requirements")
    top_a2a = df_filtered.nlargest(10, 'Payment_A2A')[['Country', 'Payment_A2A']]
    
    fig_a2a = px.bar(
        top_a2a,
        x='Payment_A2A',
        y='Country',
        orientation='h',
        text='Payment_A2A',
        title='Top 10 Markets by A2A Transfer Usage',
        color='Payment_A2A',
        color_continuous_scale='Reds'
    )
    fig_a2a.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_a2a.update_layout(height=400)
    st.plotly_chart(fig_a2a, use_container_width=True)
    
    st.warning("⚠️ **Critical**: Markets with >50% A2A usage require bank transfer support. Launching without this payment method will result in significant checkout abandonment.")

# TAB 7: PURCHASE MOTIVATORS
with tab7:
    st.header("Purchase Motivator Analysis")
    
    st.markdown("Understanding what drives purchase decisions helps optimize marketing and operations.")
    
    st.markdown("---")
    
    # Motivator horizontal bar chart
    st.subheader("Purchase Motivator Importance by Country")
    
    if len(selected_countries) > 0:
        # Country selector - default to single country if only one is selected
        if len(selected_countries) == 1:
            # Automatically use the single selected country
            selected_country_for_chart = selected_countries[0]
        else:
            # Show dropdown if multiple countries are selected
            selected_country_for_chart = st.selectbox(
                "Select Country for Detailed Analysis",
                selected_countries,
                key="motivator_country_selector"
            )
        
        # Prepare data for horizontal bar chart
        motivator_display_names = {
            'Motivator_Free_Delivery': 'Free Delivery',
            'Motivator_Next_Day': 'Next Day Delivery',
            'Motivator_Coupons': 'Coupons & Discounts',
            'Motivator_Simple_Checkout': 'Simple Checkout',
            'Motivator_Reviews': 'Customer Reviews',
            'Motivator_Returns': 'Easy Returns',
            'Motivator_Loyalty': 'Loyalty Points',
            'Motivator_Interest_Free': 'Interest Free',
            'Motivator_Eco_Friendly': 'Eco-Friendly',
            'Motivator_Guest_Checkout': 'Guest Checkout',
            'Motivator_Click_Collect': 'Click & Collect'
        }
        
        # Get data for the selected country
        country_data = df_filtered[df_filtered['Country'] == selected_country_for_chart]
        
        if not country_data.empty:
            # Prepare data for the chart
            chart_data = []
            for motivator_col in motivator_cols:
                motivator_name = motivator_display_names.get(motivator_col, motivator_col)
                value = country_data[motivator_col].iloc[0]
                chart_data.append({
                    'Motivator': motivator_name,
                    'Importance': value
                })
            
            chart_df = pd.DataFrame(chart_data)
            chart_df = chart_df.sort_values('Importance', ascending=True)
            
            # Create horizontal bar chart for single country
            fig_motivator_bar = px.bar(
                chart_df,
                x='Importance',
                y='Motivator',
                orientation='h',
                text='Importance',
                color='Importance',
                color_continuous_scale='RdYlGn',
                title=f'Purchase Motivators for {selected_country_for_chart}',
                labels={'Importance': 'Importance (%)', 'Motivator': 'Motivator'}
            )
            
            fig_motivator_bar.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig_motivator_bar.update_layout(
                height=max(500, len(chart_df) * 40),
                xaxis_title='Importance (%)',
                yaxis_title='Motivator',
                showlegend=False
            )
            
            st.plotly_chart(fig_motivator_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Country-specific motivators
    if len(selected_countries) > 0:
        selected_for_motivator = st.selectbox("Select Country for Detailed Analysis", selected_countries)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            country_motivators = df_filtered[df_filtered['Country'] == selected_for_motivator][motivator_cols].T
            country_motivators.columns = ['Importance']
            country_motivators = country_motivators.reset_index()
            country_motivators.columns = ['Motivator', 'Importance']
            country_motivators['Motivator'] = country_motivators['Motivator'].map(motivator_display_names)
            country_motivators = country_motivators.sort_values('Importance', ascending=True)
            
            fig_country_mot = px.bar(
                country_motivators,
                x='Importance',
                y='Motivator',
                orientation='h',
                text='Importance',
                title=f'Purchase Motivators for {selected_for_motivator}',
                color='Importance',
                color_continuous_scale='RdYlGn'
            )
            fig_country_mot.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_country_mot.update_layout(height=500)
            st.plotly_chart(fig_country_mot, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Action Items")
            
            top_3_motivators = country_motivators.nlargest(3, 'Importance')
            
            st.markdown("**Top 3 Priorities:**")
            for idx, row in top_3_motivators.iterrows():
                st.markdown(f"**{idx+1}. {row['Motivator']}** ({row['Importance']:.1f}%)")
            
            st.markdown("---")
            
            st.success(f"💡 **Recommendation**: Focus on {top_3_motivators.iloc[0]['Motivator']} to maximize conversion in {selected_for_motivator}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**: Global E-commerce Market Research 2025")