import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
from datetime import datetime

# For PDF export
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import plotly.io as pio
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

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

# Load LPI Metrics data
@st.cache_data
def load_lpi_data():
    """Load and clean LPI Metrics data (now includes Business Region, World Bank Region, and ISO2)"""
    lpi_df = pd.read_csv('LPI Metrics.csv')
    
    # Clean column names - remove extra spaces
    lpi_df.columns = lpi_df.columns.str.strip()
    
    # Rename 'Economy' to 'Country' for consistency
    lpi_df = lpi_df.rename(columns={'Economy': 'Country'})
    
    return lpi_df

# Helper function to format numbers with commas
def format_number_with_commas(value, format_type='default'):
    """
    Format numbers with comma separators for thousands.
    
    Args:
        value: The numeric value to format
        format_type: 'default', 'M' (millions), 'B' (billions), 'currency', 'percent'
    
    Returns:
        Formatted string with commas
    """
    if pd.isna(value):
        return ''
    
    # Format the number part with commas
    if format_type == 'M':
        # For millions: 3077.9 -> "3,077.9M"
        formatted = f"{value:,.1f}M"
    elif format_type == 'B':
        # For billions: 3077.9 -> "$3,077.9B"
        formatted = f"${value:,.1f}B"
    elif format_type == 'currency':
        # For currency: 3077.9 -> "$3,077.90"
        formatted = f"${value:,.0f}"
    elif format_type == 'percent':
        # For percentages: 3077.9 -> "3,077.9%"
        formatted = f"{value:,.1f}%"
    elif format_type == 'decimal':
        # For decimals: 3077.9 -> "3,077.9"
        formatted = f"{value:,.1f}"
    elif format_type == 'integer':
        # For integers: 3077 -> "3,077"
        formatted = f"{value:,.0f}"
    else:
        # Default: just add commas
        if isinstance(value, float) and value % 1 == 0:
            formatted = f"{value:,.0f}"
        else:
            formatted = f"{value:,.1f}"
    
    return formatted

# Function to create PDF export
def create_pdf_report(df_export, selected_countries_list, lpi_data, selected_region):
    """Create a comprehensive PDF report with all tabs"""
    
    if not PDF_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        topMargin=0.75*inch, 
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    story = []
    
    # Brand colors
    cobalt = colors.HexColor('#175CFF')
    denim = colors.HexColor('#0A083B')
    light_denim = colors.HexColor('#57586E')
    white = colors.white
    sky = colors.HexColor('#52C4FF')
    carrot = colors.HexColor('#FF8030')
    canary = colors.HexColor('#FFD61A')
    dark_ash = colors.HexColor('#999999')
    cloud = colors.HexColor('#E8E8E8')
    
    # Custom styles with Neurial Grotesk (fallback to Helvetica)
    # H1 - 60pt Bold
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=60,
        leading=68,
        textColor=denim,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # H2 - 50pt Medium
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=50,
        leading=56,
        textColor=denim,
        spaceAfter=24,
        spaceBefore=24
    )
    
    # H3 - 40pt Medium
    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=40,
        leading=48,
        textColor=denim,
        spaceAfter=20,
        spaceBefore=20
    )
    
    # H4 - 24pt Medium
    h4_style = ParagraphStyle(
        'H4',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=denim,
        spaceAfter=16,
        spaceBefore=16
    )
    
    # H5 - 20pt Bold
    h5_style = ParagraphStyle(
        'H5',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=26,
        textColor=cobalt,
        spaceAfter=12,
        spaceBefore=12
    )
    
    # H6 - 11pt Bold
    h6_style = ParagraphStyle(
        'H6',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        textColor=dark_ash,
        spaceAfter=8
    )
    
    # Body text - 15pt
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=15,
        leading=22,
        textColor=light_denim
    )
    
    # Title Page
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("International E-commerce", h1_style))
    story.append(Paragraph("Market Analysis Report", h2_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Add a colored line separator
    from reportlab.graphics.shapes import Drawing, Line
    d = Drawing(6*inch, 3)
    line = Line(0, 0, 6*inch, 0)
    line.strokeColor = cobalt
    line.strokeWidth = 3
    d.add(line)
    story.append(d)
    story.append(Spacer(1, 0.3*inch))
    
    country_list = ", ".join(selected_countries_list) if len(selected_countries_list) <= 5 else f"{len(selected_countries_list)} countries selected"
    story.append(Paragraph(f"<b>Selected Markets:</b> {country_list}", body_style))
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # ShipBob branding
    story.append(Paragraph("Powered by ShipBob International Expansion Tool", h6_style))
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", h3_style))
    story.append(Spacer(1, 0.2*inch))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total E-commerce Shoppers', format_number_with_commas(df_export['Shoppers_Millions'].sum(), 'M')],
        ['Total Market Size', format_number_with_commas(df_export['Spend_Billions'].sum(), 'B')],
        ['Average Revenue per Shopper', format_number_with_commas(df_export['Revenue_Per_Shopper'].mean(), 'currency')],
        ['Average YoY Growth Rate', format_number_with_commas(df_export['Spend_YoY'].mean(), 'percent')]
    ]
    
    summary_table = Table(summary_data, colWidths=[3.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), cobalt),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 15),
        ('FONTSIZE', (0, 1), (-1, -1), 13),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('LINEBELOW', (0, 0), (-1, 0), 2, cobalt),
        ('LINEBELOW', (0, 1), (-1, -2), 0.5, cloud),
        ('LINEBELOW', (0, -1), (-1, -1), 2, cobalt)
    ]))
    story.append(summary_table)
    story.append(PageBreak())
    
    # Top Markets Table
    story.append(Paragraph("Top Markets by Total Spend", h4_style))
    story.append(Spacer(1, 0.2*inch))
    
    top_markets = df_export.nlargest(10, 'Spend_Billions')[['Country', 'Spend_Billions', 'Shoppers_Millions', 'Revenue_Per_Shopper', 'Spend_YoY']]
    
    table_data = [['Country', 'Market Size', 'Shoppers', 'AOV', 'Growth']]
    for idx, row in top_markets.iterrows():
        table_data.append([
            row['Country'],
            format_number_with_commas(row['Spend_Billions'], 'B'),
            format_number_with_commas(row['Shoppers_Millions'], 'M'),
            format_number_with_commas(row['Revenue_Per_Shopper'], 'currency'),
            format_number_with_commas(row['Spend_YoY'], 'percent')
        ])
    
    markets_table = Table(table_data, colWidths=[1.8*inch, 1.2*inch, 1.1*inch, 1.0*inch, 1.0*inch])
    markets_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), cobalt),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#F5F5F5')]),
        ('LINEBELOW', (0, 0), (-1, 0), 2, cobalt),
        ('GRID', (0, 0), (-1, -1), 0.5, cloud)
    ]))
    story.append(markets_table)
    story.append(PageBreak())
    
    # Market Segmentation
    story.append(Paragraph("Market Segmentation by Region", h4_style))
    story.append(Spacer(1, 0.2*inch))
    
    region_summary = df_export.groupby('Business_Region').agg({
        'Spend_Billions': 'sum',
        'Shoppers_Millions': 'sum',
        'Country': 'count'
    }).reset_index()
    region_summary = region_summary.sort_values('Spend_Billions', ascending=False)
    
    region_data = [['Business Region', 'Total Spend', 'Total Shoppers', 'Countries']]
    for _, row in region_summary.iterrows():
        region_data.append([
            row['Business_Region'],
            format_number_with_commas(row['Spend_Billions'], 'B'),
            format_number_with_commas(row['Shoppers_Millions'], 'M'),
            format_number_with_commas(row['Country'], 'integer')
        ])
    
    region_table = Table(region_data, colWidths=[2.2*inch, 1.6*inch, 1.6*inch, 1.0*inch])
    region_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), sky),
        ('TEXTCOLOR', (0, 0), (-1, 0), denim),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#F5F5F5')]),
        ('LINEBELOW', (0, 0), (-1, 0), 2, sky),
        ('GRID', (0, 0), (-1, -1), 0.5, cloud)
    ]))
    story.append(region_table)
    story.append(PageBreak())
    
    # Shopper Growth Analysis
    story.append(Paragraph("Shopper Growth Analysis", h4_style))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(f"Total E-commerce Shoppers: <b>{format_number_with_commas(df_export['Shoppers_Millions'].sum(), 'M')}</b>", body_style))
    story.append(Paragraph(f"Average Shopper Growth Rate: <b>{format_number_with_commas(df_export['Shoppers_YoY'].mean(), 'percent')}</b>", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Fastest Growing Markets", h5_style))
    top_growth = df_export.nlargest(10, 'Shoppers_YoY')[['Country', 'Shoppers_YoY', 'Shoppers_Millions']]
    growth_data = [['Country', 'Growth Rate', 'Current Shoppers']]
    for _, row in top_growth.iterrows():
        growth_data.append([
            row['Country'],
            format_number_with_commas(row['Shoppers_YoY'], 'percent'),
            format_number_with_commas(row['Shoppers_Millions'], 'M')
        ])
    
    growth_table = Table(growth_data, colWidths=[2.5*inch, 1.5*inch, 1.8*inch])
    growth_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), carrot),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#FFF5F0')]),
        ('LINEBELOW', (0, 0), (-1, 0), 2, carrot),
        ('GRID', (0, 0), (-1, -1), 0.5, cloud)
    ]))
    story.append(growth_table)
    story.append(PageBreak())
    
    # Payment Methods Analysis
    story.append(Paragraph("Payment Method Preferences", h4_style))
    story.append(Paragraph("Critical for checkout optimization and reducing cart abandonment", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Top BNPL markets
    story.append(Paragraph("Top Buy Now Pay Later (BNPL) Markets", h5_style))
    top_bnpl = df_export.nlargest(5, 'Payment_BNPL')[['Country', 'Payment_BNPL']]
    bnpl_data = [['Country', 'BNPL Share']]
    for _, row in top_bnpl.iterrows():
        bnpl_data.append([row['Country'], format_number_with_commas(row['Payment_BNPL'], 'percent')])
    
    bnpl_table = Table(bnpl_data, colWidths=[3.5*inch, 2*inch])
    bnpl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), canary),
        ('TEXTCOLOR', (0, 0), (-1, 0), denim),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#FFFEF5')]),
        ('LINEBELOW', (0, 0), (-1, 0), 2, canary),
        ('GRID', (0, 0), (-1, -1), 0.5, cloud)
    ]))
    story.append(bnpl_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Top A2A markets - Critical warning
    story.append(Paragraph("⚠️ Account-to-Account (A2A) Transfer Markets", h5_style))
    story.append(Paragraph("<b>Critical:</b> Markets with high A2A usage require bank transfer support", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    top_a2a = df_export.nlargest(5, 'Payment_A2A')[['Country', 'Payment_A2A']]
    a2a_data = [['Country', 'A2A Share']]
    for _, row in top_a2a.iterrows():
        a2a_data.append([row['Country'], format_number_with_commas(row['Payment_A2A'], 'percent')])
    
    a2a_table = Table(a2a_data, colWidths=[3.5*inch, 2*inch])
    a2a_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF4444')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFF0F0'), colors.HexColor('#FFE5E5')]),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#FF4444')),
        ('GRID', (0, 0), (-1, -1), 0.5, cloud)
    ]))
    story.append(a2a_table)
    story.append(PageBreak())
    
    # Product Categories
    story.append(Paragraph("Product Category Analysis", h4_style))
    story.append(Paragraph("Top markets by category spend", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    category_cols = ['Fashion', 'Food', 'Furniture', 'Electronics', 'Beverages']
    category_colors = [cobalt, sky, carrot, canary, colors.HexColor('#9B59B6')]
    
    for idx, category in enumerate(category_cols):
        story.append(Paragraph(f"{category}", h5_style))
        top_cat = df_export.nlargest(5, category)[['Country', category]]
        cat_data = [['Country', 'Spend']]
        for _, row in top_cat.iterrows():
            cat_data.append([row['Country'], format_number_with_commas(row[category], 'B')])
        
        cat_table = Table(cat_data, colWidths=[3.5*inch, 2*inch])
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), category_colors[idx]),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#F9F9F9')]),
            ('LINEBELOW', (0, 0), (-1, 0), 2, category_colors[idx]),
            ('GRID', (0, 0), (-1, -1), 0.5, cloud)
        ]))
        story.append(cat_table)
        story.append(Spacer(1, 0.15*inch))
    
    # LPI Metrics Section
    story.append(PageBreak())
    story.append(Paragraph("Logistics Performance Index (LPI) Metrics", h3_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Filter LPI data for selected countries
    lpi_export = lpi_data[lpi_data['Country'].isin(selected_countries_list)].copy()
    
    # If Business Region is selected, also filter by Business Region
    if selected_region != 'All':
        if 'Business Region' in lpi_export.columns:
            lpi_export = lpi_export[lpi_export['Business Region'] == selected_region].copy()
        elif 'Business_Region' in lpi_export.columns:
            lpi_export = lpi_export[lpi_export['Business_Region'] == selected_region].copy()
    
    if len(lpi_export) > 0:
        story.append(Paragraph("The Logistics Performance Index (LPI) measures logistics performance across countries. Higher scores indicate better logistics infrastructure and efficiency (scale: 1-5).", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # LPI Summary Metrics
        story.append(Paragraph("LPI Summary", h4_style))
        story.append(Spacer(1, 0.1*inch))
        
        lpi_summary_data = [
            ['Metric', 'Value'],
            ['Average LPI Score', f"{lpi_export['LPI Score'].mean():.2f}"],
            ['Highest LPI Score', f"{lpi_export['LPI Score'].max():.2f}"],
            ['Lowest LPI Score', f"{lpi_export['LPI Score'].min():.2f}"],
            ['Number of Countries', str(len(lpi_export))]
        ]
        
        lpi_summary_table = Table(lpi_summary_data, colWidths=[3.5*inch, 2.5*inch])
        lpi_summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), cobalt),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 13),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, cobalt),
            ('GRID', (0, 0), (-1, -1), 0.5, cloud)
        ]))
        story.append(lpi_summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Top LPI Countries
        story.append(Paragraph("Top Countries by LPI Score", h4_style))
        story.append(Spacer(1, 0.1*inch))
        
        top_lpi = lpi_export.nlargest(15, 'LPI Score')[['Country', 'LPI Score', 'Customs Score', 'Infrastructure Score', 'Timeliness Score']]
        
        lpi_table_data = [['Country', 'LPI Score', 'Customs', 'Infrastructure', 'Timeliness']]
        for _, row in top_lpi.iterrows():
            lpi_table_data.append([
                row['Country'],
                f"{row['LPI Score']:.2f}",
                f"{row['Customs Score']:.2f}",
                f"{row['Infrastructure Score']:.2f}",
                f"{row['Timeliness Score']:.2f}"
            ])
        
        lpi_table = Table(lpi_table_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 1.0*inch, 0.9*inch])
        lpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), sky),
            ('TEXTCOLOR', (0, 0), (-1, 0), denim),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#F9F9F9')]),
            ('LINEBELOW', (0, 0), (-1, 0), 2, sky),
            ('GRID', (0, 0), (-1, -1), 0.5, cloud)
        ]))
        story.append(lpi_table)
        story.append(Spacer(1, 0.3*inch))
        
        # LPI Component Averages
        story.append(Paragraph("Average LPI Component Scores", h4_style))
        story.append(Spacer(1, 0.1*inch))
        
        lpi_components = ['Customs Score', 'Infrastructure Score', 'International Shipments Score',
                         'Logistics Competence and Quality Score', 'Timeliness Score', 'Tracking and Tracing Score']
        component_labels = ['Customs', 'Infrastructure', 'Intl Shipments', 'Competence', 'Timeliness', 'Tracking']
        
        component_data = [['Component', 'Average Score']]
        for comp, label in zip(lpi_components, component_labels):
            if comp in lpi_export.columns:
                component_data.append([label, f"{lpi_export[comp].mean():.2f}"])
        
        component_table = Table(component_data, colWidths=[3.5*inch, 2*inch])
        component_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), carrot),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, colors.HexColor('#FFF5F0')]),
            ('LINEBELOW', (0, 0), (-1, 0), 2, carrot),
            ('GRID', (0, 0), (-1, -1), 0.5, cloud)
        ]))
        story.append(component_table)
    else:
        story.append(Paragraph("No LPI data available for the selected countries.", body_style))
    
    # Footer on last page
    story.append(PageBreak())
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("End of Report", h3_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add separator line
    from reportlab.graphics.shapes import Drawing, Line
    d2 = Drawing(6*inch, 3)
    line2 = Line(0, 0, 6*inch, 0)
    line2.strokeColor = cobalt
    line2.strokeWidth = 2
    d2.add(line2)
    story.append(d2)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", h6_style))
    story.append(Paragraph("Powered by ShipBob International Expansion Tool", h6_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

df = load_data()
lpi_df = load_lpi_data()

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

# Export button
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Export Report")

# Debug: Show if PDF libraries are available
st.sidebar.write(f"PDF_AVAILABLE: {PDF_AVAILABLE}")

if PDF_AVAILABLE:
    if st.sidebar.button("Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            pdf_buffer = create_pdf_report(df_filtered, selected_countries if selected_countries else df_filtered['Country'].tolist(), lpi_df, selected_region)
            
            if pdf_buffer:
                st.sidebar.download_button(
                    label="📥 Download PDF",
                    data=pdf_buffer,
                    file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.sidebar.success("✅ PDF generated successfully!")
else:
    st.sidebar.warning("⚠️ PDF export requires: pip install reportlab kaleido")
    st.sidebar.info("Run this in your terminal and restart the app.")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Summary", 
    "🌐 Market Segmentation",
    "👥 Shopper Growth", 
    "💰 Revenue Metrics",
    "🛍️ Product Categories",
    "💳 Payment Methods",
    "⭐ Purchase Motivators",
    "📦 LPI Metrics"
])

# TAB 1: SUMMARY
with tab1:
    st.header("Market Overview Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_shoppers = df_filtered['Shoppers_Millions'].sum()
        st.metric("Total Shoppers", format_number_with_commas(total_shoppers, 'M'))
    
    with col2:
        total_spend = df_filtered['Spend_Billions'].sum()
        st.metric("Total Market Size", format_number_with_commas(total_spend, 'B'))
    
    with col3:
        avg_revenue = df_filtered['Revenue_Per_Shopper'].mean()
        st.metric("Avg Revenue/Shopper", format_number_with_commas(avg_revenue, 'currency'))
    
    with col4:
        avg_growth = df_filtered['Spend_YoY'].mean()
        st.metric("Avg Growth Rate", format_number_with_commas(avg_growth, 'percent'))
    
    st.markdown("---")
    
    # Top markets table
    st.subheader("Top Markets by Total Spend")
    
    top_markets = df_filtered.nlargest(10, 'Spend_Billions')[['Country', 'Spend_Billions', 'Shoppers_Millions', 'Revenue_Per_Shopper', 'Spend_YoY']]
    top_markets_display = top_markets.copy()
    top_markets_display['Spend_Billions'] = top_markets_display['Spend_Billions'].apply(lambda x: format_number_with_commas(x, 'B'))
    top_markets_display['Shoppers_Millions'] = top_markets_display['Shoppers_Millions'].apply(lambda x: format_number_with_commas(x, 'M'))
    top_markets_display['Revenue_Per_Shopper'] = top_markets_display['Revenue_Per_Shopper'].apply(lambda x: format_number_with_commas(x, 'currency'))
    top_markets_display['Spend_YoY'] = top_markets_display['Spend_YoY'].apply(lambda x: format_number_with_commas(x, 'percent'))
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
        st.metric("Total Shoppers", format_number_with_commas(total_shoppers, 'M'))
    
    with col2:
        avg_growth = df_filtered['Shoppers_YoY'].mean()
        st.metric("Avg Shopper Growth", format_number_with_commas(avg_growth, 'percent'))
    
    with col3:
        fastest_growth = df_filtered.nlargest(1, 'Shoppers_YoY')
        if not fastest_growth.empty:
            st.metric("Fastest Growing", f"{fastest_growth.iloc[0]['Country']}", format_number_with_commas(fastest_growth.iloc[0]['Shoppers_YoY'], 'percent'))
    
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
        st.metric("Total Market Size", format_number_with_commas(total_spend, 'B'))
    
    with col2:
        avg_spend_growth = df_filtered['Spend_YoY'].mean()
        st.metric("Avg Spend Growth", format_number_with_commas(avg_spend_growth, 'percent'))
    
    with col3:
        avg_aov = df_filtered['Revenue_Per_Shopper'].mean()
        st.metric("Avg AOV", format_number_with_commas(avg_aov, 'currency'))
    
    with col4:
        avg_aov_growth = df_filtered['Revenue_YoY'].mean()
        st.metric("Avg AOV Growth", format_number_with_commas(avg_aov_growth, 'percent'))
    
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
        st.metric("Total Category Spend", format_number_with_commas(total_category_spend, 'B'))
        
        avg_category_spend = df_filtered[selected_category].mean()
        st.metric("Avg per Country", format_number_with_commas(avg_category_spend, 'B'))
        
        top_market = df_filtered.nlargest(1, selected_category)
        if not top_market.empty:
            st.metric("Top Market", top_market.iloc[0]['Country'], format_number_with_commas(top_market.iloc[0][selected_category], 'B'))
    
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
        top_bnpl_display['Payment_BNPL'] = top_bnpl_display['Payment_BNPL'].apply(lambda x: format_number_with_commas(x, 'percent'))
        top_bnpl_display.columns = ['Country', 'BNPL Share']
        st.dataframe(top_bnpl_display, hide_index=True, use_container_width=True)
        
        st.info("💡 **Key Insight**: These markets require BNPL integration to maximize conversion.")
    
    with col2:
        st.subheader("📱 Top Mobile Wallet Markets")
        top_mobile = df_filtered.nlargest(10, 'Payment_Mobile_Wallets')[['Country', 'Payment_Mobile_Wallets']]
        top_mobile_display = top_mobile.copy()
        top_mobile_display['Payment_Mobile_Wallets'] = top_mobile_display['Payment_Mobile_Wallets'].apply(lambda x: format_number_with_commas(x, 'percent'))
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
    
    # Motivator heatmap
    st.subheader("Purchase Motivator Importance by Country")
    
    if len(selected_countries) > 0:
        motivator_data = df_filtered[df_filtered['Country'].isin(selected_countries)][['Country'] + motivator_cols].set_index('Country')
        
        # Rename for display
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
        motivator_data.columns = [motivator_display_names.get(col, col) for col in motivator_data.columns]
        
        fig_motivator_heat = px.imshow(
            motivator_data.T,
            labels=dict(x="Country", y="Motivator", color="Importance (%)"),
            x=motivator_data.index,
            y=motivator_data.columns,
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig_motivator_heat.update_layout(height=600)
        st.plotly_chart(fig_motivator_heat, use_container_width=True)
    
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

# TAB 8: LPI METRICS
with tab8:
    st.header("📦 Logistics Performance Index (LPI) Metrics")
    
    st.markdown("""
    The **Logistics Performance Index (LPI)** is a World Bank metric that measures logistics performance across countries.
    Higher scores indicate better logistics infrastructure and efficiency (scale: 1-5).
    """)
    
    # Filter LPI data to only countries in our main dataset and respect Business Region filter
    lpi_filtered = lpi_df[lpi_df['Country'].isin(df_filtered['Country'])].copy()
    
    # If Business Region is selected, also filter by Business Region
    # Check for both 'Business Region' (from CSV) and 'Business_Region' (from df_filtered merge)
    if selected_region != 'All':
        if 'Business Region' in lpi_filtered.columns:
            lpi_filtered = lpi_filtered[lpi_filtered['Business Region'] == selected_region].copy()
        elif 'Business_Region' in lpi_filtered.columns:
            lpi_filtered = lpi_filtered[lpi_filtered['Business_Region'] == selected_region].copy()
    
    if len(lpi_filtered) == 0:
        st.warning("No LPI data available for selected countries")
    else:
        # Key Metrics
        st.subheader("📊 Overall LPI Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_lpi = lpi_filtered['LPI Score'].mean()
            st.metric("Average LPI Score", f"{avg_lpi:.2f}")
        
        with col2:
            top_performer = lpi_filtered.nlargest(1, 'LPI Score')
            st.metric("Top Performer", f"{top_performer['Country'].values[0]}")
        
        with col3:
            best_customs = lpi_filtered.nlargest(1, 'Customs Score')
            st.metric("Best Customs", f"{best_customs['Country'].values[0]}")
        
        with col4:
            best_timeliness = lpi_filtered.nlargest(1, 'Timeliness Score')
            st.metric("Best Timeliness", f"{best_timeliness['Country'].values[0]}")
        
        st.markdown("---")
        
        # Overall LPI Ranking
        st.subheader("🏆 Overall LPI Rankings")
        
        # Top 15 countries by LPI Score
        top_lpi = lpi_filtered.nlargest(15, 'LPI Score')
        
        fig_lpi_bar = px.bar(
            top_lpi,
            y='Country',
            x='LPI Score',
            orientation='h',
            title='Top 15 Countries by LPI Score',
            labels={'LPI Score': 'LPI Score (1-5)', 'Country': ''},
            color='LPI Score',
            color_continuous_scale='RdYlGn',
            text='LPI Score'
        )
        fig_lpi_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_lpi_bar.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_lpi_bar, use_container_width=True)
        
        st.markdown("---")
        
        # LPI Components Breakdown
        st.subheader("📈 LPI Components Analysis")
        
        st.markdown("""
        LPI is composed of 6 key components:
        1. **Customs** - Efficiency of customs clearance
        2. **Infrastructure** - Quality of trade and transport infrastructure
        3. **International Shipments** - Ease of arranging competitively priced shipments
        4. **Logistics Competence** - Competence and quality of logistics services
        5. **Timeliness** - Frequency of shipments reaching consignee within scheduled time
        6. **Tracking & Tracing** - Ability to track and trace consignments
        """)
        
        # Radar chart for selected country comparison
        st.subheader("🎯 Country LPI Component Comparison")
        
        # Country selector
        lpi_countries_list = lpi_filtered.nlargest(20, 'LPI Score')['Country'].tolist()
        selected_lpi_countries = st.multiselect(
            "Select countries to compare (up to 5)",
            lpi_countries_list,
            default=lpi_countries_list[:3],
            max_selections=5
        )
        
        if selected_lpi_countries:
            # Create radar chart
            lpi_components = ['Customs Score', 'Infrastructure Score', 'International Shipments Score',
                             'Logistics Competence and Quality Score', 'Timeliness Score', 'Tracking and Tracing Score']
            
            fig_radar = go.Figure()
            
            for country in selected_lpi_countries:
                country_data = lpi_filtered[lpi_filtered['Country'] == country]
                if len(country_data) > 0:
                    values = [country_data[comp].values[0] for comp in lpi_components]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=['Customs', 'Infrastructure', 'Intl Shipments', 'Competence', 'Timeliness', 'Tracking'],
                        fill='toself',
                        name=country
                    ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                showlegend=True,
                title="LPI Component Comparison by Country",
                height=600
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        # Component-by-component analysis
        st.subheader("🔍 Detailed Component Analysis")
        
        # Component selector
        component_options = {
            'Customs': 'Customs Score',
            'Infrastructure': 'Infrastructure Score',
            'International Shipments': 'International Shipments Score',
            'Logistics Competence': 'Logistics Competence and Quality Score',
            'Timeliness': 'Timeliness Score',
            'Tracking & Tracing': 'Tracking and Tracing Score'
        }
        
        selected_component = st.selectbox("Select LPI Component", list(component_options.keys()))
        component_col = component_options[selected_component]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers for selected component
            st.markdown(f"**Top 10 Countries - {selected_component}**")
            top_component = lpi_filtered.nlargest(10, component_col)
            
            fig_component = px.bar(
                top_component,
                y='Country',
                x=component_col,
                orientation='h',
                color=component_col,
                color_continuous_scale='Greens',
                text=component_col
            )
            fig_component.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_component.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_component, use_container_width=True)
        
        with col2:
            # Statistics for selected component
            st.markdown(f"**{selected_component} Statistics**")
            
            component_stats = {
                'Average Score': lpi_filtered[component_col].mean(),
                'Highest Score': lpi_filtered[component_col].max(),
                'Lowest Score': lpi_filtered[component_col].min(),
                'Std Deviation': lpi_filtered[component_col].std()
            }
            
            for stat, value in component_stats.items():
                st.metric(stat, f"{value:.2f}")
        
        st.markdown("---")
        
        # Horizontal Bar Chart - Top 25 countries by LPI Score
        st.subheader("🗺️ LPI Score - Top 25 Countries")
        
        # Limit to top 25 countries for readability
        top_countries = lpi_filtered.nlargest(25, 'LPI Score').sort_values('LPI Score', ascending=True)
        
        fig_bar = px.bar(
            top_countries,
            x='LPI Score',
            y='Country',
            orientation='h',
            color='LPI Score',
            color_continuous_scale='RdYlGn',
            title='LPI Score - Top 25 Countries',
            labels={'LPI Score': 'LPI Score (1-5)'},
            text='LPI Score'
        )
        fig_bar.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>LPI Score: %{x:.2f}<extra></extra>'
        )
        fig_bar.update_layout(
            height=800,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            xaxis_title='LPI Score',
            yaxis_title='Country'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Scatter plot: LPI vs Market Size
        st.subheader("💡 LPI Score vs Market Size")
        
        # Merge LPI data with market data
        # Include Business_Region and World Bank Region from lpi_filtered (now in CSV)
        merge_cols = ['Country', 'Spend_Billions', 'Shoppers_Millions']
        
        # Start with LPI data (which now includes Business Region, World Bank Region, ISO2)
        # Note: CSV has "Business Region" with space, not "Business_Region" with underscore
        lpi_merge_cols = ['Country']
        if 'Business Region' in lpi_filtered.columns:
            lpi_merge_cols.append('Business Region')
        elif 'Business_Region' in lpi_filtered.columns:
            lpi_merge_cols.append('Business_Region')
        if 'World Bank Region' in lpi_filtered.columns:
            lpi_merge_cols.append('World Bank Region')
        if 'ISO2' in lpi_filtered.columns:
            lpi_merge_cols.append('ISO2')
        
        # Merge LPI data with market data
        merged_data = pd.merge(
            lpi_filtered[lpi_merge_cols],
            df_filtered[merge_cols],
            on='Country',
            how='inner',
            suffixes=('', '_drop')
        )
        
        # Drop any duplicate columns created by merge
        merged_data = merged_data.loc[:, ~merged_data.columns.str.endswith('_drop')]
        
        # Standardize column name - use 'Business Region' if it exists, otherwise try 'Business_Region'
        business_region_col = None
        if 'Business Region' in merged_data.columns:
            business_region_col = 'Business Region'
        elif 'Business_Region' in merged_data.columns:
            business_region_col = 'Business_Region'
            # Rename to 'Business Region' for consistency
            merged_data = merged_data.rename(columns={'Business_Region': 'Business Region'})
            business_region_col = 'Business Region'
        
        # If Business Region not in merged_data, try to get it from df_filtered
        if business_region_col is None and 'Business_Region' in df_filtered.columns:
            region_map = df_filtered[['Country', 'Business_Region']].drop_duplicates(subset=['Country'])
            merged_data = merged_data.merge(region_map, on='Country', how='left')
            merged_data = merged_data.rename(columns={'Business_Region': 'Business Region'})
            business_region_col = 'Business Region'
        
        if len(merged_data) > 0:
            # Region filter - allow filtering by Business Region or World Bank Region
            col1, col2 = st.columns(2)
            
            selected_lpi_region = 'All'
            selected_wb_region = 'All'
            
            with col1:
                if business_region_col and business_region_col in merged_data.columns:
                    unique_regions = merged_data[business_region_col].dropna().unique().tolist()
                    if len(unique_regions) > 0:
                        available_regions = ['All'] + sorted(unique_regions)
                        selected_lpi_region = st.selectbox(
                            "Filter by Business Region",
                            available_regions,
                            key='lpi_business_region_filter'
                        )
            
            with col2:
                if 'World Bank Region' in merged_data.columns:
                    unique_wb_regions = merged_data['World Bank Region'].dropna().unique().tolist()
                    if len(unique_wb_regions) > 0:
                        available_wb_regions = ['All'] + sorted(unique_wb_regions)
                        selected_wb_region = st.selectbox(
                            "Filter by World Bank Region",
                            available_wb_regions,
                            key='lpi_wb_region_filter'
                        )
            
            # Apply filters
            if selected_lpi_region != 'All' and business_region_col and business_region_col in merged_data.columns:
                merged_data = merged_data[merged_data[business_region_col] == selected_lpi_region].copy()
            
            if selected_wb_region != 'All':
                merged_data = merged_data[merged_data['World Bank Region'] == selected_wb_region].copy()
            
            if len(merged_data) > 0:
                # Add tabs for different views
                tab1, tab2 = st.tabs(["📊 Scatter Plot", "📋 Table View"])
                
                with tab1:
                    # Determine color column - use Business Region if available and multiple regions, otherwise use LPI Score
                    if business_region_col and business_region_col in merged_data.columns and selected_lpi_region == 'All' and selected_wb_region == 'All' and merged_data[business_region_col].nunique() > 1:
                        color_col = business_region_col
                        color_scale = None
                    elif 'World Bank Region' in merged_data.columns and selected_lpi_region == 'All' and selected_wb_region == 'All' and merged_data['World Bank Region'].nunique() > 1:
                        color_col = 'World Bank Region'
                        color_scale = None
                    else:
                        color_col = 'LPI Score'
                        color_scale = 'Viridis'
                    
                    fig_scatter = px.scatter(
                        merged_data,
                        x='LPI Score',
                        y='Spend_Billions',
                        size='Shoppers_Millions',
                        hover_name='Country',
                        color=color_col,
                        color_continuous_scale=color_scale,
                        title='LPI Score vs E-commerce Market Size',
                        labels={
                            'LPI Score': 'LPI Score (1-5)',
                            'Spend_Billions': 'Market Size ($B)',
                            'Shoppers_Millions': 'Shoppers (M)',
                            'Business Region': 'Business Region',
                            'World Bank Region': 'World Bank Region'
                        }
                    )
                    fig_scatter.update_layout(height=600)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with tab2:
                    # Create a formatted table view
                    table_cols = ['Country', 'LPI Score', 'Spend_Billions', 'Shoppers_Millions']
                    if business_region_col and business_region_col in merged_data.columns:
                        table_cols.insert(1, business_region_col)
                    if 'World Bank Region' in merged_data.columns:
                        table_cols.insert(2, 'World Bank Region')
                    
                    display_data = merged_data[table_cols].copy()
                    display_data = display_data.sort_values('LPI Score', ascending=False)
                    display_data['LPI Score'] = display_data['LPI Score'].apply(lambda x: f"{x:.2f}")
                    display_data['Spend_Billions'] = display_data['Spend_Billions'].apply(lambda x: format_number_with_commas(x, 'B'))
                    display_data['Shoppers_Millions'] = display_data['Shoppers_Millions'].apply(lambda x: format_number_with_commas(x, 'M'))
                    
                    # Rename columns for display (already using 'Business Region' if business_region_col is set)
                    col_rename = {'Country': 'Country', 'LPI Score': 'LPI Score', 'Spend_Billions': 'Market Size ($B)', 'Shoppers_Millions': 'Shoppers (M)'}
                    if 'World Bank Region' in display_data.columns:
                        col_rename['World Bank Region'] = 'World Bank Region'
                    display_data.columns = [col_rename.get(col, col) for col in display_data.columns]
                    
                    # Style the dataframe
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
                
                st.info("""
                **💡 Key Insight**: Strong correlation between LPI scores and market size suggests that
                better logistics infrastructure supports larger e-commerce markets. Countries with higher
                LPI scores typically have more efficient supply chains and faster delivery times.
                """)
            else:
                st.warning("No data available for the selected Business Region.")
        
        st.markdown("---")
        
        # Full data table
        st.subheader("📋 Complete LPI Data")
        
        display_cols = ['Country']
        
        # Add region columns if available (check for 'Business Region' with space first, then 'Business_Region' with underscore)
        if 'Business Region' in lpi_filtered.columns:
            display_cols.append('Business Region')
        elif 'Business_Region' in lpi_filtered.columns:
            display_cols.append('Business_Region')
        if 'World Bank Region' in lpi_filtered.columns:
            display_cols.append('World Bank Region')
        if 'ISO2' in lpi_filtered.columns:
            display_cols.append('ISO2')
        
        # Add LPI scores
        display_cols.extend(['LPI Score', 'LPI Grouped Rank', 
                       'Customs Score', 'Infrastructure Score', 'International Shipments Score',
                       'Logistics Competence and Quality Score', 'Timeliness Score', 
                       'Tracking and Tracing Score'])
        
        # Only include columns that exist
        display_cols = [col for col in display_cols if col in lpi_filtered.columns]
        
        lpi_table = lpi_filtered[display_cols].sort_values('LPI Score', ascending=False).reset_index(drop=True)
        
        st.dataframe(
            lpi_table,
            use_container_width=True,
            height=400
        )

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**: Global E-commerce Market Research 2025")