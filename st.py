import pandas as pd
import streamlit as st
import plotly.express as px
import os
import json

# Configuration
DATA_DIR = "data"
RENTAL_FILE = os.path.join(DATA_DIR, "ura_rental_combined.parquet")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "ura_transaction_combined.parquet")
CONDO_FILE = os.path.join(DATA_DIR, "joined_data.csv")
DISTRICT_MAP_FILE = os.path.join(DATA_DIR, "district_map.json")

@st.cache_data
def load_data():
    """Load all data sources"""
    # Load rental and transaction data
    rental = pd.read_parquet(RENTAL_FILE)
    transactions = pd.read_parquet(TRANSACTIONS_FILE)
    
    # Load condo data with pre-calculated MRT distances
    condos = pd.read_csv(CONDO_FILE)
    
    # Convert dates and calculate PSF
    rental['lease_commencement_date'] = pd.to_datetime(rental['lease_commencement_date'], format="%b-%y" ) # MMM-YY
    # str of (500 to 600) convert to midpoint
    rental['floor_area_(sqft)'] = rental['floor_area_(sqft)'].str.replace(' to ', '-').str.replace(',', '').str.split('-').apply(lambda x: (int(x[0]) + int(x[1])) / 2 if len(x) == 2 else int(x[0]))
    # coerce monthly rent to numeric, handling commas and dollar signs
    rental['monthly_rent_($)'] = pd.to_numeric(rental['monthly_rent_($)'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    rental['rent_psf'] = rental['monthly_rent_($)'] / rental['floor_area_(sqft)']
    
    transactions['sale_date'] = pd.to_datetime(transactions['sale_date'], format="%b-%y" ) # MMM-YY
    # Convert area to numeric, handling commas and dollar signs
    transactions['area_(sqft)'] = pd.to_numeric(transactions['area_(sqft)'].str.replace(',', '').str.replace(' sqft', ''), errors='coerce')
    transactions['transacted_price_($)'] = pd.to_numeric(transactions['transacted_price_($)'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    transactions['sale_psf'] = transactions['transacted_price_($)'] / transactions['area_(sqft)']
    
    # Clean condo data
    condos['latitude'] = pd.to_numeric(condos['latitude'], errors='coerce')
    condos['longitude'] = pd.to_numeric(condos['longitude'], errors='coerce')
    condos['mrt_distance'] = pd.to_numeric(condos['mrt_distance'], errors='coerce')
    # remove " units" and "," from number_of_units
    condos['number_of_units'] = pd.to_numeric(condos['number_of_units'].str.replace(' units', '').str.replace(',', ''), errors='coerce')
    # Remove "Unknown" from completion year
    condos['completion'] = pd.to_numeric(condos['completion'].str.replace('Unknown', ''), errors='coerce')

    # Infer "NA" bedrooms using floor area if possible
    rental['no_of_bedroom'] = rental['no_of_bedroom'].replace({'NA': None, 'N/A': None, 'nan': None})
    # If no_of_bedroom is NA, infer from floor_area_(sqft)
    def infer_bedroom(row):
        if pd.isna(row['no_of_bedroom']):
            sqft = row['floor_area_(sqft)']
            if sqft < 600:
                return 1
            elif sqft < 800:
                return 2
            elif sqft < 1200:
                return 3
            elif sqft < 1600:
                return 4
            else:
                return 5
        try:
            return int(row['no_of_bedroom'])
        except (ValueError, TypeError):
            return pd.NA
    rental['no_of_bedroom'] = rental.apply(infer_bedroom, axis=1)
    rental['no_of_bedroom'] = rental['no_of_bedroom'].astype('Int64')  # Use Int64 to handle NaN

    
    return rental, transactions, condos

@st.cache_data
def load_district_map() -> dict[str, str]:
    """Load district name mapping from JSON file"""
    with open(DISTRICT_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_with_condo_data(df, condos):
    """Enrich transaction/rental data with condo info"""
    print(df, condos)
    return pd.merge(
        df,
        condos[['original_name', 'nearest_mrt', 'mrt_distance', 'number_of_units', 'completion', 'latitude', 'longitude']],
        left_on='project_name',
        right_on='original_name',
        how='left'
    ).drop(columns=['original_name'])

@st.cache_data
def load_mrt_data():
    """Load MRT station data"""
    # This is a sample - replace with your actual MRT station data
    mrt_stations = pd.read_csv(os.path.join(DATA_DIR, "MRT Stations.csv"))
    # lower case the Latitude Loingtude column names
    mrt_stations.columns = mrt_stations.columns.str.lower()
    return mrt_stations
def create_filters(df):
    st.sidebar.header("Filters")
    d, p, b = sorted(df['postal_district'].dropna().unique()), df['project_name'], df['no_of_bedroom']
    district_name_map = load_district_map()
    district_options = [('ALL', 'All Districts')] + [(s, f"{district_name_map.get(str(s).zfill(2), '')}") for s in d]
    sd = st.sidebar.multiselect(
        "Postal District", options=[opt[0] for opt in district_options], default=[district_options[1][0]],
        format_func=lambda x: dict(district_options).get(x, f"District {x}"))
    sd = d if 'ALL' in sd else [i for i in sd if i != 'ALL']
    sp = st.sidebar.multiselect("Projects", ['ALL']+sorted(p[df['postal_district'].isin(sd)].dropna().unique()), default=['ALL'])
    sp = sorted(p.unique()) if 'ALL' in sp else [i for i in sp if i != 'ALL']
    sb = st.sidebar.multiselect("Bedrooms", ['ALL']+sorted(b.dropna().unique()), default=[0, 1])
    sb = sorted(b.dropna().unique()) if 'ALL' in sb else [i for i in sb if i != 'ALL']

    # --- New Filters ---
    # Rent PSF slider
    rent_psf_min = float(df['rent_psf'].min(skipna=True)) if 'rent_psf' in df else 0.0
    rent_psf_max = float(df['rent_psf'].max(skipna=True)) if 'rent_psf' in df else 10.0
    rent_psf_range = st.sidebar.slider(
        "Rent PSF Range ($)", min_value=round(rent_psf_min, 2), max_value=round(rent_psf_max, 2),
        value=(3.0, 8.0), step=0.1)
    include_na_rent_psf = st.sidebar.checkbox("Include NA Rent PSF", value=False)

    # Distance to MRT slider
    mrt_distance_min = float(df['mrt_distance'].min(skipna=True)) if 'mrt_distance' in df else 0.0
    mrt_distance_max = float(df['mrt_distance'].max(skipna=True)) if 'mrt_distance' in df else 2000.0
    mrt_distance_range = st.sidebar.slider(
        "Distance to MRT (m)", min_value=int(mrt_distance_min), max_value=int(mrt_distance_max),
        value=(int(mrt_distance_min), 800), step=10)
    include_na_mrt_distance = st.sidebar.checkbox("Include NA MRT Distance", value=True)

    # Completion year slider
    completion_min = int(df['completion'].min(skipna=True)) if 'completion' in df and df['completion'].notna().any() else 1980
    completion_max = int(df['completion'].max(skipna=True)) if 'completion' in df and df['completion'].notna().any() else 2025
    completion_range = st.sidebar.slider(
        "Completion Year", min_value=completion_min, max_value=completion_max,
        value=(2000, completion_max), step=1)
    include_na_completion = st.sidebar.checkbox("Include NA Completion Year", value=True)

    return {
        'postal_district': sd,
        'project_name': sp,
        'no_of_bedroom': sb,
        'rent_psf_range': rent_psf_range,
        'include_na_rent_psf': include_na_rent_psf,
        'mrt_distance_range': mrt_distance_range,
        'include_na_mrt_distance': include_na_mrt_distance,
        'completion_range': completion_range,
        'include_na_completion': include_na_completion,
    }

def merge_condo(df, condos):
    # Merge but keep the left (df) column if there are duplicate column names
    merged = pd.merge(df,condos,left_on='project_name',right_on='original_name',how='left',suffixes=('', '_condo'))
    for col in df.columns:
        dup_col = f"{col}_condo"
        if dup_col in merged.columns:
            merged = merged.drop(columns=[dup_col])
    return merged

def plot_trend(df, x, y, color, title, yaxis="($)"):
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_layout(xaxis_title="", yaxis_title=f"{y} {yaxis}")
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Singapore Property Dashboard")
    st.title("🏢 Singapore Property Analysis")
    st.markdown("Comprehensive analysis combining URA rental/transaction data with MRT proximity.")

    rental, transactions, condos = load_data()
    rental = merge_condo(rental, condos)
    filters = create_filters(rental)

    # Apply all filters to rental data
    rf = rental[
        (rental['postal_district'].isin(filters['postal_district'])) &
        (rental['project_name'].isin(filters['project_name'])) &
        (rental['no_of_bedroom'].isin(filters['no_of_bedroom']))
    ]

    # Rent PSF filter
    # Use latest year
    rf_2025 = rf[rf['lease_commencement_date'].dt.year == rf['lease_commencement_date'].dt.year.max()]
    # Group by project_name and get average rent_psf within the year
    rent_psf_min, rent_psf_max = filters['rent_psf_range']
    avg_rent_psf = rf_2025.groupby('project_name')['rent_psf'].mean()
    # Filter project names within the selected rent_psf range
    valid_projects = avg_rent_psf[(avg_rent_psf >= rent_psf_min) & (avg_rent_psf <= rent_psf_max)].index.tolist()
    rf = rf[rf['project_name'].isin(valid_projects)]
    if filters['include_na_rent_psf']:
        rf = rf[(rf['project_name'].isin(valid_projects)) | (rf['rent_psf'].isna())]
    else:
        rf = rf[rf['project_name'].isin(valid_projects)]

    # MRT distance filter
    
    # Merge filtered rental data with condo info for further filtering
    mrt_distance_min, mrt_distance_max = filters['mrt_distance_range']
    if filters['include_na_mrt_distance']:
        rf = rf[(rf['mrt_distance'].between(mrt_distance_min, mrt_distance_max)) | (rf['mrt_distance'].isna())]
    else:
        rf = rf[rf['mrt_distance'].between(mrt_distance_min, mrt_distance_max)]

    # Completion year filter
    completion_min, completion_max = filters['completion_range']
    if filters['include_na_completion']:
        rf = rf[((rf['completion'] >= completion_min) & (rf['completion'] <= completion_max)) | (rf['completion'].isna())]
    else:
        rf = rf[(rf['completion'] >= completion_min) & (rf['completion'] <= completion_max)]

    rf = rf[rf['rent_psf'].notna()]
    tf = transactions[(transactions['postal_district'].isin(filters['postal_district'])) &
                      (transactions['project_name'].isin(filters['project_name']))]

    col1, col2, col3 = st.columns(3)
    col1.metric("Projects", len(rf['project_name'].unique()))
    col2.metric("Rental Listings", len(rf))
    col3.metric("Sale Transactions", len(tf))

    st.header("Price Trends Over Time")
    rt = rf.groupby(['lease_commencement_date','project_name'])['rent_psf'].mean().reset_index()
    st.plotly_chart(plot_trend(rt, 'lease_commencement_date','rent_psf','project_name','Rental PSF Trend'), use_container_width=True)
    st.plotly_chart(plot_trend(tf.groupby(['sale_date','project_name'])['sale_psf'].mean().reset_index(), 'sale_date','sale_psf','project_name','Sale PSF Trend'), use_container_width=True)

    st.header("Rental Yield Analysis")
    yt = pd.merge_asof(rt.sort_values('lease_commencement_date'), tf.groupby(['sale_date','project_name'])['sale_psf'].mean().reset_index().sort_values('sale_date'), left_on='lease_commencement_date', right_on='sale_date', by='project_name', direction='nearest')
    yt['yield_pct'] = (yt['rent_psf'] * 12 / yt['sale_psf']) * 100
    st.plotly_chart(plot_trend(yt, 'lease_commencement_date','yield_pct','project_name','Estimated Rental Yield', yaxis="%"), use_container_width=True)

    st.header("MRT Proximity Analysis")
    rm = merge_condo(rf, condos)
    if not rm.empty:
        y = rm['lease_commencement_date'].dt.year.max()
        fig = px.scatter(rm[rm['lease_commencement_date'].dt.year==y], x='mrt_distance', y='rent_psf', color='project_name', hover_data=['nearest_mrt', 'completion', 'no_of_bedroom'], title=f"MRT Distance vs Rental PSF ({y})")
        fig.update_layout(xaxis_title="Distance to MRT (m)", yaxis_title="Rental PSF ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Completion Year vs Rental PSF (Scatter)")
    comp_data = merge_condo(rf, condos)
    comp_data = comp_data[comp_data['completion'].notna()]
    y = comp_data['lease_commencement_date'].dt.year.max()
    scatter_df = comp_data[comp_data['lease_commencement_date'].dt.year == y]

    fig = px.violin(
        scatter_df,
        x='completion',
        y='rent_psf',
        color='project_name',
        box=False,
        # points='all',
        hover_data=['nearest_mrt', 'mrt_distance'],
        title=f"Completion Year vs Rental PSF ({y})",
    )
    # Make the violin plot wider by adjusting the width of each violin
    fig.update_traces(width=0.9)  # Default is 0.6, increase to make violins fatter

    # Overlay median rental psf as a line
    median_psf = scatter_df.groupby('completion')['rent_psf'].median().reset_index()
    fig.add_scatter(
        x=median_psf['completion'],
        y=median_psf['rent_psf'],
        mode='lines+markers',
        name='Median Rental PSF',
        line=dict(color='white', width=2, dash='dash'),
        marker=dict(size=6, color='white')
    )

    fig.update_layout(xaxis_title="Completion Year", yaxis_title="Rental PSF ($)")
    st.plotly_chart(fig, use_container_width=True)




    st.header("Project Summary")
    rf_2025 = rf[rf['lease_commencement_date'].dt.year == 2025]
    summary = pd.merge(rf_2025.groupby('project_name')['rent_psf'].mean().reset_index(), tf.groupby('project_name')['sale_psf'].mean().reset_index(), on='project_name', how='outer')
    summary = pd.merge(summary, condos[['original_name','nearest_mrt','mrt_distance','number_of_units','completion']], left_on='project_name', right_on='original_name', how='left').drop(columns=['original_name'])
    summary['yield_pct'] = (summary['rent_psf'] * 12 / summary['sale_psf']) * 100
    summary = summary[summary['rent_psf'].notna()]
    if not summary.empty:
        st.dataframe(summary[['project_name','rent_psf','sale_psf','yield_pct','nearest_mrt','mrt_distance','number_of_units','completion']], use_container_width=True)
        st.download_button("Download Summary Data", summary.to_csv(index=False), "property_summary.csv", "text/csv")

    st.header("Property & MRT Locations")
    mrt_stations = load_mrt_data()
    recent_map = merge_condo(rf, condos)
    if not recent_map.empty:
        y = recent_map['lease_commencement_date'].dt.year.max()
        mp = recent_map[recent_map['lease_commencement_date'].dt.year == y]
        cp = mp.groupby(['project_name','latitude','longitude']).agg({'rent_psf':'mean','mrt_distance':'mean'}).reset_index()
        cp['size'] = 5 + (cp['rent_psf'] / cp['rent_psf'].max() * 15)
        cp['color'] = '#1f77b4'
        mrts = mrt_stations[mrt_stations['stn_name'].isin(mp['nearest_mrt'].unique())].copy()
        mrts['size'], mrts['color'] = 10, '#ff0000'
        map_df = pd.concat([cp[['latitude','longitude','color','size','project_name']], mrts.rename(columns={'stn_name':'project_name'})[['latitude','longitude','color','size','project_name']]])
        st.map(map_df, latitude='latitude', longitude='longitude', color='color', size='size', zoom=12, use_container_width=True)
        st.markdown("""
        **Map Legend:**
        - <span style='color:blue; font-weight:bold'>Blue circles</span>: Condos (size = rent PSF)
        - <span style='color:red; font-weight:bold'>Red circles</span>: MRT stations
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
