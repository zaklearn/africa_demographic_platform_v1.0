import streamlit as st
import pandas as pd
from datetime import datetime
from config import Config
from api_service import WorldBankAPIService
from analytics import DemographicAnalytics
from visualization import Visualizations
from cache_manager import CacheManager
from debug_tools import DebugTools

# Page config
st.set_page_config(
    page_title="🌍 Africa Demographics Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .dividend-status {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #f0f0f0, #e0e0e0);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_demographic_data(use_core_only: bool = False):
    """Load and cache demographic data"""
    service = WorldBankAPIService()
    return service.load_all_demographic_data(use_core_only=use_core_only)

def main():
    # Setup
    Config.setup_directories()
    
    # Initialize components
    api_service = WorldBankAPIService()
    analytics = DemographicAnalytics()
    viz = Visualizations()
    cache = CacheManager()
    debug = DebugTools()
    
    # Sidebar navigation - COMPLETE LIST per cahier des charges
    st.sidebar.markdown("## 🌍 Navigation")
    page = st.sidebar.selectbox(
        "Choose a view:",
        [
            "Continental Overview",      # Vue continentale avec tracker dividend
            "Country Profiles",         # Profils pays avec pyramides animées
            "Trend Analysis",           # Comparaison multi-pays
            "Clustering Analysis",      # Intelligence ML clustering
            "Data Explorer",           # Explorateur avec export personnalisé
            "API Status & Debug",      # Statut API temps réel
            "Cache Management"         # Gestion cache
        ]
    )
    
    # Data source info
    st.sidebar.markdown("## 📊 Data Source")
    st.sidebar.info("""
    **World Bank Open Data API**
    - Real-time demographic data
    - 54 African countries
    - Years: 1990-2023
    - Population-weighted metrics
    """)
    
    # Data loading options
    st.sidebar.markdown("## ⚙️ Data Loading")
    use_core_indicators = st.sidebar.checkbox("Use core indicators only (faster)", value=False)
    
    if st.sidebar.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Cache status
    st.sidebar.markdown("## 💾 Cache Status")
    cache_info = cache.get_cache_info()
    
    if cache_info['total_files'] > 0:
        st.sidebar.success(f"📁 {cache_info['total_files']} files cached")
        st.sidebar.caption(f"💾 {cache_info['total_size_mb']:.1f} MB")
        
        if st.sidebar.button("🗑️ Clear Cache"):
            cache.clear_cache()
            st.rerun()
    else:
        st.sidebar.info("📁 No cache files")
    
    # Main content based on selected page
    if page == "Continental Overview":
        """📊 Vue Continentale - Métriques clés + Tracker dividende démographique"""
        try:
            with st.spinner("Loading demographic data..."):
                df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error("Failed to load data. Check API Status & Debug page.")
                return
            
            # Show data status
            st.sidebar.success(f"✅ {df['country_iso2'].nunique()}/54 countries")
            
            # Create main overview with dividend tracker
            viz.create_continental_overview(df)
            
            # Interactive features
            
            if not df.empty:
                st.markdown("---")

                st.markdown("### 🗺️ Interactive Africa Map")
                
                available_indicators = [col for col in df.columns 
                                        if col in ['total_fertility_rate', 'population_growth_rate', 'median_age', 'dividend_score']]

                if available_indicators:
                    selected_indicator = st.selectbox(
                        "Select indicator:",
                        available_indicators,
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    year_options = sorted(df['year'].unique(), reverse=True)
                    selected_year = st.selectbox("Select year:", year_options)
                    
                    # La carte est affichée en dehors de toute colonne
                    viz.create_africa_map(df, selected_indicator, selected_year)

                st.markdown("---")

                st.markdown("### 📊 Key Statistics")
                
                # Création de deux colonnes pour les statistiques
                col_stats1, col_stats2 = st.columns(2)
                
                latest_year = df['year'].max()
                latest_data = df[df['year'] == latest_year]

                with col_stats1:
                    if 'total_fertility_rate' in df.columns:
                        st.markdown("**🏆 Lowest Fertility Rates:**")
                        top_tfr = latest_data.nsmallest(5, 'total_fertility_rate')[['country_name', 'total_fertility_rate']]
                        if not top_tfr.empty:
                            st.dataframe(top_tfr, hide_index=True)

                with col_stats2:
                    if 'population_growth_rate' in df.columns:
                        st.markdown("**📈 Highest Growth Rates:**")
                        top_growth = latest_data.nlargest(5, 'population_growth_rate')[['country_name', 'population_growth_rate']]
                        if not top_growth.empty:
                            st.dataframe(top_growth, hide_index=True)
        
        except Exception as e:
            st.error(f"Error loading overview: {e}")
            st.info("Try the API Status & Debug page for diagnostics")
    
    elif page == "Country Profiles":
        """🇳🇬 Profils Pays - Pyramides de population animées + tendances historiques"""
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty or 'country_name' not in df.columns:
                st.error("No country data available")
                return
            
            st.markdown('<div class="section-header">🇳🇬 Country Profiles - Population Pyramids & Historical Trends</div>', unsafe_allow_html=True)
            
            # Country selector
            available_countries = sorted(df['country_name'].unique())
            selected_country = st.selectbox("Select country for detailed profile:", available_countries)
            
            country_data = df[df['country_name'] == selected_country].copy()
            
            if not country_data.empty:
                # Country overview metrics
                latest_data = country_data[country_data['year'] == country_data['year'].max()].iloc[0]
                
                st.markdown(f"### 📊 {selected_country} - Profile Overview ({latest_data['year']:.0f})")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'total_fertility_rate' in latest_data.index and pd.notna(latest_data['total_fertility_rate']):
                        st.metric("Fertility Rate", f"{latest_data['total_fertility_rate']:.1f}")
                    else:
                        st.metric("Fertility Rate", "N/A")
                
                with col2:
                    if 'total_population' in latest_data.index and pd.notna(latest_data['total_population']):
                        pop_millions = latest_data['total_population'] / 1e6
                        st.metric("Population", f"{pop_millions:.1f}M")
                    else:
                        st.metric("Population", "N/A")
                
                with col3:
                    if 'median_age' in latest_data.index and pd.notna(latest_data['median_age']):
                        st.metric("Median Age", f"{latest_data['median_age']:.1f} years")
                    else:
                        st.metric("Median Age", "N/A")
                
                with col4:
                    if 'dividend_status' in latest_data.index:
                        status = latest_data['dividend_status']
                        color = {'High Opportunity': '🟢', 'Opening Window': '🟡', 'Limited Window': '🔴', 'No Window': '⚪'}.get(status, '❓')
                        st.metric("Dividend Status", f"{color} {status}")
                    else:
                        st.metric("Dividend Status", "N/A")
                
                # Population Pyramids - Animated feature
                st.markdown("### 📈 Population Pyramid Analysis")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    pyramid_year = st.selectbox("Select year:", sorted(country_data['year'].unique(), reverse=True))
                    animate_pyramid = st.checkbox("🎬 Animate over time (1990-2023)")
                    
                    if animate_pyramid:
                        st.info("🎬 Animation will show demographic transition over time")
                
                with col1:
                    viz.create_population_pyramid(df, selected_country, pyramid_year, animate_pyramid)
                
                # Historical trends for all indicators
                st.markdown("### 📈 Historical Trends Analysis")
                
                numeric_cols = country_data.select_dtypes(include=['number']).columns
                available_for_trend = [col for col in numeric_cols 
                                     if col not in ['year'] and country_data[col].notna().sum() > 3]
                
                if available_for_trend:
                    selected_indicators = st.multiselect(
                        "Select indicators to display:",
                        available_for_trend,
                        default=available_for_trend[:3],  # Default to first 3
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    if selected_indicators:
                        viz.create_trend_comparison(country_data, [selected_country], selected_indicators)
        
        except Exception as e:
            st.error(f"Error in country profiles: {e}")
    
    elif page == "Trend Analysis":
        """📈 Analyse de Tendances - Comparaison multi-pays avec graphiques interactifs"""
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error("No data available for trend analysis")
                return
            
            st.markdown('<div class="section-header">📈 Trend Analysis - Multi-Country Comparison</div>', unsafe_allow_html=True)
            
            # Country and indicator selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Select Countries for Comparison:**")
                available_countries = sorted(df['country_name'].unique())
                selected_countries = st.multiselect(
                    "Countries (select 2-6 for best visualization):",
                    available_countries,
                    default=available_countries[:4] if len(available_countries) >= 4 else available_countries,
                    max_selections=6
                )
            
            with col2:
                st.markdown("**Select Indicators:**")
                numeric_cols = df.select_dtypes(include=['number']).columns
                available_indicators = [col for col in numeric_cols 
                                      if col not in ['year'] and df[col].notna().sum() > 50]
                
                selected_indicators = st.multiselect(
                    "Demographic indicators:",
                    available_indicators,
                    default=['total_fertility_rate', 'population_growth_rate'] if all(x in available_indicators for x in ['total_fertility_rate', 'population_growth_rate']) else available_indicators[:2],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            if selected_countries and selected_indicators:
                # Multi-country comparison
                viz.create_trend_comparison(df, selected_countries, selected_indicators)
                
                # Statistical analysis
                st.markdown("### 📊 Statistical Analysis")
                
                comparison_analysis = analytics.generate_country_comparison(df, selected_countries, selected_indicators)
                
                if comparison_analysis:
                    # Latest values comparison
                    st.markdown("#### Latest Values Comparison")
                    for indicator, data in comparison_analysis['latest_comparison'].items():
                        st.markdown(f"**{indicator.replace('_', ' ').title()}:**")
                        ranking_df = pd.DataFrame(data['ranking'], columns=['Country', 'Value'])
                        st.dataframe(ranking_df, hide_index=True)
                    
                    # Correlations
                    if comparison_analysis['correlations']:
                        st.markdown("#### Indicator Correlations")
                        corr_data = []
                        for pair, corr in comparison_analysis['correlations'].items():
                            indicators_pair = pair.replace('_vs_', ' vs ').replace('_', ' ').title()
                            corr_data.append({'Indicators': indicators_pair, 'Correlation': f"{corr:.3f}"})
                        
                        if corr_data:
                            st.dataframe(pd.DataFrame(corr_data), hide_index=True)
            else:
                st.info("Please select countries and indicators to begin analysis")
        
        except Exception as e:
            st.error(f"Error in trend analysis: {e}")
    
    elif page == "Clustering Analysis":
        """🔬 Clustering Intelligence - Regroupement ML par profils démographiques"""
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error("No data available for clustering analysis")
                return
            
            st.markdown('<div class="section-header">🔬 Clustering Intelligence - ML Demographic Profiling</div>', unsafe_allow_html=True)
            
            # Clustering controls
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_year = st.selectbox(
                    "Select year for clustering analysis:",
                    sorted(df['year'].unique(), reverse=True)
                )
            
            with col2:
                st.info(f"""
                **Clustering Method:** K-Means with {Config.CLUSTERING_CONFIG['n_clusters']} clusters
                **Indicators Used:** {', '.join(Config.CLUSTERING_CONFIG['indicators'])}
                **Classification:** Demographic transition stages
                """)
            
            # Perform clustering
            clustered_data = analytics.get_country_clusters(df, cluster_year)
            
            if not clustered_data.empty:
                # Clustering visualization
                st.markdown("### 🎯 Country Clusters by Demographic Profile")
                
                fig_scatter, cluster_summary = viz.create_clustering_visualization(clustered_data)
                
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Cluster analysis table
                if cluster_summary is not None:
                    st.markdown("### 📊 Cluster Characteristics")
                    st.dataframe(cluster_summary)
                
                # Countries by cluster
                st.markdown("### 🗂️ Countries by Demographic Transition Stage")
                
                if 'cluster_label' in clustered_data.columns:
                    for cluster_label in sorted(clustered_data['cluster_label'].unique()):
                        cluster_countries = clustered_data[clustered_data['cluster_label'] == cluster_label]['country_name'].tolist()
                        
                        # Color coding for clusters
                        if 'High Fertility' in cluster_label:
                            color = "🔴"
                        elif 'Moderate' in cluster_label:
                            color = "🟡"
                        elif 'Advanced' in cluster_label:
                            color = "🟢"
                        else:
                            color = "🔵"
                        
                        st.markdown(f"**{color} {cluster_label}** ({len(cluster_countries)} countries):")
                        st.write(", ".join(cluster_countries))
            else:
                st.warning("Insufficient data for clustering analysis. Try a different year or check data availability.")
        
        except Exception as e:
            st.error(f"Error in clustering analysis: {e}")
    
    elif page == "Data Explorer":
        """🔍 Explorateur de Données - Filtrage avancé + export personnalisé"""
        try:
            df = load_demographic_data(use_core_only=use_core_indicators)
            
            if df.empty:
                st.error("No data available for exploration")
                return
            
            st.markdown('<div class="section-header">🔍 Data Explorer - Advanced Filtering & Custom Export</div>', unsafe_allow_html=True)
            
            # Advanced filters
            st.markdown("### 🎛️ Advanced Filters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Countries:**")
                available_countries = sorted(df['country_name'].unique())
                selected_countries_explorer = st.multiselect(
                    "Select countries:",
                    available_countries,
                    default=available_countries,
                    key="explorer_countries"
                )
            
            with col2:
                st.markdown("**Years:**")
                year_range = st.slider(
                    "Year range:",
                    min_value=int(df['year'].min()),
                    max_value=int(df['year'].max()),
                    value=(int(df['year'].min()), int(df['year'].max())),
                    key="explorer_years"
                )
            
            with col3:
                st.markdown("**Indicators:**")
                numeric_cols = df.select_dtypes(include=['number']).columns
                available_indicators_explorer = [col for col in numeric_cols if col not in ['year']]
                selected_indicators_explorer = st.multiselect(
                    "Select indicators:",
                    available_indicators_explorer,
                    default=available_indicators_explorer[:5],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="explorer_indicators"
                )
            
            # Apply filters
            filtered_df = df[
                (df['country_name'].isin(selected_countries_explorer)) &
                (df['year'] >= year_range[0]) &
                (df['year'] <= year_range[1])
            ].copy()
            
            # Select columns
            display_cols = ['country_name', 'year'] + selected_indicators_explorer
            display_df = filtered_df[display_cols].copy()
            
            # Data summary
            st.markdown("### 📊 Filtered Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Countries", len(selected_countries_explorer))
            with col2:
                st.metric("Years", f"{year_range[1] - year_range[0] + 1}")
            with col3:
                st.metric("Indicators", len(selected_indicators_explorer))
            with col4:
                st.metric("Records", len(display_df))
            
            # Display data
            st.markdown("### 📋 Filtered Dataset")
            st.dataframe(display_df, use_container_width=True)
            
            # Statistics
            if selected_indicators_explorer:
                st.markdown("### 📈 Descriptive Statistics")
                
                stats_df = display_df[selected_indicators_explorer].describe().round(2)
                st.dataframe(stats_df)
            
            # Export options
            st.markdown("### 📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    "📊 Download as CSV",
                    csv_data,
                    f"africa_demographics_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key="download_csv"
                )
            
            with col2:
                # JSON export
                json_data = display_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📋 Download as JSON",
                    json_data,
                    f"africa_demographics_filtered_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    key="download_json"
                )
        
        except Exception as e:
            st.error(f"Error in data explorer: {e}")
    
    elif page == "API Status & Debug":
        """🔌 Statut API - Connexion temps réel + vérification qualité"""
        st.markdown('<div class="section-header">🔌 API Status & Debug Tools</div>', unsafe_allow_html=True)
        
        # Basic connectivity test
        if st.button("🌐 Test Basic Connectivity"):
            with st.spinner("Testing connectivity..."):
                results = debug.test_basic_connectivity()
                
                for test in results['tests']:
                    if test['status'] == 'PASS':
                        st.success(f"✅ {test['test']}: {test['details']}")
                    else:
                        st.error(f"❌ {test['test']}: {test['details']}")
        
        st.markdown("---")
        
        # Comprehensive test
        if st.button("🧪 Run Comprehensive Test"):
            debug.run_comprehensive_test()
        
        st.markdown("---")
        
        # Debug raw data
        if st.button("🔍 Debug Raw Data"):
            api_service.debug_raw_data()
        
        st.markdown("---")
        
        # Individual indicator test
        st.markdown("### 🔍 Test Individual Indicators")
        
        test_indicator = st.selectbox(
            "Select indicator to test:",
            list(Config.CORE_INDICATORS.keys()),
            format_func=lambda x: Config.CORE_INDICATORS[x].replace('_', ' ').title()
        )
        
        if st.button("Test Selected Indicator"):
            with st.spinner("Testing indicator..."):
                result = debug.test_single_indicator(test_indicator)
                
                if result.get('status') == 'SUCCESS':
                    st.success(f"✅ Success: {result.get('data_count', 0)} records")
                    if result.get('sample_record'):
                        st.json(result['sample_record'])
                else:
                    st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    elif page == "Cache Management":
        """💾 Gestion Cache - Monitoring et contrôles"""
        st.markdown('<div class="section-header">💾 Cache Management</div>', unsafe_allow_html=True)
        
        cache_info = cache.get_cache_info()
        
        if cache_info['total_files'] == 0:
            st.info("No cached files found")
        else:
            st.success(f"Found {cache_info['total_files']} cached files ({cache_info['total_size_mb']:.1f} MB)")
            
            # Cache details
            if cache_info['files']:
                cache_df = pd.DataFrame(cache_info['files'])
                cache_df['age_hours'] = cache_df['age_hours'].round(1)
                cache_df['size_kb'] = cache_df['size_kb'].round(1)
                st.dataframe(cache_df, use_container_width=True)
        
        # Cache actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Cache"):
                cache.clear_cache()
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Info"):
                st.rerun()
        
        with col3:
            st.info(f"Cache expires after {Config.CACHE_HOURS} hours")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>🌍 Africa Demographics Platform - Complete Implementation - Conception et developpement Zakaria Benhoumad</p>
        <p>📊 Features: Continental Overview • Country Profiles • Trend Analysis • ML Clustering • Data Explorer</p>
        <p>🔗 World Bank API • Real-time Data • {datetime.now().strftime("%B %d, %Y")}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                