import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analytics import DemographicAnalytics
from config import Config

class Visualizations:
    """Complete visualization components for demographics platform"""
    
    def __init__(self):
        self.analytics = DemographicAnalytics()
    
    def create_continental_overview(self, df: pd.DataFrame):
        """Main continental dashboard with dividend tracker"""
        st.markdown('<h1 style="text-align: center; color: #2E7D32;">🌍 Africa Demographics Platform</h1>', unsafe_allow_html=True)
        
        if df.empty:
            st.error("No demographic data available")
            return
        
        # Calculate continental metrics
        continental_metrics = self.analytics.calculate_continental_metrics(df)
        
        if 'error' in continental_metrics:
            st.error(f"❌ {continental_metrics['error']}")
            return
        
        # Population highlight
        pop_millions = continental_metrics.get('total_population_millions', 0)
        if pop_millions > 0:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #4CAF50, #2E7D32); color: white; 
                       padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.2rem; margin: 1rem 0;">
                🌍 <strong>Africa Population (Real API Data)</strong><br>
                <strong>{pop_millions:,.0f} million people</strong> ({pop_millions/1000:.2f} billion)<br>
                📊 Calculated from World Bank country-level data
            </div>
            ''', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌍 Population", f"{pop_millions:.0f}M" if pop_millions > 0 else "N/A")
        
        with col2:
            median_age = continental_metrics.get('weighted_median_age', np.nan)
            st.metric("👥 Median Age", f"{median_age:.1f} years" if not np.isnan(median_age) else "N/A")
        
        with col3:
            tfr = continental_metrics.get('weighted_tfr', np.nan)
            st.metric("👶 Fertility Rate", f"{tfr:.1f}" if not np.isnan(tfr) else "N/A")
        
        with col4:
            growth_rate = continental_metrics.get('weighted_growth_rate', np.nan)
            st.metric("📈 Growth Rate", f"{growth_rate:.1f}%" if not np.isnan(growth_rate) else "N/A")
        
        # Demographic Dividend Tracker
        st.markdown("### 🎯 Demographic Dividend Status - Real-time Tracker")
        self.create_dividend_tracker(continental_metrics.get('dividend_distribution', {}))
        
        # Show analysis metadata
        metadata = continental_metrics.get('metadata', {})
        if metadata:
            countries_analyzed = metadata.get('countries_with_demographic_data', 0)
            st.info(f"📊 Analysis based on {countries_analyzed} countries with complete data")
    
    def create_dividend_tracker(self, dividend_dist: dict):
        """Create demographic dividend status dashboard"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_count = dividend_dist.get('High Opportunity', 0)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #00C851, #007E33); color: white; 
                       padding: 0.5rem; border-radius: 5px; text-align: center;">
                <h3>🟢 Peak Window</h3>
                <h2>{high_count} countries</h2>
                <p>Optimal dividend opportunity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            opening_count = dividend_dist.get('Opening Window', 0)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffbb33, #FF8800); color: white; 
                       padding: 0.5rem; border-radius: 5px; text-align: center;">
                <h3>🟡 Opening Window</h3>
                <h2>{opening_count} countries</h2>
                <p>Dividend window opening</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            limited_count = dividend_dist.get('Limited Window', 0)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff4444, #CC0000); color: white; 
                       padding: 0.5rem; border-radius: 5px; text-align: center;">
                <h3>🔴 Limited Window</h3>
                <h2>{limited_count} countries</h2>
                <p>Narrow opportunity window</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            no_window_count = dividend_dist.get('No Window', 0)
            st.markdown(f"""
            <div style="background: #6c757d; color: white; padding: 0.5rem; 
                       border-radius: 5px; text-align: center;">
                <h3>⚪ No Window</h3>
                <h2>{no_window_count} countries</h2>
                <p>No dividend opportunity</p>
            </div>
            """, unsafe_allow_html=True)
    

    def create_africa_map(self, df: pd.DataFrame, indicator: str = 'total_fertility_rate', year: int = 2023):
        """Interactive choropleth map focused on Africa"""
        
        map_data = df[df['year'] == year].copy()
        if map_data.empty or indicator not in map_data.columns:
            st.error(f"No data available for {indicator} in {year}")
            return
        
        map_data = map_data.dropna(subset=[indicator])
        if map_data.empty:
            return
        
        # ISO2 to ISO3 mapping
        iso2_to_iso3 = {
            'DZ': 'DZA', 'AO': 'AGO', 'BJ': 'BEN', 'BW': 'BWA', 'BF': 'BFA',
            'BI': 'BDI', 'CM': 'CMR', 'CV': 'CPV', 'CF': 'CAF', 'TD': 'TCD',
            'KM': 'COM', 'CG': 'COG', 'CD': 'COD', 'CI': 'CIV', 'DJ': 'DJI',
            'EG': 'EGY', 'GQ': 'GNQ', 'ER': 'ERI', 'SZ': 'SWZ', 'ET': 'ETH',
            'GA': 'GAB', 'GM': 'GMB', 'GH': 'GHA', 'GN': 'GIN', 'GW': 'GNB',
            'KE': 'KEN', 'LS': 'LSO', 'LR': 'LBR', 'LY': 'LBY', 'MG': 'MDG',
            'MW': 'MWI', 'ML': 'MLI', 'MR': 'MRT', 'MU': 'MUS', 'MA': 'MAR',
            'MZ': 'MOZ', 'NA': 'NAM', 'NE': 'NER', 'NG': 'NGA', 'RW': 'RWA',
            'ST': 'STP', 'SN': 'SEN', 'SC': 'SYC', 'SL': 'SLE', 'SO': 'SOM',
            'ZA': 'ZAF', 'SS': 'SSD', 'SD': 'SDN', 'TZ': 'TZA', 'TG': 'TGO',
            'TN': 'TUN', 'UG': 'UGA', 'ZM': 'ZMB', 'ZW': 'ZWE'
        }
        
        map_data['country_iso3'] = map_data['country_iso2'].map(iso2_to_iso3)
        map_data = map_data.dropna(subset=['country_iso3'])
        
        color_scale = 'Viridis'
        
        fig = px.choropleth(
            map_data,
            locations='country_iso3',
            color=indicator,
            hover_name='country_name',
            color_continuous_scale=color_scale,
            title=f'Africa: {indicator.replace("_", " ").title()} ({year})'
        )
        
        # Focus sur l'Afrique
        fig.update_geos(
            projection_type="natural earth",
            showframe=False,
            showcoastlines=True,
            lonaxis_range=[-20, 55],
            lataxis_range=[-40, 40]
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


    
    def create_population_pyramid(self, df: pd.DataFrame, country_name: str, year: int = 2023, animate: bool = False):
        """Create population pyramid with animation support"""
        
        country_data = df[df['country_name'] == country_name].copy()
        
        if country_data.empty:
            st.error(f"No data available for {country_name}")
            return
        
        # If animate, show all years, otherwise just selected year
        if animate:
            animation_years = sorted(country_data['year'].unique())
            pyramid_data = country_data
        else:
            pyramid_data = country_data[country_data['year'] == year]
            animation_years = [year]
        
        if pyramid_data.empty:
            st.error(f"No data for {country_name} in {year}")
            return
        
        # Generate age structure data based on demographic indicators
        age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', 
                      '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', 
                      '70-74', '75-79', '80+']
        
        fig = go.Figure()
        
        # Create population pyramid for each year
        for yr in animation_years:
            year_data = pyramid_data[pyramid_data['year'] == yr]
            if year_data.empty:
                continue
            
            # Generate realistic population distribution
            tfr = year_data['total_fertility_rate'].iloc[0] if 'total_fertility_rate' in year_data.columns else 4.0
            pop_0_14 = year_data['population_0_14_percent'].iloc[0] if 'population_0_14_percent' in year_data.columns else 40
            pop_65_plus = year_data['population_65_plus_percent'].iloc[0] if 'population_65_plus_percent' in year_data.columns else 3
            
            # Calculate age distribution
            population_by_age = self._generate_age_distribution(tfr, pop_0_14, pop_65_plus)
            
            # Split by gender (assume 51% male, 49% female)
            male_pop = [-pop * 0.51 for pop in population_by_age]
            female_pop = [pop * 0.49 for pop in population_by_age]
            
            # Add traces
            fig.add_trace(go.Bar(
                y=age_groups,
                x=male_pop,
                name='Male',
                orientation='h',
                marker_color='lightblue',
                visible=(yr == animation_years[0]),
                hovertemplate='<b>Male %{y}</b><br>Population: %{customdata:.1f}%<extra></extra>',
                customdata=[abs(x) for x in male_pop]
            ))
            
            fig.add_trace(go.Bar(
                y=age_groups,
                x=female_pop,
                name='Female',
                orientation='h',
                marker_color='pink',
                visible=(yr == animation_years[0]),
                hovertemplate='<b>Female %{y}</b><br>Population: %{x:.1f}%<extra></extra>'
            ))
        
        # Animation controls if multiple years
        if animate and len(animation_years) > 1:
            # Create animation frames
            frames = []
            for i, yr in enumerate(animation_years):
                frame_data = []
                year_data = pyramid_data[pyramid_data['year'] == yr].iloc[0]
                
                tfr = year_data['total_fertility_rate'] if 'total_fertility_rate' in year_data.index else 4.0
                pop_0_14 = year_data['population_0_14_percent'] if 'population_0_14_percent' in year_data.index else 40
                pop_65_plus = year_data['population_65_plus_percent'] if 'population_65_plus_percent' in year_data.index else 3
                
                population_by_age = self._generate_age_distribution(tfr, pop_0_14, pop_65_plus)
                male_pop = [-pop * 0.51 for pop in population_by_age]
                female_pop = [pop * 0.49 for pop in population_by_age]
                
                frames.append(go.Frame(
                    data=[
                        go.Bar(y=age_groups, x=male_pop, name='Male'),
                        go.Bar(y=age_groups, x=female_pop, name='Female')
                    ],
                    name=str(yr)
                ))
            
            fig.frames = frames
            
            # Add animation controls
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {'frame': {'duration': Config.VIZ_CONFIG['animation_duration']}}]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {'frame': {'duration': 0}}]
                        }
                    ],
                    'direction': 'left',
                    'showactive': False,
                    'x': 0.1,
                    'y': 0
                }],
                sliders=[{
                    'steps': [
                        {
                            'args': [[str(yr)], {'frame': {'duration': 0}}],
                            'label': str(yr),
                            'method': 'animate'
                        } for yr in animation_years
                    ],
                    'active': 0,
                    'y': 0,
                    'len': 0.9,
                    'x': 0.1
                }]
            )
        
        fig.update_layout(
            title=f"Population Pyramid - {country_name} ({animation_years[0] if not animate else f'{min(animation_years)}-{max(animation_years)}'})",
            xaxis_title='Population (%)',
            yaxis_title='Age Groups',
            barmode='relative',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_age_distribution(self, tfr: float, pop_0_14: float, pop_65_plus: float) -> list:
        """Generate realistic age distribution based on demographic indicators"""
        
        # Base population percentages by age group (17 groups)
        base_dist = [6, 5.5, 5, 4.5, 4.2, 4, 3.8, 3.6, 3.4, 3.2, 3, 2.8, 2.6, 2.4, 2.2, 2, 1.8]
        
        # Adjust based on fertility rate
        fertility_factor = max(0.5, min(2.0, tfr / 3.0))
        
        # Boost young age groups based on fertility
        for i in range(3):  # 0-14 age groups
            base_dist[i] *= fertility_factor
        
        # Adjust elderly groups
        elderly_factor = pop_65_plus / 10  # Normalize to expected range
        for i in range(13, 17):  # 65+ age groups
            base_dist[i] *= elderly_factor
        
        # Normalize to ensure sum is reasonable
        total = sum(base_dist)
        if total > 0:
            base_dist = [x / total * 100 for x in base_dist]
        
        return base_dist
    
    def create_clustering_visualization(self, clustered_data: pd.DataFrame):
        """Create ML clustering visualization"""
        
        if clustered_data.empty:
            st.error("No clustering data available")
            return None, None
        
        # Scatter plot for clusters
        available_indicators = ['total_fertility_rate', 'median_age', 'population_growth_rate', 'life_expectancy']
        valid_indicators = [ind for ind in available_indicators if ind in clustered_data.columns]
        
        if len(valid_indicators) < 2:
            st.error("Insufficient indicators for clustering visualization")
            return None, None
        
        try:
            # Main scatter plot
            x_indicator = valid_indicators[0]  # fertility rate
            y_indicator = valid_indicators[1]  # median age
            
            # Size by population growth if available
            size_col = None
            if 'population_growth_rate' in clustered_data.columns:
                size_data = clustered_data['population_growth_rate'].fillna(0)
                size_data = np.abs(size_data) + 1  # Ensure positive values
                size_col = size_data
            
            fig_scatter = px.scatter(
                clustered_data,
                x=x_indicator,
                y=y_indicator,
                color='cluster_label' if 'cluster_label' in clustered_data.columns else 'cluster',
                size=size_col,
                hover_name='country_name',
                title='Country Clustering by Demographic Profile',
                labels={
                    x_indicator: x_indicator.replace('_', ' ').title(),
                    y_indicator: y_indicator.replace('_', ' ').title()
                },
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig_scatter.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01
                )
            )
            
            # Cluster summary table
            cluster_summary = None
            if 'cluster_label' in clustered_data.columns:
                summary_indicators = [ind for ind in valid_indicators if ind in clustered_data.columns]
                
                cluster_stats = clustered_data.groupby('cluster_label').agg({
                    **{ind: ['mean', 'count'] for ind in summary_indicators[:3]},  # Limit to avoid wide table
                    'country_name': 'count'
                }).round(2)
                
                # Flatten column names
                cluster_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != 'count' or col[0] != 'country_name' 
                                       else 'Countries' for col in cluster_stats.columns]
                
                cluster_summary = cluster_stats
            
            return fig_scatter, cluster_summary
            
        except Exception as e:
            st.error(f"Error creating clustering visualization: {e}")
            return None, None
    
    def create_trend_comparison(self, df: pd.DataFrame, countries: list, indicators: list):
        """Create multi-country trend comparison"""
        
        if not countries or not indicators:
            st.warning("Please select countries and indicators for comparison")
            return
        
        # Filter data
        trend_data = df[df['country_name'].isin(countries)].copy()
        
        # Create subplots for each indicator
        fig = make_subplots(
            rows=len(indicators), 
            cols=1,
            subplot_titles=[ind.replace('_', ' ').title() for ind in indicators],
            vertical_spacing=0.08
        )
        
        colors = px.colors.qualitative.Set1[:len(countries)]
        
        for i, indicator in enumerate(indicators):
            if indicator not in trend_data.columns:
                continue
                
            for j, country in enumerate(countries):
                country_data = trend_data[trend_data['country_name'] == country]
                clean_data = country_data[['year', indicator]].dropna()
                
                if not clean_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=clean_data['year'],
                            y=clean_data[indicator],
                            mode='lines+markers',
                            name=country,
                            line=dict(color=colors[j]),
                            showlegend=(i == 0),  # Show legend only for first subplot
                        ),
                        row=i+1, col=1
                    )
        
        fig.update_layout(
            height=300 * len(indicators),
            title="Multi-Country Demographic Trends Comparison",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)