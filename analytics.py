import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from api_service import WorldBankAPIService
from config import Config

class DemographicAnalytics:
    """Complete analytics for demographic data"""
    
    def __init__(self):
        self.api_service = WorldBankAPIService()
    
    def calculate_real_africa_population(self, year: int = 2023) -> Tuple[float, Dict, Dict]:
        """Calculate Africa population by summing country populations"""
        return self.api_service.get_population_data(year)
    
    def calculate_continental_metrics(self, df: pd.DataFrame, year: int = 2023) -> Dict:
        """Calculate population-weighted continental metrics"""
        
        # Get real population data
        total_pop, country_pops, metadata = self.calculate_real_africa_population(year)
        
        if total_pop == 0 or not country_pops:
            return {
                'error': 'No population data available',
                'total_population_millions': 0,
                'metadata': metadata
            }
        
        # Get demographic data for the year
        year_data = df[df['year'] == year].copy()
        if year_data.empty:
            latest_year = df['year'].max()
            year_data = df[df['year'] == latest_year].copy()
        
        if year_data.empty:
            return {
                'error': 'No demographic data available',
                'total_population_millions': total_pop / 1e6,
                'metadata': metadata
            }
        
        # Add population weights
        year_data['real_population'] = year_data['country_iso2'].map(
            lambda x: country_pops.get(x, {}).get('population', 0)
        )
        
        weighted_data = year_data[year_data['real_population'] > 0].copy()
        
        if weighted_data.empty:
            return {
                'error': 'No matching data',
                'total_population_millions': total_pop / 1e6,
                'metadata': metadata
            }
        
        # Calculate weighted metrics
        weighted_metrics = {}
        
        for indicator in ['total_fertility_rate', 'median_age', 'population_growth_rate', 'life_expectancy']:
            if indicator in weighted_data.columns:
                indicator_data = weighted_data.dropna(subset=[indicator])
                
                if not indicator_data.empty and len(indicator_data) >= 5:
                    weighted_sum = (indicator_data[indicator] * indicator_data['real_population']).sum()
                    total_weight = indicator_data['real_population'].sum()
                    
                    if total_weight > 0:
                        weighted_metrics[indicator] = weighted_sum / total_weight
                    else:
                        weighted_metrics[indicator] = np.nan
                else:
                    weighted_metrics[indicator] = np.nan
            else:
                weighted_metrics[indicator] = np.nan
        
        # Dividend distribution
        dividend_counts = {}
        if 'dividend_status' in weighted_data.columns:
            dividend_counts = weighted_data['dividend_status'].value_counts().to_dict()
        
        return {
            'total_population_millions': total_pop / 1e6,
            'weighted_tfr': weighted_metrics.get('total_fertility_rate', np.nan),
            'weighted_median_age': weighted_metrics.get('median_age', np.nan),
            'weighted_growth_rate': weighted_metrics.get('population_growth_rate', np.nan),
            'weighted_life_expectancy': weighted_metrics.get('life_expectancy', np.nan),
            'dividend_distribution': dividend_counts,
            'countries_analyzed': len(weighted_data),
            'metadata': {
                **metadata,
                'calculation_year': year_data['year'].iloc[0] if not year_data.empty else year,
                'countries_with_demographic_data': len(weighted_data)
            }
        }
    
    def get_country_clusters(self, df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
        """Advanced ML clustering with demographic transition stages"""
        
        year_data = df[df['year'] == year].copy()
        if year_data.empty:
            year_data = df[df['year'] == df['year'].max()].copy()
        
        if year_data.empty:
            return pd.DataFrame()
        
        # Use configured clustering indicators
        clustering_indicators = []
        config_indicators = Config.CLUSTERING_CONFIG['indicators']
        
        for indicator in config_indicators:
            if indicator in year_data.columns and year_data[indicator].notna().sum() >= 10:
                clustering_indicators.append(indicator)
        
        if len(clustering_indicators) < 2:
            return pd.DataFrame()
        
        # Prepare clustering data
        cluster_data = year_data[clustering_indicators].fillna(year_data[clustering_indicators].mean())
        valid_rows = cluster_data.dropna().index
        
        if len(valid_rows) < 10:
            return pd.DataFrame()
        
        # Perform clustering with config parameters
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data.loc[valid_rows])
        
        n_clusters = min(Config.CLUSTERING_CONFIG['n_clusters'], len(valid_rows) // 3)
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=Config.CLUSTERING_CONFIG['random_state'], 
            n_init=10
        )
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels
        clustered_data = year_data.loc[valid_rows].copy()
        clustered_data['cluster'] = clusters
        
        # Generate meaningful cluster labels based on fertility transition
        if 'total_fertility_rate' in clustered_data.columns:
            cluster_means = clustered_data.groupby('cluster')['total_fertility_rate'].mean()
            sorted_clusters = cluster_means.sort_values(ascending=False)
            
            label_map = {}
            cluster_labels = Config.CLUSTERING_CONFIG['cluster_labels']
            
            for i, cluster_id in enumerate(sorted_clusters.index):
                if i < len(cluster_labels):
                    label_map[cluster_id] = cluster_labels[i]
                else:
                    label_map[cluster_id] = f'Cluster {cluster_id}'
            
            clustered_data['cluster_label'] = clustered_data['cluster'].map(label_map)
            
            # Add cluster characteristics
            clustered_data = self._add_cluster_characteristics(clustered_data, clustering_indicators)
        
        return clustered_data
    
    def _add_cluster_characteristics(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Add cluster characteristic descriptions"""
        
        cluster_profiles = {}
        for cluster_label in df['cluster_label'].unique():
            cluster_data = df[df['cluster_label'] == cluster_label]
            
            profile = {}
            for indicator in indicators:
                if indicator in cluster_data.columns:
                    profile[indicator] = {
                        'mean': cluster_data[indicator].mean(),
                        'median': cluster_data[indicator].median(),
                        'std': cluster_data[indicator].std()
                    }
            
            cluster_profiles[cluster_label] = profile
        
        df['cluster_profiles'] = df['cluster_label'].map(cluster_profiles)
        
        return df
    
    def analyze_demographic_dividend_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze demographic dividend evolution over time"""
        
        if 'dividend_status' not in df.columns:
            return {}
        
        # Evolution over time
        dividend_evolution = df.groupby(['year', 'dividend_status']).size().unstack(fill_value=0)
        
        # Countries by status for latest year
        latest_year = df['year'].max()
        latest_dividend = df[df['year'] == latest_year].groupby('dividend_status')['country_name'].apply(list)
        
        # Transition analysis - countries moving between categories
        transitions = {}
        for country in df['country_name'].unique():
            country_data = df[df['country_name'] == country].sort_values('year')
            if not country_data.empty and len(country_data) > 1:
                first_status = country_data['dividend_status'].iloc[0]
                last_status = country_data['dividend_status'].iloc[-1]
                if first_status != last_status:
                    transitions[country] = {
                        'from': first_status,
                        'to': last_status,
                        'years': f"{country_data['year'].min()}-{country_data['year'].max()}"
                    }
        
        return {
            'evolution': dividend_evolution,
            'latest_distribution': latest_dividend.to_dict(),
            'transitions': transitions,
            'total_countries': df['country_name'].nunique()
        }
    
    def generate_country_comparison(self, df: pd.DataFrame, countries: List[str], indicators: List[str]) -> Dict:
        """Generate detailed comparison between selected countries"""
        
        comparison_data = df[df['country_name'].isin(countries)].copy()
        
        if comparison_data.empty:
            return {}
        
        # Latest values comparison
        latest_year = comparison_data['year'].max()
        latest_comparison = comparison_data[comparison_data['year'] == latest_year]
        
        comparison_metrics = {}
        for indicator in indicators:
            if indicator in latest_comparison.columns:
                country_values = latest_comparison.set_index('country_name')[indicator].to_dict()
                comparison_metrics[indicator] = {
                    'values': country_values,
                    'ranking': sorted(country_values.items(), key=lambda x: x[1] if pd.notna(x[1]) else 0, reverse=True),
                    'range': {
                        'min': min(v for v in country_values.values() if pd.notna(v)),
                        'max': max(v for v in country_values.values() if pd.notna(v))
                    } if any(pd.notna(v) for v in country_values.values()) else {}
                }
        
        # Trend correlation analysis
        correlations = {}
        for i, indicator1 in enumerate(indicators):
            for indicator2 in indicators[i+1:]:
                if both_cols_exist := (indicator1 in comparison_data.columns and indicator2 in comparison_data.columns):
                    corr_data = comparison_data[[indicator1, indicator2]].dropna()
                    if len(corr_data) > 3:
                        correlation = corr_data[indicator1].corr(corr_data[indicator2])
                        correlations[f"{indicator1}_vs_{indicator2}"] = correlation
        
        return {
            'latest_comparison': comparison_metrics,
            'correlations': correlations,
            'data_coverage': {
                'countries': len(countries),
                'years_available': sorted(comparison_data['year'].unique()),
                'indicators_available': [ind for ind in indicators if ind in comparison_data.columns]
            }
        }