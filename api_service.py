import requests
import time
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
from config import Config
from cache_manager import CacheManager
from debug_tools import DebugTools

class WorldBankAPIService:
    """Production-ready World Bank API service"""
    
    def __init__(self):
        self.base_url = Config.WORLD_BANK_BASE_URL
        self.cache = CacheManager()
        self.debug = DebugTools()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Africa-Demographics-Platform/2.0'
        })
    
    def fetch_indicator_data(self, indicator_code: str, start_year: int = 1990, end_year: int = 2023) -> pd.DataFrame:
        """Fetch data for specific indicator"""
        
        cache_key = f"{indicator_code}_{start_year}_{end_year}"
        
        # Try cache first
        cached_data = self.cache.load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Build API request
        country_codes = ';'.join(Config.AFRICAN_COUNTRIES.keys())
        url = f"{self.base_url}/country/{country_codes}/indicator/{indicator_code}"
        params = {
            'format': 'json',
            'date': f"{start_year}:{end_year}",
            'per_page': 5000
        }
        
        try:
            response = self.session.get(url, params=params, timeout=Config.API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                return pd.DataFrame()
            
            # Process records efficiently
            parsed_records = []
            for record in data[1]:
                if not record or record.get('value') is None:
                    continue
                
                country_info = record.get('country', {})
                country_code = country_info.get('id', '')
                
                # Skip non-African countries
                if country_code not in Config.AFRICAN_COUNTRIES:
                    continue
                
                try:
                    year = int(record.get('date', 0))
                    if start_year <= year <= end_year:
                        parsed_records.append({
                            'country_iso2': country_code,
                            'country_name': country_info.get('value', ''),
                            'year': year,
                            'value': float(record.get('value')),
                            'indicator_code': indicator_code
                        })
                except (ValueError, TypeError):
                    continue
            
            if not parsed_records:
                return pd.DataFrame()
            
            df = pd.DataFrame(parsed_records)
            self.cache.save_to_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            st.error(f"API error for {indicator_code}: {str(e)}")
            return pd.DataFrame()
    
    def load_all_demographic_data(self, use_core_only: bool = False) -> pd.DataFrame:
        """Load demographic indicators"""
        
        indicators = Config.CORE_INDICATORS if use_core_only else Config.INDICATORS
        
        progress_bar = st.progress(0)
        all_data = []
        success_count = 0
        
        for i, (wb_code, indicator_name) in enumerate(indicators.items()):
            df = self.fetch_indicator_data(wb_code, 1990, 2023)
            
            if not df.empty:
                df['indicator_name'] = indicator_name
                all_data.append(df)
                success_count += 1
            
            progress_bar.progress((i + 1) / len(indicators))
            time.sleep(Config.REQUEST_DELAY)
        
        progress_bar.empty()
        
        if not all_data:
            st.error("No data loaded from API")
            return pd.DataFrame()
        
        # Combine and process data
        try:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Pivot to wide format
            pivot_df = combined_df.pivot_table(
                index=['country_iso2', 'country_name', 'year'],
                columns='indicator_name',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            pivot_df.columns.name = None
            
            # Calculate derived indicators
            pivot_df = self._add_derived_indicators(pivot_df)
            
            # Remove empty rows
            indicator_cols = [col for col in pivot_df.columns 
                            if col not in ['country_iso2', 'country_name', 'year']]
            pivot_df = pivot_df.dropna(subset=indicator_cols, how='all')
            
            return pivot_df
            
        except Exception as e:
            st.error(f"Data processing error: {e}")
            return pd.DataFrame()
    
    def _add_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated demographic indicators"""
        df = df.copy()
        
        # Calculate median age from age structure
        if all(col in df.columns for col in ['population_0_14_percent', 'population_65_plus_percent']):
            conditions = [
                (df['population_0_14_percent'] > 40),
                (df['population_0_14_percent'] > 30),
                (df['population_65_plus_percent'] > 10)
            ]
            values = [18, 22, 35]
            df['median_age'] = np.select(conditions, values, default=25)
        
        # Dependency ratios
        if all(col in df.columns for col in ['population_0_14_percent', 'population_15_64_percent']):
            mask = df['population_15_64_percent'] > 0
            df.loc[mask, 'child_dependency_ratio'] = (
                df.loc[mask, 'population_0_14_percent'] / df.loc[mask, 'population_15_64_percent'] * 100
            )
        
        if all(col in df.columns for col in ['population_65_plus_percent', 'population_15_64_percent']):
            mask = df['population_15_64_percent'] > 0
            df.loc[mask, 'old_age_dependency_ratio'] = (
                df.loc[mask, 'population_65_plus_percent'] / df.loc[mask, 'population_15_64_percent'] * 100
            )
        
        # Total dependency ratio
        if all(col in df.columns for col in ['child_dependency_ratio', 'old_age_dependency_ratio']):
            df['total_dependency_ratio'] = df['child_dependency_ratio'] + df['old_age_dependency_ratio']
        
        # Demographic dividend scoring
        df = self._calculate_demographic_dividend(df)
        
        return df
    
    def _calculate_demographic_dividend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate demographic dividend status using Config thresholds"""
        df = df.copy()
        
        # Initialize dividend score
        df['dividend_score'] = 0
        
        thresholds = Config.DIVIDEND_THRESHOLDS
        
        # Scoring based on thresholds
        if 'child_dependency_ratio' in df.columns:
            df['dividend_score'] += (df['child_dependency_ratio'] < thresholds['high_opportunity']['child_dependency']).astype(int) * 40
        
        if 'old_age_dependency_ratio' in df.columns:
            df['dividend_score'] += (df['old_age_dependency_ratio'] < thresholds['high_opportunity']['old_dependency']).astype(int) * 30
        
        if 'population_15_64_percent' in df.columns:
            df['dividend_score'] += (df['population_15_64_percent'] > thresholds['high_opportunity']['working_age']).astype(int) * 30
        
        # Dividend status classification
        def classify_dividend(score):
            if pd.isna(score):
                return 'Data Unavailable'
            elif score >= 80:
                return 'High Opportunity'
            elif score >= 50:
                return 'Opening Window'
            elif score >= 20:
                return 'Limited Window'
            else:
                return 'No Window'
        
        df['dividend_status'] = df['dividend_score'].apply(classify_dividend)
        df['dividend_window'] = df['dividend_score'] >= 50
        
        return df
    
    def get_population_data(self, year: int = 2023) -> Tuple[float, Dict, Dict]:
        """Calculate Africa population by summing countries"""
        
        pop_df = self.fetch_indicator_data('SP.POP.TOTL', year-5, year)
        
        if pop_df.empty:
            return 0.0, {}, {'error': 'No population data'}
        
        # Get latest data per country
        latest_pop = pop_df.groupby('country_iso2').apply(
            lambda x: x.loc[x['year'].idxmax()]
        ).reset_index(drop=True)
        
        total_population = 0
        country_populations = {}
        
        for _, row in latest_pop.iterrows():
            country_code = row['country_iso2']
            population = row['value']
            
            if population > 0:
                total_population += population
                country_populations[country_code] = {
                    'population': population,
                    'name': row['country_name'],
                    'year': row['year']
                }
        
        metadata = {
            'year_requested': year,
            'countries_with_data': len(country_populations),
            'calculation_method': 'world_bank_api_sum'
        }
        
        return total_population, country_populations, metadata
    
    def debug_raw_data(self, indicator_code: str = 'SP.DYN.TFRT.IN') -> None:
        """Debug raw API response"""
        st.markdown(f"### Debug Raw Data: {indicator_code}")
        
        df = self.fetch_indicator_data(indicator_code, 2020, 2023)
        
        if not df.empty:
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", df.columns.tolist())
            st.dataframe(df.head())
            st.write("Countries:", sorted(df['country_iso2'].unique()))
            st.write("Years:", sorted(df['year'].unique()))
        else:
            st.error("No data returned")
    
    def test_connection(self) -> Dict:
        """Test API connection"""
        return self.debug.test_basic_connectivity()