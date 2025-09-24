# ==================================================
# File: debug_tools.py
# ==================================================

import requests
import streamlit as st
from typing import Dict, List
from config import Config

class DebugTools:
    """Debugging utilities for API and data issues"""
    
    @staticmethod
    def test_basic_connectivity() -> Dict:
        """Test basic internet and API connectivity"""
        tests = []
        
        # Test 1: Basic internet
        try:
            response = requests.get("https://httpbin.org/get", timeout=10)
            tests.append({
                'test': 'Internet Connectivity',
                'status': 'PASS' if response.status_code == 200 else 'FAIL',
                'details': f"Status: {response.status_code}"
            })
        except Exception as e:
            tests.append({
                'test': 'Internet Connectivity',
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test 2: World Bank API access
        try:
            response = requests.get(f"{Config.WORLD_BANK_BASE_URL}/country?format=json&per_page=1", timeout=15)
            tests.append({
                'test': 'World Bank API Access',
                'status': 'PASS' if response.status_code == 200 else 'FAIL',
                'details': f"Status: {response.status_code}, Response type: {type(response.json())}"
            })
        except Exception as e:
            tests.append({
                'test': 'World Bank API Access',
                'status': 'FAIL',
                'details': str(e)
            })
        
        return {'tests': tests}
    
    @staticmethod
    def test_single_indicator(indicator_code: str, countries: List[str] = None) -> Dict:
        """Test fetching a single indicator"""
        if countries is None:
            countries = ['NGA', 'KEN', 'ZAF']  # Test countries
        
        country_codes = ';'.join(countries)
        url = f"{Config.WORLD_BANK_BASE_URL}/country/{country_codes}/indicator/{indicator_code}"
        params = {
            'format': 'json',
            'date': '2020:2023',
            'per_page': 100
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'status': 'SUCCESS',
                'url': response.url,
                'response_type': type(data).__name__,
                'response_length': len(data) if isinstance(data, list) else 'Not a list',
                'has_metadata': isinstance(data, list) and len(data) > 0,
                'has_data': isinstance(data, list) and len(data) > 1 and data[1] is not None,
                'data_count': len(data[1]) if isinstance(data, list) and len(data) > 1 and data[1] else 0,
                'sample_record': None
            }
            
            if result['has_data'] and data[1]:
                result['sample_record'] = data[1][0]
            
            return result
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'url': f"{url}?{requests.compat.urlencode(params)}"
            }
    
    @staticmethod
    def run_comprehensive_test() -> Dict:
        """Run comprehensive system test"""
        st.markdown("### 🔧 Comprehensive System Test")
        
        results = {
            'connectivity': DebugTools.test_basic_connectivity(),
            'indicators': {},
            'summary': {'passed': 0, 'failed': 0}
        }
        
        # Test core indicators
        for wb_code, indicator_name in Config.CORE_INDICATORS.items():
            st.write(f"Testing {indicator_name}...")
            test_result = DebugTools.test_single_indicator(wb_code)
            results['indicators'][indicator_name] = test_result
            
            if test_result.get('status') == 'SUCCESS' and test_result.get('data_count', 0) > 0:
                results['summary']['passed'] += 1
                st.success(f"✅ {indicator_name}: {test_result.get('data_count', 0)} records")
            else:
                results['summary']['failed'] += 1
                st.error(f"❌ {indicator_name}: Failed")
        
        return results