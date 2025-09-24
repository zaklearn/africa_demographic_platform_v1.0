import pickle
import time
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import streamlit as st
from config import Config

class CacheManager:
    """Handles local data caching for API responses"""
    
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.cache_hours = Config.CACHE_HOURS
        
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        
        cache_time = cache_path.stat().st_mtime
        current_time = time.time()
        age_hours = (current_time - cache_time) / 3600
        
        return age_hours < self.cache_hours
    
    def load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception:
                return None
        
        return None
    
    def save_to_cache(self, cache_key: str, data: pd.DataFrame):
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass
    
    def clear_cache(self):
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1
            st.success(f"Cleared {count} cache files")
        except Exception as e:
            st.error(f"Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict:
        cache_info = {
            'total_files': 0,
            'total_size_mb': 0,
            'files': []
        }
        
        if not self.cache_dir.exists():
            return cache_info
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_info['total_files'] = len(cache_files)
        
        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files)
            cache_info['total_size_mb'] = total_size / (1024 * 1024)
            
            for cache_file in cache_files:
                stat = cache_file.stat()
                age_hours = (time.time() - stat.st_mtime) / 3600
                cache_info['files'].append({
                    'name': cache_file.stem,
                    'size_kb': stat.st_size / 1024,
                    'age_hours': age_hours,
                    'valid': age_hours < self.cache_hours
                })
        
        return cache_info