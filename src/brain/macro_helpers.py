"""
Macro Economic Data Integration Helpers (FAIL FAST MODE)

This module provides clean, isolated functions for integrating macro economic data
into the trading system without polluting the main analyzer files.

⚠️ FAIL FAST: Makro veri alınamazsa None döner, DUMMY SÜTUN YOK!
   Sinyal üretimi makro veriye bağlıysa DURDURULMALIDIR.
"""
import logging
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger("MACRO_HELPERS")


async def fetch_and_merge_macro(macro_connector, crypto_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Fetch macro data and merge with crypto DataFrame.
    
    FAIL FAST: Makro veri alınamazsa None döner, dummy sütun EKLENMEZ!
    
    Args:
        macro_connector: MacroConnector instance
        crypto_df: Crypto data DataFrame
    
    Returns:
        Merged DataFrame with macro columns, or None if fetch fails
    """
    try:
        # Try to fetch macro data
        macro_df = await macro_connector.fetch_macro_data(period="5d", interval="1h")
        
        if macro_df is None or macro_df.empty:
            logger.warning("❌ FAIL FAST: Makro veri alınamadı, veri YOK (dummy eklenmedi)")
            return None
        
        # Import here to avoid circular dependency
        from src.brain.feature_engineering import FeatureEngineer
        
        # FIX: Reset indices to avoid dtype mismatch (crypto uses int, macro uses datetime)
        crypto_df_copy = crypto_df.copy()
        macro_df_copy = macro_df.copy()
        
        # Ensure both have compatible indices for merge
        if not crypto_df_copy.index.equals(macro_df_copy.index):
            # Reset both to integer indices
            crypto_df_copy = crypto_df_copy.reset_index(drop=False)
            macro_df_copy = macro_df_copy.reset_index(drop=True)
        
        # Merge successfully
        try:
            merged_df = FeatureEngineer.merge_crypto_and_macro(crypto_df_copy, macro_df_copy)
        except Exception as merge_error:
            logger.error(f"❌ FAIL FAST: Merge hatası - {merge_error}")
            return None
        
        # Log success
        macro_score = macro_df['macro_score'].iloc[0] if 'macro_score' in macro_df.columns else 0
        logger.info(f"✅ Macro data integrated: Score={macro_score:.1f}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"❌ FAIL FAST: Macro integration failed: {e}")
        return None


# _add_dummy_macro_columns KALDIRILDI - FAIL FAST MODE
# Dummy sütunlar KULLANILMIYOR, sinyal üretimi gerçek veri olmadan DURDURULUR


async def fetch_macro_for_training(macro_connector, crypto_df: pd.DataFrame, 
                                   period: str = "1y", interval: str = "1h") -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Fetch macro data for training purposes (longer timeframes).
    
    FAIL FAST: Makro veri alınamazsa None döner.
    
    Args:
        macro_connector: MacroConnector instance
        crypto_df: Crypto data DataFrame
        period: Time period for data
        interval: Data interval
    
    Returns:
        Tuple of (merged_df, macro_df) or (None, empty_df) if fetch fails
    """
    try:
        macro_df = await macro_connector.fetch_macro_data(period=period, interval=interval)
        
        if macro_df is None or macro_df.empty:
            logger.warning("❌ FAIL FAST: Training için makro veri alınamadı")
            return None, pd.DataFrame()
        
        from src.brain.feature_engineering import FeatureEngineer
        merged_df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        return merged_df, macro_df
        
    except Exception as e:
        logger.error(f"❌ FAIL FAST: Training macro fetch failed: {e}")
        return None, pd.DataFrame()
