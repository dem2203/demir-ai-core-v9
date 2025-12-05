"""
Macro Economic Data Integration Helpers

This module provides clean, isolated functions for integrating macro economic data
into the trading system without polluting the main analyzer files.
"""
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger("MACRO_HELPERS")


async def fetch_and_merge_macro(macro_connector, crypto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch macro data and merge with crypto DataFrame.
    
    Args:
        macro_connector: MacroConnector instance
        crypto_df: Crypto data DataFrame
    
    Returns:
        Merged DataFrame with macro columns, or crypto_df with dummy columns if fetch fails
    """
    try:
        # Try to fetch macro data
        macro_df = await macro_connector.fetch_macro_data(period="5d", interval="1h")
        
        if macro_df is None or macro_df.empty:
            logger.warning("⚠️ Macro data unavailable. Using crypto-only data.")
            return _add_dummy_macro_columns(crypto_df)
        
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
            logger.warning(f"Merge failed even after index reset: {merge_error}")
            # Return crypto with dummy macro columns
            return _add_dummy_macro_columns(crypto_df)
        
        # Log success
        macro_score = macro_df['macro_score'].iloc[0] if 'macro_score' in macro_df.columns else 0
        logger.info(f"✅ Macro data integrated: Score={macro_score:.1f}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Macro integration failed: {e}")
        return _add_dummy_macro_columns(crypto_df)


def _add_dummy_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dummy macro columns to DataFrame for compatibility.
    
    Args:
        df: Original DataFrame
    
    Returns:
        DataFrame with dummy macro columns
    """
    df_copy = df.copy()
    
    # Add all expected macro columns with default values
    macro_columns = {
        'macro_DXY': 100.0,  # Default DXY
        'macro_VIX': 20.0,   # Default VIX
        'macro_SPX': 0.0,
        'macro_NDQ': 0.0,
        'macro_TNX': 0.0,
        'macro_GOLD': 0.0,
        'macro_SILVER': 0.0,
        'macro_OIL': 0.0
    }
    
    for col, default_val in macro_columns.items():
        df_copy[col] = default_val
    
    return df_copy


async def fetch_macro_for_training(macro_connector, crypto_df: pd.DataFrame, 
                                   period: str = "1y", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch macro data for training purposes (longer timeframes).
    
    Args:
        macro_connector: MacroConnector instance
        crypto_df: Crypto data DataFrame
        period: Time period for data
        interval: Data interval
    
    Returns:
        Merged DataFrame or crypto_df with empty macro if fetch fails
    """
    try:
        macro_df = await macro_connector.fetch_macro_data(period=period, interval=interval)
        
        if macro_df is None or macro_df.empty:
            logger.warning("Macro data unavailable for training")
            return crypto_df, pd.DataFrame()  # Empty macro_df for training
        
        from src.brain.feature_engineering import FeatureEngineer
        merged_df = FeatureEngineer.merge_crypto_and_macro(crypto_df, macro_df)
        
        return merged_df, macro_df
        
    except Exception as e:
        logger.error(f"Macro fetch failed: {e}")
        return crypto_df, pd.DataFrame()
