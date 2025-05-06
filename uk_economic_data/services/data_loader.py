import pandas as pd
from datetime import datetime
import logging
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

class DataLoaderError(Exception):
    """Base exception for DataLoader."""
    pass

class DataLoader:
    """Service class for loading and preprocessing retail sales data."""
    
    # Constants
    REQUIRED_COLUMNS = ['date', 'product_name', 'age', 'gender', 'sales_amount']
    VALID_GENDERS = {'M', 'F'}
    MIN_AGE = 0
    MAX_AGE = 100
    
    @staticmethod
    def load_data(path: str = "data/data.csv") -> Optional[pd.DataFrame]:
        """
        Read CSV and convert 'date' column to datetime.
        
        Args:
            path: Path to the data file
            
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails
            
        Raises:
            DataLoaderError: If data loading fails
        """
        try:
            if not os.path.exists(path):
                raise DataLoaderError(f"Data file not found: {path}")
                
            df = pd.read_csv(path)
            
            if df.empty:
                raise DataLoaderError(f"Data file is empty: {path}")
                
            logger.info(f"Successfully loaded data from {path}")
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"Data file is empty: {path}")
            raise DataLoaderError(f"Data file is empty: {path}")
            
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            raise DataLoaderError(f"Error loading data: {str(e)}")

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess the input DataFrame.
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            Optional[pd.DataFrame]: Preprocessed DataFrame or None if preprocessing fails
            
        Raises:
            DataLoaderError: If preprocessing fails
        """
        try:
            if df is None or df.empty:
                raise DataLoaderError("Input DataFrame is None or empty")

            # Check required columns
            missing_columns = [col for col in DataLoader.REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                raise DataLoaderError(f"Missing required columns: {missing_columns}")

            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Remove rows with missing values
            missing_before = df.isnull().sum().sum()
            df = df.dropna()
            missing_after = df.isnull().sum().sum()
            if missing_before > 0:
                logger.warning(f"Removed {missing_before - missing_after} rows with missing values")
            
            # Clean negative sales
            negative_sales = (df['sales_amount'] < 0).sum()
            if negative_sales > 0:
                logger.warning(f"Found {negative_sales} rows with negative sales amounts")
                df.loc[df['sales_amount'] < 0, 'sales_amount'] = 0
            
            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                raise DataLoaderError("Failed to convert date column to datetime")
                
            if not pd.api.types.is_numeric_dtype(df['sales_amount']):
                raise DataLoaderError("Sales amount column is not numeric")
                
            if not pd.api.types.is_numeric_dtype(df['age']):
                raise DataLoaderError("Age column is not numeric")
                
            # Validate gender values
            invalid_genders = set(df['gender'].unique()) - DataLoader.VALID_GENDERS
            if invalid_genders:
                raise DataLoaderError(f"Invalid gender values found: {invalid_genders}")
                
            # Validate age range
            if (df['age'] < DataLoader.MIN_AGE).any() or (df['age'] > DataLoader.MAX_AGE).any():
                raise DataLoaderError(f"Age values must be between {DataLoader.MIN_AGE} and {DataLoader.MAX_AGE}")
            
            logger.info("Data preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise DataLoaderError(f"Data preprocessing failed: {str(e)}")

    @staticmethod
    def get_product_categories(df: pd.DataFrame) -> List[str]:
        """
        Get unique product categories from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List[str]: List of unique product categories
            
        Raises:
            DataLoaderError: If product categories cannot be retrieved
        """
        try:
            if df is None or df.empty:
                raise DataLoaderError("Input DataFrame is None or empty")
                
            if 'product_name' not in df.columns:
                raise DataLoaderError("Product name column not found in DataFrame")
                
            categories = sorted(df["product_name"].unique().tolist())
            logger.info(f"Found {len(categories)} unique product categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error getting product categories: {str(e)}")
            raise DataLoaderError(f"Failed to get product categories: {str(e)}")

    @staticmethod
    def get_customer_segments(df: pd.DataFrame) -> List[str]:
        """
        Get unique customer segments from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List[str]: List of unique customer segments
            
        Raises:
            DataLoaderError: If customer segments cannot be retrieved
        """
        try:
            if df is None or df.empty:
                raise DataLoaderError("Input DataFrame is None or empty")
                
            if 'age' not in df.columns or 'gender' not in df.columns:
                raise DataLoaderError("Age or gender columns not found in DataFrame")
                
            # Create age groups
            df['age_group'] = (df['age'] // 10) * 10
            segments = sorted(df.apply(lambda x: f"{x['age_group']}s_{x['gender']}", axis=1).unique().tolist())
            
            logger.info(f"Found {len(segments)} unique customer segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error getting customer segments: {str(e)}")
            raise DataLoaderError(f"Failed to get customer segments: {str(e)}") 