import pandas as pd
import os
import time
from datetime import datetime
import logging
from typing import Optional, List, Dict
import pyarrow as pa
import pyarrow.parquet as pq

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
    
    # Data type specifications
    DTYPE_SPEC = {
        'date': 'datetime64[ns]',
        'customer_id': 'str',
        'store_id': 'str',
        'age': 'int32',
        'gender': 'str',
        'age_group': 'str',
        'product_name': 'str',
        'category': 'str',
        'units_sold': 'int32',
        'unit_price': 'float32',
        'discount_applied': 'float32',
        'discounted': 'bool',
        'sales_amount': 'float32'
    }
    
    @staticmethod
    def _should_convert_to_parquet(path: str) -> bool:
        """
        Check if file should be converted to parquet format.
        
        Args:
            path: Path to the data file
            
        Returns:
            bool: True if file should be converted
        """
        try:
            # Check file size
            file_size = os.path.getsize(path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                return True
                
            # Check loading time
            start_time = time.time()
            pd.read_csv(path, nrows=1000)  # Sample read
            load_time = time.time() - start_time
            if load_time > 1.0:  # 1 second
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking file conversion: {str(e)}")
            return False
            
    @staticmethod
    def _convert_to_parquet(csv_path: str) -> str:
        """
        Convert CSV file to Parquet format.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            str: Path to parquet file
        """
        try:
            parquet_path = csv_path.replace('.csv', '.parquet')
            
            # Read CSV with optimized settings
            df = pd.read_csv(
                csv_path,
                dtype=DataLoader.DTYPE_SPEC,
                parse_dates=['date'],
                na_values=['NA', 'NULL', 'null', 'NaN', 'nan', ''],
                encoding='utf-8'
            )
            
            # Convert to parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path)
            
            logger.info(f"Successfully converted {csv_path} to {parquet_path}")
            return parquet_path
            
        except Exception as e:
            logger.error(f"Error converting to parquet: {str(e)}")
            raise DataLoaderError(f"Error converting to parquet: {str(e)}")
    
    @staticmethod
    def load_data(path: str = "data/data.csv") -> Optional[pd.DataFrame]:
        """
        Read data file and optimize for performance.
        
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
            
            # Check if should convert to parquet
            if path.endswith('.csv') and DataLoader._should_convert_to_parquet(path):
                path = DataLoader._convert_to_parquet(path)
            
            # Load data based on file type
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(
                    path,
                    dtype={k: v for k, v in DataLoader.DTYPE_SPEC.items() if k != 'date'},
                    parse_dates=['date'],
                    na_values=['NA', 'NULL', 'null', 'NaN', 'nan', ''],
                    encoding='utf-8'
                )
            
            if df.empty:
                raise DataLoaderError(f"Data file is empty: {path}")
            
            # Create a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Keep only required columns
            df = df[DataLoader.REQUIRED_COLUMNS]
            
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