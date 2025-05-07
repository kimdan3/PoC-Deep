"""
UK Economic Data Collection and Analysis System
"""

import logging
from datetime import datetime, timedelta
import asyncio
from .api.ons_client import ONSClient
from .models.retail_sales import RetailSalesResponse
import httpx
import os
from typing import Optional, Dict
from functools import lru_cache
import json
from pathlib import Path

# Configure logging
def setup_logging():
    """Configure logging for the application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('uk_economic_data.log')
        ]
    )
    
    # Set specific log levels for external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class UKEconomicDataFetcher:
    """Main class for fetching and analyzing UK economic data"""
    
    def __init__(self):
        self.base_url = "https://api.ons.gov.uk/dataset/retail-sales-index"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "UK-Economic-Data-Analysis/1.0"
        }
        self.client = httpx.AsyncClient(timeout=30.0)
        self.ons_client = ONSClient()
        self._cache: Dict[str, dict] = {}
        self._cache_file = Path("uk_economic_data_cache.json")
        self._load_cache()
        logger.info("UKEconomicDataFetcher initialized")

    def _load_cache(self) -> None:
        """Load data from cache file."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Cache loaded from {self._cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache data to file."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
            logger.info(f"Cache saved to {self._cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")

    def _get_default_data(self, date: str) -> dict:
        """Generate default data for the given date."""
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            # Return lower values for weekends and bank holidays
            is_weekend = dt.weekday() >= 5
            seasonal_context = self._get_seasonal_context(dt)
            
            base_value = 80.0 if is_weekend else 100.0
            if "Easter period" in seasonal_context:
                base_value *= 1.2
            elif "School holidays" in seasonal_context:
                base_value *= 1.1
            
            return {
                "data": [{
                    "date": date,
                    "value": round(base_value, 2),
                    "status": "estimated",
                    "is_weekend": is_weekend,
                    "seasonal_context": seasonal_context
                }]
            }
        except Exception as e:
            logger.error(f"Error generating default data: {str(e)}")
            return {
                "data": [{
                    "date": date,
                    "value": 100.0,
                    "status": "estimated",
                    "error": str(e)
                }]
            }

    async def fetch_retail_sales(self, date: str) -> dict:
        """Fetch retail sales data from ONS API with caching."""
        # Check cache
        cache_key = f"retail_sales_{date}"
        if cache_key in self._cache:
            logger.info(f"Using cached data for {date}")
            return self._cache[cache_key]

        try:
            url = f"{self.base_url}/timeseries/J5EA/data"
            params = {
                "start_date": date,
                "end_date": date
            }
            
            # Rate limiting
            await asyncio.sleep(0.1)  # Add delay between API calls
            
            response = await self.client.get(url, headers=self.headers, params=params)
            if response.status_code == 429:  # Rate limit
                retry_after = int(response.headers.get('Retry-After', '5'))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                await asyncio.sleep(retry_after)
                return await self.fetch_retail_sales(date)
            
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            self._cache[cache_key] = data
            self._save_cache()
            
            logger.info(f"Successfully fetched retail sales data for {date}")
            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Retail sales data not found for {date}. Using default data.")
                default_data = self._get_default_data(date)
                self._cache[cache_key] = default_data
                self._save_cache()
                return default_data
            elif e.response.status_code == 401:
                logger.error("API authentication failed")
                raise ValueError("API authentication failed")
            elif e.response.status_code >= 500:
                logger.error(f"ONS API server error: {str(e)}")
                raise ConnectionError("ONS API server error")
            logger.error(f"Error fetching retail sales data: {str(e)}")
            return {"error": str(e)}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching data for {date}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Error fetching retail sales data: {str(e)}")
            return {"error": str(e)}

    async def get_background_info(self, date: str) -> str:
        """Get comprehensive background information for the given date."""
        try:
            retail_sales = await self.fetch_retail_sales(date)
            
            background_info = []
            
            if isinstance(retail_sales, dict) and "error" not in retail_sales:
                background_info.append("Retail Sales Data:")
                background_info.append(str(retail_sales))

            result = "\n".join(background_info) if background_info else "No background information available."
            logger.info(f"Successfully gathered background info for {date}")
            return result
        except Exception as e:
            logger.error(f"Error getting background info: {str(e)}")
            return "Unable to fetch background information."

    async def close(self):
        """Close the HTTP client and save cache."""
        try:
            await self.client.aclose()
            self._save_cache()
            logger.info("HTTP client closed and cache saved successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _get_seasonal_context(self, date: datetime) -> str:
        """Get seasonal context for the given date with improved event detection."""
        try:
            events = []
            
            # Easter calculation (simplified)
            if 3 <= date.month <= 4:
                events.append("Easter period")
            
            # School holidays (UK school holiday periods)
            uk_school_holidays = {
                (7, 20): (9, 1),  # Summer holidays
                (12, 20): (1, 4),  # Christmas holidays
                (4, 1): (4, 15),   # Easter holidays
                (10, 25): (11, 2)  # Autumn half-term
            }
            
            for (start_month, start_day), (end_month, end_day) in uk_school_holidays.items():
                start = datetime(date.year, start_month, start_day)
                end = datetime(date.year, end_month, end_day)
                if start <= date <= end:
                    events.append("School holidays")
                    break
            
            # Bank holidays (UK public holidays)
            bank_holidays = [
                (1, 1),    # New Year's Day
                (5, 1),    # Early May Bank Holiday
                (5, 29),   # Spring Bank Holiday
                (8, 28),   # Summer Bank Holiday
                (12, 25),  # Christmas Day
                (12, 26)   # Boxing Day
            ]
            
            if (date.month, date.day) in bank_holidays:
                events.append("Bank holiday")
            
            # Seasonality
            seasons = {
                (12, 1): "Winter shopping season",
                (3, 1): "Spring shopping season",
                (6, 1): "Summer shopping season",
                (9, 1): "Autumn shopping season"
            }
            
            for (month, day), season in seasons.items():
                if date.month == month and date.day >= day:
                    events.append(season)
                    break
            
            result = ", ".join(events) if events else "No major seasonal events"
            logger.info(f"Seasonal context for {date}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting seasonal context: {str(e)}")
            return "Seasonal context unavailable" 