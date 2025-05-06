"""
UK Economic Data Collection and Analysis System
"""

import logging
from datetime import datetime, timedelta
import asyncio
from .api.ons_client import ONSClient
from .services.retail_analysis import RetailAnalysisService
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
        self.retail_analysis = RetailAnalysisService()
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
            
            # 캐시 업데이트
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

    async def fetch_weather_data(self, date: str) -> dict:
        """Fetch weather data for the given date with caching."""
        cache_key = f"weather_{date}"
        if cache_key in self._cache:
            logger.info(f"Using cached weather data for {date}")
            return self._cache[cache_key]

        try:
            api_key = os.getenv("WEATHER_API_KEY")
            if not api_key:
                raise ValueError("Weather API key not configured")
            
            # 실제 날씨 API 호출 로직 구현
            # 임시 데이터
            data = {"temperature": 15, "condition": "sunny"}
            
            self._cache[cache_key] = data
            self._save_cache()
            
            return data
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return {"error": str(e)}

    async def fetch_transport_issues(self) -> dict:
        """Fetch current transport issues from TfL API with caching."""
        cache_key = "transport_current"
        cache_timeout = 300  # 5분

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if datetime.now().timestamp() - cached_data.get("timestamp", 0) < cache_timeout:
                logger.info("Using cached transport data")
                return cached_data["data"]

        try:
            url = "https://api.tfl.gov.uk/Line/mode/tube/status"
            response = await self.client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            self._cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now().timestamp()
            }
            self._save_cache()
            
            return data
        except Exception as e:
            logger.error(f"Error fetching transport issues: {str(e)}")
            return {"error": str(e)}

    async def get_background_info(self, date: str) -> str:
        """Get comprehensive background information for the given date."""
        try:
            tasks = [
                self.fetch_retail_sales(date),
                self.fetch_weather_data(date),
                self.fetch_transport_issues()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            retail_sales, weather_data, transport_issues = results
            
            background_info = []
            
            if isinstance(retail_sales, dict) and "error" not in retail_sales:
                background_info.append("Retail Sales Data:")
                background_info.append(str(retail_sales))
            
            if isinstance(weather_data, dict) and "error" not in weather_data:
                background_info.append("\nWeather Conditions:")
                background_info.append(str(weather_data))
            
            if isinstance(transport_issues, dict) and "error" not in transport_issues:
                background_info.append("\nTransport Status:")
                background_info.append(str(transport_issues))

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
            
            # School holidays (영국 학교 방학 기간)
            uk_school_holidays = {
                (7, 20): (9, 1),  # 여름 방학
                (12, 20): (1, 4),  # 크리스마스 방학
                (4, 1): (4, 15),   # 부활절 방학
                (10, 25): (11, 2)  # 가을 방학
            }
            
            for (start_month, start_day), (end_month, end_day) in uk_school_holidays.items():
                start = datetime(date.year, start_month, start_day)
                end = datetime(date.year, end_month, end_day)
                if start <= date <= end:
                    events.append("School holidays")
                    break
            
            # Bank holidays (영국 공휴일)
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
            
            # 계절성
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