"""
ONS API Client for fetching economic data
"""

import logging
import httpx
from typing import Optional
from ..models.retail_sales import RetailSalesResponse, RetailSalesData

logger = logging.getLogger(__name__)

class ONSClient:
    """Client for interacting with the ONS API"""
    
    BASE_URL = "https://api.beta.ons.gov.uk/v1"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def _make_api_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make an API request with retry logic
        
        Args:
            endpoint (str): API endpoint
            params (Optional[dict]): Query parameters
            
        Returns:
            dict: API response
            
        Raises:
            Exception: If the request fails after retries
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await self.client.get(f"{self.BASE_URL}{endpoint}", params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue
                logger.error(f"API request failed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during API request: {str(e)}")
                raise
    
    async def fetch_retail_sales(self, start_date: str, end_date: str) -> RetailSalesResponse:
        """Fetch retail sales data from the ONS API
        
        Args:
            start_date (str): Start date in 'yyyy-mm-dd' format
            end_date (str): End date in 'yyyy-mm-dd' format
            
        Returns:
            RetailSalesResponse: Formatted retail sales data
        """
        try:
            endpoint = "/timeseries/J5EA/dataset/retail-sales-index/data"
            params = {
                "start_date": start_date,
                "end_date": end_date
            }
            
            response = await self._make_api_request(endpoint, params)
            
            # Format the response
            formatted_data = []
            for item in response.get("data", []):
                try:
                    formatted_data.append(RetailSalesData(
                        date=item["date"],
                        value=float(item["value"])
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid data point: {str(e)}")
                    continue
            
            return RetailSalesResponse(data=formatted_data)
            
        except Exception as e:
            logger.error(f"Error fetching retail sales data: {str(e)}")
            return RetailSalesResponse(data=[]) 