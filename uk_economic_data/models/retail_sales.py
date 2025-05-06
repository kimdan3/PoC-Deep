"""
Data models for retail sales data
"""

from typing import TypedDict, List

class RetailSalesData(TypedDict):
    """Represents a single retail sales data point"""
    date: str
    value: float

class RetailSalesResponse(TypedDict):
    """Represents the API response containing retail sales data"""
    data: List[RetailSalesData] 