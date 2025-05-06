"""
Retail sales analysis service
"""

import logging
from typing import List
from ..models.retail_sales import RetailSalesResponse

logger = logging.getLogger(__name__)

class RetailAnalysisService:
    """Service for analyzing retail sales data"""
    
    def __init__(self):
        self._cache = {}
    
    def _validate_date_format(self, date: str) -> bool:
        """Validate date format
        
        Args:
            date (str): Date string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if len(date) != 10:  # yyyy-mm-dd format
                return False
            year, month, day = map(int, date.split('-'))
            return 1 <= month <= 12 and 1 <= day <= 31
        except Exception:
            return False
    
    def analyze_retail_sales(self, data: RetailSalesResponse) -> str:
        """Analyze retail sales data and provide insights
        
        Args:
            data (RetailSalesResponse): Retail sales data to analyze
            
        Returns:
            str: Analysis results
        """
        try:
            if not data["data"]:
                return "No data available for analysis"
            
            # Calculate basic statistics
            values = [item["value"] for item in data["data"]]
            avg_value = sum(values) / len(values)
            max_value = max(values)
            min_value = min(values)
            
            # Calculate trend
            trend = "stable"
            if len(values) >= 2:
                first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
                second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                if second_half_avg > first_half_avg * 1.05:
                    trend = "increasing"
                elif second_half_avg < first_half_avg * 0.95:
                    trend = "decreasing"
            
            # Format analysis
            analysis = [
                f"Average sales: {avg_value:.2f}",
                f"Range: {min_value:.2f} to {max_value:.2f}",
                f"Trend: {trend}"
            ]
            
            return "; ".join(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing retail sales data: {str(e)}")
            return "Analysis failed"
    
    def process_batch(self, data_list: List[RetailSalesResponse]) -> List[str]:
        """Process multiple retail sales datasets
        
        Args:
            data_list (List[RetailSalesResponse]): List of retail sales datasets
            
        Returns:
            List[str]: List of analysis results
        """
        return [self.analyze_retail_sales(data) for data in data_list] 