import unittest
from datetime import datetime
import pandas as pd
import numpy as np
from demo import (
    get_top_changes,
    get_top_products_with_insights,
    analyze_changes
)
from config import DATE_CONFIG, UI_CONFIG

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        """Test data setup"""
        self.test_data = pd.DataFrame({
            'date': pd.date_range(start='2025-03-01', end='2025-03-10'),
            'product_name': ['Product A'] * 10 + ['Product B'] * 10,
            'age_group': ['20s_M'] * 20,
            'sales_amount': np.random.randint(100, 1000, 20)
        })
        
    def test_get_top_changes(self):
        """Test for get_top_changes function"""
        start_date = datetime(2025, 3, 1)
        end_date = datetime(2025, 3, 5)
        
        result = get_top_changes(
            self.test_data,
            start_date,
            end_date,
            product="Product A",
            customer_segment="20s_M"
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('change_percentage', result.columns)
        
    def test_analyze_changes_date_validation(self):
        """Test for date validation"""
        # Invalid date range
        with self.assertRaises(ValueError):
            analyze_changes(
                "2025-02-01",  # Before allowed range
                "2025-03-01",
                5,
                18,
                35,
                "M",
                5,
                "weekly"
            )
            
    def test_analyze_changes_age_validation(self):
        """Test for age range validation"""
        # Invalid age range
        with self.assertRaises(ValueError):
            analyze_changes(
                "2025-03-01",
                "2025-03-10",
                5,
                35,  # min_age greater than max_age
                18,
                "M",
                5,
                "weekly"
            )

if __name__ == '__main__':
    unittest.main() 