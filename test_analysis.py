import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from uk_economic_data.services.analysis_service import AnalysisService
import asyncio

async def main():
    service = AnalysisService()
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = {
        'date': dates.repeat(5),
        'product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'] * len(dates),
        'sales_amount': np.random.uniform(100, 1000, len(dates) * 5),
        'gender': np.random.choice(['M', 'F'], len(dates) * 5),
        'age': np.random.randint(18, 65, len(dates) * 5)
    }
    df = pd.DataFrame(data)
    
    products, insights = await service.get_top_products_with_insights(
        df=df,
        gender='M',
        min_age=20,
        max_age=30
    )
    
    print('Top Products:', products)
    print('\nInsights:')
    for product, insight_list in insights.items():
        print(f'\n{product}:')
        print('\n'.join(insight_list))

if __name__ == '__main__':
    asyncio.run(main()) 