import pytest
import pandas as pd
from datetime import datetime
from uk_economic_data.services.analysis_service import (
    AnalysisService,
    DataValidationError,
    APITimeoutError,
    UKDataFetchError,
    ServiceConfig
)

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'date': pd.date_range(start='2025-03-01', end='2025-03-10'),
        'product_name': ['Product A', 'Product B'] * 5,
        'sales_amount': [100, 200, 150, 250, 120, 220, 130, 230, 140, 240],
        'gender': ['M', 'F'] * 5,
        'age': [25, 35] * 5
    })

@pytest.fixture
def config():
    """Create test configuration"""
    return ServiceConfig(
        timeout_seconds=5,
        cache_size=100,
        batch_size=2,
        api_key="test_key",
        cache_ttl=3600
    )

@pytest.fixture
def analysis_service(config):
    """Create AnalysisService instance with test configuration"""
    return AnalysisService(config=config)

def test_validate_input_data(analysis_service, sample_data):
    """Test basic data validation"""
    assert analysis_service._validate_input_data(sample_data) is True

def test_validate_input_data_missing_columns(analysis_service):
    """Test validation with missing columns"""
    invalid_data = pd.DataFrame({'date': [datetime.now()]})
    with pytest.raises(DataValidationError):
        analysis_service._validate_input_data(invalid_data)

def test_validate_input_data_invalid_age(analysis_service):
    """Test validation with invalid age values"""
    invalid_data = pd.DataFrame({
        'date': [datetime.now()],
        'product_name': ['Test'],
        'sales_amount': [100],
        'gender': ['M'],
        'age': [-1]  # Invalid age
    })
    with pytest.raises(DataValidationError):
        analysis_service._validate_input_data(invalid_data)

def test_validate_input_data_invalid_gender(analysis_service):
    """Test validation with invalid gender values"""
    invalid_data = pd.DataFrame({
        'date': [datetime.now()],
        'product_name': ['Test'],
        'sales_amount': [100],
        'gender': ['X'],  # Invalid gender
        'age': [25]
    })
    with pytest.raises(DataValidationError):
        analysis_service._validate_input_data(invalid_data)

def test_validate_input_data_negative_sales(analysis_service):
    """Test validation with negative sales amounts"""
    invalid_data = pd.DataFrame({
        'date': [datetime.now()],
        'product_name': ['Test'],
        'sales_amount': [-100],  # Negative sales
        'gender': ['M'],
        'age': [25]
    })
    with pytest.raises(DataValidationError):
        analysis_service._validate_input_data(invalid_data)

def test_validate_parameters(analysis_service):
    """Test parameter validation"""
    # Valid parameters
    analysis_service._validate_parameters('M', 20, 30, 'weekly')
    
    # Invalid gender
    with pytest.raises(ValueError):
        analysis_service._validate_parameters('X', 20, 30, 'weekly')
    
    # Invalid period
    with pytest.raises(ValueError):
        analysis_service._validate_parameters('M', 20, 30, 'yearly')
    
    # Invalid age range
    with pytest.raises(ValueError):
        analysis_service._validate_parameters('M', 30, 20, 'weekly')

def test_filter_data(analysis_service, sample_data):
    """Test data filtering"""
    filtered_data = analysis_service._filter_data(sample_data, 'M', 20, 30)
    assert all(filtered_data['gender'] == 'M')
    assert all((filtered_data['age'] >= 20) & (filtered_data['age'] <= 30))

def test_get_top_products(analysis_service, sample_data):
    """Test getting top products"""
    products = analysis_service._get_top_products(sample_data, 2)
    assert len(products) == 2
    assert all(isinstance(p, str) for p in products)

def test_get_age_group(analysis_service):
    """Test age group calculation"""
    assert analysis_service._get_age_group(25) == "20-29"
    assert analysis_service._get_age_group(30) == "30-39"
    assert analysis_service._get_age_group(0) == "0-9"

@pytest.mark.asyncio
async def test_get_top_products_with_insights(analysis_service, sample_data):
    """Test basic product analysis functionality"""
    products, insights = await analysis_service.get_top_products_with_insights(
        df=sample_data,
        gender='M',
        min_age=20,
        max_age=30,
        top_n=2
    )
    assert len(products) == 2
    assert len(insights) == 2

@pytest.mark.asyncio
async def test_generate_marketing_insights(analysis_service):
    """Test marketing insights generation"""
    cause_analysis, strategies = await analysis_service.generate_marketing_insights(
        product="Test Product",
        date="2025-03-01",
        segment="M, Age group 20-29",
        change_rate=-10.0
    )
    assert isinstance(cause_analysis, str)
    assert isinstance(strategies, list)
    assert len(strategies) > 0

def test_clear_cache(analysis_service):
    """Test cache clearing"""
    analysis_service.clear_cache()
    assert len(analysis_service._insights_cache) == 0 