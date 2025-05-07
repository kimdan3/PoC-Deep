# UK Economic Data Collection and Analysis System

## Project Overview
This system analyses UK retail sales data by integrating various data sources to provide insights into sales patterns and external factors affecting retail performance.

## System Requirements

### Hardware Requirements
- CPU: 2+ cores
- RAM: 4GB minimum
- Storage: 1GB free space

### Software Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Operating System Support
- macOS
- Linux
- Windows 10/11

## Quick Start Guide

1. Create and activate a virtual environment:
```bash
# For macOS/Linux
python -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the project root:
```bash
# Required API Keys
WEATHER_API_KEY="your_weather_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"

# Optional API Keys
USE_PAID_APIS="false"
TWITTER_API_KEY=""
TWITTER_API_SECRET=""
TWITTER_ACCESS_TOKEN=""
TWITTER_ACCESS_TOKEN_SECRET=""

# DeepSeek Configuration
TIMEOUT_SECONDS=90
BATCH_SIZE=3
CACHE_SIZE=1000
CACHE_TTL=3600
```

4. Run the application:
```bash
python demo.py
```

The application will be available at `http://localhost:7861`.

## Key Features

### Data Collection
- **ONS API**: Official retail sales data
- **WeatherAPI**: Weather data and forecasts
- **Social Media**: Twitter data analysis (optional)

### Analysis Capabilities
- Retail sales trend analysis
- Weather impact prediction
- Seasonal factor analysis
- Social media sentiment analysis

## Architecture

### Data Sources
1. Official Statistics (ONS)
2. Free Public APIs (Weather)
3. Paid APIs (Optional: Twitter)

### Analysis Components
- Random Forest-based impact prediction model
- Weather impact analysis
- Sales trend analysis
- Sentiment analysis

## DeepSeek Configuration Guide

The system uses DeepSeek's API for generating marketing insights. To configure DeepSeek:

1. Obtain an API key from DeepSeek
2. Add the API key to your `.env` file:
   ```
   DEEPSEEK_API_KEY="your_api_key_here"
   ```

3. Configure additional settings in `.env`:
   ```
   TIMEOUT_SECONDS=90
   BATCH_SIZE=3
   CACHE_SIZE=1000
   CACHE_TTL=3600
   ```

These settings control:
- API request timeout
- Number of concurrent requests
- Cache size for responses
- Cache time-to-live in seconds

## Troubleshooting

### Common Issues
1. **API Connection Errors**
   - Verify API keys in `.env`
   - Check internet connection
   - Verify API service status

2. **Data Loading Issues**
   - Clear cache: `rm -rf data/cache/*`
   - Check file permissions
   - Verify data integrity

3. **Performance Issues**
   - Check system resources
   - Clear browser cache
   - Monitor background processes

## Development

### Setup
1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit a pull request

## License
MIT License 