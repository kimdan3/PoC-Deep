from datetime import datetime
from typing import Dict, Any

# Date-related settings
DATE_CONFIG = {
    "MIN_DATE": datetime(2025, 3, 1),
    "MAX_DATE": datetime(2025, 5, 10),
    "DATE_FORMAT": "%Y-%m-%d"
}

# UI-related settings
UI_CONFIG = {
    "TOP_N": {
        "MIN": 1,
        "MAX": 20,
        "DEFAULT": 5,
        "CHOICES": ["5", "10", "Custom"]
    },
    "AGE": {
        "MIN": 0,
        "MAX": 100,
        "DEFAULT_MIN": 18,
        "DEFAULT_MAX": 35
    },
    "THRESHOLD": {
        "MIN": 1,
        "MAX": 50,
        "DEFAULT": 5
    }
}

# Analysis-related settings
ANALYSIS_CONFIG = {
    "PERIOD_CHOICES": ["daily", "weekly", "monthly"],
    "DEFAULT_PERIOD": "weekly",
    "GENDER_CHOICES": ["M", "F"],
    "DEFAULT_GENDER": "M"
}

# Logging-related settings
LOGGING_CONFIG = {
    "LOG_FILE": "app.log",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_LEVEL": "INFO"
} 