# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:09:32 2025

@author: amrit
"""

# financial_data.py

"""
Currency and regional cost data for solar PV systems
"""

# Currency configurations
CURRENCIES = {
    "INR": {
        "symbol": "₹",
        "name": "Indian Rupee",
        "country": "India"
    },
    "USD": {
        "symbol": "$",
        "name": "US Dollar",
        "country": "USA"
    },
    "CAD": {
        "symbol": "C$",
        "name": "Canadian Dollar",
        "country": "Canada"
    },
    "EUR": {
        "symbol": "€",
        "name": "Euro",
        "country": "Europe"
    },
    "AED": {
        "symbol": "د.إ",
        "name": "UAE Dirham",
        "country": "UAE"
    },
    "OMR": {
        "symbol": "ر.ع.",
        "name": "Omani Rial",
        "country": "Oman"
    },
    "SAR": {
        "symbol": "﷼",
        "name": "Saudi Riyal",
        "country": "Saudi Arabia"
    },
    "JPY": {
        "symbol": "¥",
        "name": "Japanese Yen",
        "country": "Japan"
    },
    "GBP": {
        "symbol": "£",
        "name": "British Pound",
        "country": "UK"
    }
}

REGION_COSTS = {
    "India": {
        "cost_per_kw": 800,  # USD
        "base_currency": "USD",
        "local_currency": "INR",
        "om_cost_percent": 1.0,
        "default_tariff": 0.10,  # USD
        "default_escalation": 3.0,
        "countries": ["India"]
    },
    "Middle East": {
        "cost_per_kw": 1050,  # USD
        "base_currency": "USD",
        "local_currency": "USD",
        "om_cost_percent": 1.5,
        "default_tariff": 0.10,  # USD
        "default_escalation": 2.5,
        "countries": ["UAE", "Oman", "Saudi Arabia"]
    },
    "Europe": {
        "cost_per_kw": 900,  # USD
        "base_currency": "USD",
        "local_currency": "EUR",
        "om_cost_percent": 1.2,
        "default_tariff": 0.25,  # USD
        "default_escalation": 2.0,
        "countries": ["Europe"]
    },
    "North America": {
        "cost_per_kw": 900,  # USD
        "base_currency": "USD",
        "local_currency": "USD",
        "om_cost_percent": 1.0,
        "default_tariff": 0.15,  # USD
        "default_escalation": 2.5,
        "countries": ["USA", "Canada"]
    },
    "Asia Pacific": {
        "cost_per_kw": 850,  # USD
        "base_currency": "USD",
        "local_currency": "USD",
        "om_cost_percent": 1.2,
        "default_tariff": 0.12,  # USD
        "default_escalation": 3.0,
        "countries": ["Japan"]
    }
}


# Default exchange rates (fallback when API is not available)
DEFAULT_EXCHANGE_RATES = {
    "USD": {
        "INR": 83.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "AED": 3.67,
        "OMR": 0.38,
        "SAR": 3.75,
        "JPY": 148.0,
        "CAD": 1.35
    }
}

# Utility functions
def get_country_region(country: str) -> str:
    """Get region for a given country"""
    for region, data in REGION_COSTS.items():
        if country in data["countries"]:
            return region
    return "North America"  # Default region

def get_currency_for_country(country: str) -> str:
    """Get default currency for a given country"""
    for currency, data in CURRENCIES.items():
        if data["country"] == country:
            return currency
    return "USD"  # Default currency

def get_default_tariff(region: str) -> float:
    """Get default electricity tariff for a region"""
    return REGION_COSTS.get(region, REGION_COSTS["North America"])["default_tariff"]

def get_om_cost_percent(region: str) -> float:
    """Get default O&M cost percentage for a region"""
    return REGION_COSTS.get(region, REGION_COSTS["North America"])["om_cost_percent"]