# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:30:15 2025
@author: amrit

Ground Coverage Ratio (GCR) configurations for different solar installation types
"""

from typing import Dict, Any

# Define GCR configurations for different project categories and installation types
GCR_CONFIGURATIONS = {
    "Utility-Scale": {
        "name": "Utility-Scale Solar Farms",
        "description": "Large-scale solar installations typically over 1MW",
        "installation_types": {
            "Fixed-Tilt": {
                "gcr_range": (0.30, 0.40),
                "default_gcr": 0.35,
                "description": "Fixed-tilt ground mount systems with optimal row spacing"
            },
            "Single-Axis Tracker": {
                "gcr_range": (0.25, 0.35),
                "default_gcr": 0.30,
                "description": "Single-axis tracking systems with spacing for rotation"
            },
            "Dual-Axis Tracker": {
                "gcr_range": (0.15, 0.25),
                "default_gcr": 0.20,
                "description": "Dual-axis tracking systems requiring maximum spacing"
            }
        }
    },
    
    "Commercial/Industrial": {
        "name": "Commercial & Industrial",
        "description": "Rooftop and ground mount systems for C&I applications",
        "installation_types": {
            "Flat Rooftop": {
                "gcr_range": (0.50, 0.70),
                "default_gcr": 0.60,
                "description": "Installations on flat commercial rooftops"
            },
            "Sloped Rooftop": {
                "gcr_range": (0.70, 0.90),
                "default_gcr": 0.80,
                "description": "Installations on pitched commercial roofs"
            },
            "Carport": {
                "gcr_range": (0.80, 0.90),
                "default_gcr": 0.85,
                "description": "Solar carport structures for parking areas"
            }
        }
    },
    
    "Residential": {
        "name": "Residential Systems",
        "description": "Small-scale residential rooftop installations",
        "installation_types": {
            "Pitched Roof": {
                "gcr_range": (0.50, 0.70),
                "default_gcr": 0.60,
                "description": "Standard residential roof installations"
            },
            "BIPV": {
                "gcr_range": (0.80, 1.00),
                "default_gcr": 0.90,
                "description": "Building-integrated photovoltaic systems"
            }
        }
    },
    
    "Special Applications": {
        "name": "Special Applications",
        "description": "Specialized solar installations",
        "installation_types": {
            "Floating Solar": {
                "gcr_range": (0.50, 0.70),
                "default_gcr": 0.60,
                "description": "Water-based floating solar installations"
            },
            "Agrivoltaics": {
                "gcr_range": (0.15, 0.30),
                "default_gcr": 0.25,
                "description": "Combined agriculture and solar installations"
            },
            "BIPV-Commercial": {
                "gcr_range": (0.80, 1.00),
                "default_gcr": 0.90,
                "description": "Building-integrated systems for commercial facades"
            }
        }
    }
}

def get_gcr_categories() -> list:
    """Get list of available project categories"""
    return list(GCR_CONFIGURATIONS.keys())

def get_installation_types(category: str) -> list:
    """Get installation types for a specific category"""
    if category in GCR_CONFIGURATIONS:
        return list(GCR_CONFIGURATIONS[category]["installation_types"].keys())
    return []

def get_gcr_value(category: str, installation_type: str) -> float:
    """Get default GCR value for specific category and installation type"""
    try:
        return GCR_CONFIGURATIONS[category]["installation_types"][installation_type]["default_gcr"]
    except KeyError:
        return 0.50  # Default fallback value

def get_gcr_range(category: str, installation_type: str) -> tuple:
    """Get GCR range for specific category and installation type"""
    try:
        return GCR_CONFIGURATIONS[category]["installation_types"][installation_type]["gcr_range"]
    except KeyError:
        return (0.30, 0.70)  # Default fallback range

def get_installation_description(category: str, installation_type: str) -> str:
    """Get description for specific installation type"""
    try:
        return GCR_CONFIGURATIONS[category]["installation_types"][installation_type]["description"]
    except KeyError:
        return "Installation type description not available"