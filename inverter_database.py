# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:30:15 2025
@author: amrit

Solar Inverter Database - Focused on Sungrow and Huawei string inverters
"""

from typing import Dict, List, Any
import math
import traceback
import streamlit as st

# Define common solar inverter specifications
COMMON_INVERTERS = {
    "Sungrow String Inverters": {
        "SG5KTL-MT": {
            "nominal_ac_power": 5.0,    # kW
            "max_dc_power": 7.5,        # kW
            "max_efficiency": 0.985,     # 98.5%
            "euro_efficiency": 0.975,    # 97.5%
            "mppt_range": [140, 850],    # V
            "max_dc_voltage": 1000,      # V
            "rated_mppt_voltage": 600,   # V
            "max_input_current": 15,     # A per MPPT
            "number_of_mppt": 2,
            "manufacturer": "Sungrow",
            "suitable_for": ["Residential", "Commercial"]
        },
        "SG8KTL-MT": {
            "nominal_ac_power": 8.0,
            "max_dc_power": 12.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1000,
            "rated_mppt_voltage": 600,
            "max_input_current": 15,
            "number_of_mppt": 2,
            "manufacturer": "Sungrow",
            "suitable_for": ["Residential", "Commercial"]
        },
        "SG10KTL-MT": {
            "nominal_ac_power": 10.0,
            "max_dc_power": 15.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1000,
            "rated_mppt_voltage": 600,
            "max_input_current": 15,
            "number_of_mppt": 2,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial"]
        },
        "SG12KTL-MT": {
            "nominal_ac_power": 12.0,
            "max_dc_power": 18.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1000,
            "rated_mppt_voltage": 600,
            "max_input_current": 15,
            "number_of_mppt": 2,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial"]
        },
        "SG15KTL-MT": {
            "nominal_ac_power": 15.0,
            "max_dc_power": 22.5,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1000,
            "rated_mppt_voltage": 600,
            "max_input_current": 20,
            "number_of_mppt": 2,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial"]
        },
        "SG20KTL-MT": {
            "nominal_ac_power": 20.0,
            "max_dc_power": 30.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1000,
            "rated_mppt_voltage": 600,
            "max_input_current": 20,
            "number_of_mppt": 2,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial"]
        },
        "SG30KTL-MT": {
            "nominal_ac_power": 30.0,
            "max_dc_power": 45.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 30,
            "number_of_mppt": 3,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial"]
        },
        "SG33KTL-MT": {
            "nominal_ac_power": 33.0,
            "max_dc_power": 49.5,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 30,
            "number_of_mppt": 3,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial"]
        },
        "SG40KTL-MT": {
            "nominal_ac_power": 40.0,
            "max_dc_power": 60.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 30,
            "number_of_mppt": 4,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial", "Industrial"]
        },
        "SG50KTL-MT": {
            "nominal_ac_power": 50.0,
            "max_dc_power": 75.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 30,
            "number_of_mppt": 4,
            "manufacturer": "Sungrow",
            "suitable_for": ["Commercial", "Industrial"]
        },
        "SG100KTL-M": {
            "nominal_ac_power": 100.0,
            "max_dc_power": 150.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 40,
            "number_of_mppt": 6,
            "manufacturer": "Sungrow",
            "suitable_for": ["Industrial"]
        },
        "SG125KTL-M": {
            "nominal_ac_power": 125.0,
            "max_dc_power": 187.5,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 40,
            "number_of_mppt": 6,
            "manufacturer": "Sungrow",
            "suitable_for": ["Industrial"]
        },
        "SG150KTL-M": {
            "nominal_ac_power": 150.0,
            "max_dc_power": 225.0,
            "max_efficiency": 0.985,
            "euro_efficiency": 0.975,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 40,
            "number_of_mppt": 8,
            "manufacturer": "Sungrow",
            "suitable_for": ["Industrial"]
        }
    },
    "Huawei String Inverters": {
        "SUN2000-8KTL-M0": {
            "nominal_ac_power": 8.0,
            "max_dc_power": 12.0,
            "max_efficiency": 0.987,
            "euro_efficiency": 0.977,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 600,
            "max_input_current": 15,
            "number_of_mppt": 2,
            "manufacturer": "Huawei",
            "suitable_for": ["Residential", "Commercial"]
        },
        "SUN2000-12KTL-M0": {
            "nominal_ac_power": 12.0,
            "max_dc_power": 18.0,
            "max_efficiency": 0.987,
            "euro_efficiency": 0.977,
            "mppt_range": [140, 850],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 600,
            "max_input_current": 15,
            "number_of_mppt": 2,
            "manufacturer": "Huawei",
            "suitable_for": ["Commercial"]
        },
        "SUN2000-20KTL-M0": {
            "nominal_ac_power": 20.0,
            "max_dc_power": 30.0,
            "max_efficiency": 0.987,
            "euro_efficiency": 0.977,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 22,
            "number_of_mppt": 2,
            "manufacturer": "Huawei",
            "suitable_for": ["Commercial"]
        },
        "SUN2000-30KTL-M3": {
            "nominal_ac_power": 30.0,
            "max_dc_power": 45.0,
            "max_efficiency": 0.987,
            "euro_efficiency": 0.977,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 26,
            "number_of_mppt": 3,
            "manufacturer": "Huawei",
            "suitable_for": ["Commercial"]
        },
        "SUN2000-50KTL-M3": {
            "nominal_ac_power": 50.0,
            "max_dc_power": 75.0,
            "max_efficiency": 0.987,
            "euro_efficiency": 0.977,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 30,
            "number_of_mppt": 4,
            "manufacturer": "Huawei",
            "suitable_for": ["Commercial", "Industrial"]
        },
        "SUN2000-100KTL-M1": {
            "nominal_ac_power": 100.0,
            "max_dc_power": 150.0,
            "max_efficiency": 0.987,
            "euro_efficiency": 0.977,
            "mppt_range": [200, 950],
            "max_dc_voltage": 1100,
            "rated_mppt_voltage": 620,
            "max_input_current": 40,
            "number_of_mppt": 6,
            "manufacturer": "Huawei",
            "suitable_for": ["Industrial"]
        }
    }
}


# Add large central inverters
CENTRAL_INVERTERS = {
    "Sungrow Central": {
        "SG3125HV": {
            "nominal_ac_power": 3125,  # kW
            "max_dc_power": 3437.5,    # kW
            "max_dc_voltage": 1500,    # V
            "mppt_range": [500, 1500], # V
            "rated_mppt_voltage": 850, # V
            "max_input_current": 3508,  # A
            "number_of_mppt": 12,
            "max_efficiency": 0.989,
            "euro_efficiency": 0.986,
            "type": "central"
        },
        "SG2500HV": {
            "nominal_ac_power": 2500,  # kW
            "max_dc_power": 2750,     # kW
            "max_dc_voltage": 1500,    # V
            "mppt_range": [500, 1500], # V
            "rated_mppt_voltage": 850, # V
            "max_input_current": 3000,  # A
            "number_of_mppt": 10,
            "max_efficiency": 0.989,
            "euro_efficiency": 0.986,
            "type": "central"
        }
    },
    "Huawei Central": {
        "SUN2000-3000KTL": {
            "nominal_ac_power": 3000,  # kW
            "max_dc_power": 3300,     # kW
            "max_dc_voltage": 1500,    # V
            "mppt_range": [500, 1500], # V
            "rated_mppt_voltage": 850, # V
            "max_input_current": 3500,  # A
            "number_of_mppt": 12,
            "max_efficiency": 0.990,
            "euro_efficiency": 0.987,
            "type": "central"
        },
        "SUN2000-2500KTL": {
            "nominal_ac_power": 2500,  # kW
            "max_dc_power": 2750,     # kW
            "max_dc_voltage": 1500,    # V
            "mppt_range": [500, 1500], # V
            "rated_mppt_voltage": 850, # V
            "max_input_current": 3000,  # A
            "number_of_mppt": 10,
            "max_efficiency": 0.990,
            "euro_efficiency": 0.987,
            "type": "central"
        }
    }
}

def get_suitable_inverters(dc_capacity: float, 
                         manufacturer: str = None,
                         application_type: str = None) -> Dict[str, List[str]]:
    """
    Get list of suitable inverters based on DC capacity
    
    Parameters:
    -----------
    dc_capacity : float
        System DC capacity in kW (this is what user enters)
    manufacturer : str, optional
        Preferred manufacturer (Sungrow/Huawei)
    application_type : str, optional
        Type of application (Residential/Commercial/Industrial)
    """
    suitable_inverters = {}
    MAX_DC_AC_RATIO = 1.2  # Maximum allowable DC/AC ratio
    MIN_DC_AC_RATIO = 0.95  # Minimum practical DC/AC ratio
    
    # Determine application type based on DC capacity
    if not application_type:
        if dc_capacity <= 10:
            application_type = "Residential"
        elif dc_capacity <= 100:
            application_type = "Commercial"
        else:
            application_type = "Industrial"
    
    # Filter by manufacturer and create categories
    categories = []
    central_categories = []
    if manufacturer == "Sungrow":
        categories = ["Sungrow String Inverters"]
        central_categories = ["Sungrow Central"]
    elif manufacturer == "Huawei":
        categories = ["Huawei String Inverters"]
        central_categories = ["Huawei Central"]
    else:
        categories = list(COMMON_INVERTERS.keys())
        central_categories = list(CENTRAL_INVERTERS.keys())

    # Process string inverters for smaller systems (< 1000 kW)
    if dc_capacity < 1000:
        for category in categories:
            suitable_inverters[category] = []
            inverters = COMMON_INVERTERS[category]
            
            # Find suitable inverter combinations
            suitable_configs = []
            for model, specs in inverters.items():
                inverter_ac_power = specs["nominal_ac_power"]
                
                # Calculate possible number of inverters
                min_inverters = math.ceil(dc_capacity / (inverter_ac_power * MAX_DC_AC_RATIO))
                max_inverters = math.floor(dc_capacity / (inverter_ac_power * MIN_DC_AC_RATIO))
                
                # Check all possible inverter quantities in this range
                for num_inverters in range(min_inverters, max_inverters + 1):
                    # Skip if too many inverters (practical limit)
                    if num_inverters > 5:
                        continue
                        
                    total_ac_capacity = num_inverters * inverter_ac_power
                    dc_ac_ratio = dc_capacity / total_ac_capacity
                    
                    # Check if this combination is viable
                    if MIN_DC_AC_RATIO <= dc_ac_ratio <= MAX_DC_AC_RATIO:
                        suitable_configs.append({
                            "model": model,
                            "num_inverters": num_inverters,
                            "total_ac_capacity": total_ac_capacity,
                            "dc_ac_ratio": dc_ac_ratio,
                            "capacity_diff": abs(dc_capacity - (total_ac_capacity * dc_ac_ratio)),
                            "inverter_ac_power": inverter_ac_power
                        })
            
            # Sort configurations
            suitable_configs.sort(key=lambda x: (
                x["num_inverters"],
                abs(x["dc_ac_ratio"] - 1.1),
                -x["inverter_ac_power"]
            ))
            
            # Add top configurations
            suitable_inverters[category] = [
                {
                    "model": config["model"],
                    "num_inverters": config["num_inverters"],
                    "total_ac": config["total_ac_capacity"],
                    "dc_ac_ratio": config["dc_ac_ratio"]
                }
                for config in suitable_configs[:6]  # Show top 6 options
            ]

    # Process central inverters for larger systems (>= 1000 kW)
    if dc_capacity >= 1000:
        for category in central_categories:
            suitable_inverters[f"{category} Central"] = []
            inverters = CENTRAL_INVERTERS[category]
            
            suitable_configs = []
            for model, specs in inverters.items():
                inverter_ac_power = specs["nominal_ac_power"]
                
                # Calculate possible number of inverters
                min_inverters = math.ceil(dc_capacity / (inverter_ac_power * MAX_DC_AC_RATIO))
                max_inverters = math.floor(dc_capacity / (inverter_ac_power * MIN_DC_AC_RATIO))
                
                # Check all possible central inverter combinations
                for num_inverters in range(min_inverters, max_inverters + 1):
                    # For central inverters, allow more units due to size
                    if num_inverters > 10:  # Higher limit for central inverters
                        continue
                        
                    total_ac_capacity = num_inverters * inverter_ac_power
                    dc_ac_ratio = dc_capacity / total_ac_capacity
                    
                    # Check if this combination is viable
                    if MIN_DC_AC_RATIO <= dc_ac_ratio <= MAX_DC_AC_RATIO:
                        suitable_configs.append({
                            "model": model,
                            "num_inverters": num_inverters,
                            "total_ac_capacity": total_ac_capacity,
                            "dc_ac_ratio": dc_ac_ratio,
                            "capacity_diff": abs(dc_capacity - (total_ac_capacity * dc_ac_ratio)),
                            "inverter_ac_power": inverter_ac_power
                        })
            
            # Sort configurations
            suitable_configs.sort(key=lambda x: (
                x["num_inverters"],
                abs(x["dc_ac_ratio"] - 1.1),
                -x["inverter_ac_power"]
            ))
            
            # Add top configurations
            suitable_inverters[f"{category} Central"] = [
                {
                    "model": config["model"],
                    "num_inverters": config["num_inverters"],
                    "total_ac": config["total_ac_capacity"],
                    "dc_ac_ratio": config["dc_ac_ratio"]
                }
                for config in suitable_configs[:6]  # Show top 6 options
            ]
    
    # Remove empty categories
    return {k: v for k, v in suitable_inverters.items() if v}

def calculate_inverter_configuration(dc_capacity: float, 
                                   inverter_model: str,
                                   module_vmp: float,
                                   module_imp: float,
                                   module_voc: float,
                                   module_isc: float,
                                   temperature_coefficient_voc: float = -0.29) -> Dict[str, Any]:
    """
    Calculate optimal inverter configuration for both string and central inverters
    
    Parameters:
    -----------
    dc_capacity : float
        System DC capacity in kW
    inverter_model : str
        Selected inverter model
    module_vmp : float
        Module voltage at maximum power point
    module_imp : float
        Module current at maximum power point
    module_voc : float
        Module open circuit voltage
    module_isc : float
        Module short circuit current
    temperature_coefficient_voc : float
        Temperature coefficient for Voc (%/°C)
        
    Returns:
    --------
    Dict[str, Any]
        Inverter configuration details
    """
    try:
        inverter_specs = get_inverter_details(inverter_model)
        if not inverter_specs:
            return None
        
        # Get inverter power and type
        inverter_power = inverter_specs["nominal_ac_power"]
        inverter_type = inverter_specs.get("inverter_type", "string")
        
        # Calculate number of inverters
        num_inverters = math.ceil(dc_capacity / inverter_power)
        
        # If inverter is larger than system size, use just one
        if inverter_power >= dc_capacity:
            num_inverters = 1
        
        # Calculate total AC capacity and DC/AC ratio
        total_ac_capacity = num_inverters * inverter_power
        dc_ac_ratio = dc_capacity / total_ac_capacity
        
        # Temperature corrections
        min_temp = -10 if inverter_type == "string" else 0  # Different temp range for central
        max_temp = 70 if inverter_type == "string" else 50
        max_voltage_correction = 1 + (min_temp - 25) * temperature_coefficient_voc / 100
        min_voltage_correction = 1 + (max_temp - 25) * temperature_coefficient_voc / 100
        
        # Calculate module string limits
        min_modules = math.ceil(inverter_specs["mppt_range"][0] / (module_vmp * min_voltage_correction))
        max_modules = math.floor(inverter_specs["max_dc_voltage"] / (module_voc * max_voltage_correction))
        
        # Calculate optimal string length based on inverter type
        if inverter_type == "string":
            # For string inverters, optimize for MPPT voltage
            optimal_modules_per_string = round(
                (inverter_specs["rated_mppt_voltage"] / module_vmp)
            )
        else:
            # For central inverters, optimize for maximum power point
            optimal_modules_per_string = round(
                (inverter_specs["rated_mppt_voltage"] * 0.95 / module_vmp)  # 95% of rated voltage
            )
        
        # Ensure string length is within limits
        optimal_modules_per_string = max(min(optimal_modules_per_string, max_modules), min_modules)
        
        # Calculate total number of modules needed
        total_modules = math.ceil((dc_capacity * 1000) / (module_vmp * module_imp))
        
        # Calculate number of strings needed
        total_strings = math.ceil(total_modules / optimal_modules_per_string)
        
        # Calculate strings per inverter
        strings_per_inverter = math.ceil(total_strings / num_inverters)
        
        # Calculate strings per MPPT
        strings_per_mppt = math.ceil(strings_per_inverter / inverter_specs["number_of_mppt"])
        
        # Additional checks for central inverters
        if inverter_type == "central":
            # Check current limits
            string_current = module_isc
            total_current = string_current * strings_per_mppt
            if total_current > inverter_specs["max_input_current"]:
                strings_per_mppt = math.floor(inverter_specs["max_input_current"] / string_current)
                strings_per_inverter = strings_per_mppt * inverter_specs["number_of_mppt"]
                total_strings = strings_per_inverter * num_inverters
        
        return {
            "num_inverters": num_inverters,
            "dc_ac_ratio": dc_ac_ratio,
            "optimal_modules_per_string": optimal_modules_per_string,
            "total_strings": total_strings,
            "strings_per_inverter": strings_per_inverter,
            "strings_per_mppt": strings_per_mppt,
            "number_of_mppt": inverter_specs["number_of_mppt"],
            "manufacturer": inverter_specs.get("manufacturer", "Unknown"),
            "inverter_capacity": inverter_specs["nominal_ac_power"],
            "inverter_type": inverter_type,
            "category": inverter_specs.get("category", "Unknown")
        }
        
    except Exception as e:
        st.error(f"Error calculating inverter configuration: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_inverter_details(model: str) -> Dict[str, Any]:
    """
    Get detailed specifications for a specific inverter model
    
    Parameters:
    -----------
    model : str
        Inverter model name to look up
        
    Returns:
    --------
    Dict[str, Any]
        Inverter specifications or None if not found
    """
    # Check string inverters
    for category, inverters in COMMON_INVERTERS.items():
        if model in inverters:
            specs = inverters[model].copy()
            specs['inverter_type'] = 'string'
            specs['category'] = category
            return specs
    
    # Check central inverters
    for manufacturer, inverters in CENTRAL_INVERTERS.items():
        if model in inverters:
            specs = inverters[model].copy()
            specs['inverter_type'] = 'central'
            specs['category'] = f"{manufacturer} Central"
            return specs
            
    return None
