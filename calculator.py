# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:30:12 2025

@author: amrit
"""

import pandas as pd
import pvlib
from pvlib import location, irradiance
import plotly.express as px
import plotly.graph_objects as go
from pvlib import solarposition
import pytz
from datetime import datetime
from inverter_database import (
    get_suitable_inverters,
    get_inverter_details,
    calculate_inverter_configuration,
    COMMON_INVERTERS
)
import streamlit as st

def calculate_solar(latitude, longitude, tz_str, surface_tilt, surface_azimuth,
                    module_efficiency, performance_ratio, plant_capacity_kw,
                    module_area, module_watt_peak):
    """
    Core solar calculation engine with improved error handling and inverter integration
    
    Parameters:
    -----------
    latitude : float
        Location latitude in degrees
    longitude : float
        Location longitude in degrees
    tz_str : str
        Timezone string (e.g., 'UTC', 'Asia/Kolkata')
    surface_tilt : float
        Surface tilt angle in degrees (0-90)
    surface_azimuth : float
        Surface azimuth angle in degrees (-180 to 180)
    module_efficiency : float
        Solar module efficiency (0-1)
    performance_ratio : float
        System performance ratio (0-1)
    plant_capacity_kw : float
        Plant capacity in kilowatts
    module_area : float
        Area of single module in square meters
    module_watt_peak : float
        Peak wattage of single module
        
    Returns:
    --------
    dict
        Dictionary containing irradiation and energy production data and figures
    """
    # Input validation
    if None in [latitude, longitude, surface_tilt, surface_azimuth]:
        raise ValueError("Required parameters cannot be None")
    
    try:
        latitude = float(latitude)
        longitude = float(longitude)
        surface_tilt = float(surface_tilt)
        surface_azimuth = float(surface_azimuth)
        module_efficiency = float(module_efficiency)
        performance_ratio = float(performance_ratio)
        plant_capacity_kw = float(plant_capacity_kw)
        module_area = float(module_area)
        module_watt_peak = float(module_watt_peak)
    except ValueError as e:
        raise ValueError(f"Parameter type conversion failed: {str(e)}")
    
    # Validate parameter ranges
    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if not (0 <= surface_tilt <= 90):
        raise ValueError("Surface tilt must be between 0 and 90 degrees")
    if not (-180 <= surface_azimuth <= 180):
        raise ValueError("Surface azimuth must be between -180 and 180 degrees")
    if not (0 < module_efficiency <= 1):
        raise ValueError("Module efficiency must be between 0 and 1")
    if not (0 < performance_ratio <= 1):
        raise ValueError("Performance ratio must be between 0 and 1")
    if plant_capacity_kw <= 0:
        raise ValueError("Plant capacity must be greater than 0")
    
    # Get inverter configuration if available
    inverter_dc_ac_ratio = 1.0  # Default value
    number_of_inverters = 1     # Default value
    inverter_efficiency = 0.98  # Default inverter efficiency
    
    if st.session_state.inverter_params:
        inverter_config = st.session_state.inverter_params['configuration']
        inverter_specs = st.session_state.inverter_params['specifications']
        
        number_of_inverters = inverter_config['num_inverters']
        inverter_dc_ac_ratio = inverter_config['dc_ac_ratio']
        inverter_efficiency = inverter_specs['max_efficiency']
        
        # Calculate effective capacity considering inverter limitations
        total_ac_capacity = number_of_inverters * inverter_specs['nominal_ac_power']
        effective_capacity = min(plant_capacity_kw, total_ac_capacity)
        
        # Update performance ratio to include inverter efficiency
        performance_ratio = performance_ratio * inverter_efficiency
    else:
        effective_capacity = plant_capacity_kw
        
    # Ensure proper timezone handling
    try:
        tz = pytz.timezone(tz_str)
    except pytz.exceptions.UnknownTimeZoneError:
        tz = pytz.UTC
        
    # Calculate system parameters
    total_modules = round(plant_capacity_kw * 1000 / module_watt_peak)
    calculated_capacity = (total_modules * module_watt_peak) / 1000
    total_area = module_area * total_modules

    # Create date range for calculations
    start = datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(year=start.year + 1)
    times = pd.date_range(start=start, end=end, freq='H', tz=tz)[:-1]
    
    # Initialize Location object with error handling
    try:
        site = location.Location(latitude, longitude, tz=tz)
    except Exception as e:
        raise ValueError(f"Invalid location parameters: {str(e)}")

    # Calculate solar position with validation
    try:
        solar_position = solarposition.get_solarposition(
            time=times,
            latitude=latitude,
            longitude=longitude,
            altitude=site.altitude,
            temperature=20,  # Default temperature
            pressure=101325  # Default pressure (sea level)
        )
    except Exception as e:
        raise ValueError(f"Solar position calculation failed: {str(e)}")

    # Get clear sky data
    try:
        irrad_data = site.get_clearsky(times)
        
        # Validate irradiance data
        if irrad_data['dni'].isnull().all() or irrad_data['ghi'].isnull().all() or irrad_data['dhi'].isnull().all():
            raise ValueError("Invalid irradiance data calculated")
            
    except Exception as e:
        raise ValueError(f"Clear sky calculation failed: {str(e)}")

    # Calculate POA irradiance with validation
    try:
        poa_irrad = irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_position['apparent_zenith'].fillna(90),
            solar_azimuth=solar_position['azimuth'].fillna(0),
            dni=irrad_data['dni'],
            ghi=irrad_data['ghi'],
            dhi=irrad_data['dhi']
        )
    except Exception as e:
        raise ValueError(f"POA irradiance calculation failed: {str(e)}")

    # Irradiation analysis
    daily_gii = (poa_irrad['poa_global']/1000).resample('D').sum()
    monthly_gii = daily_gii.resample('M').sum()

    # Energy production calculation with inverter clipping
    hourly_energy = (poa_irrad['poa_global']/1000) * module_efficiency * performance_ratio
    
    # Apply inverter clipping if configuration is available
    if st.session_state.inverter_params:
        max_ac_power_per_inverter = inverter_specs['nominal_ac_power']
        max_total_ac_power = max_ac_power_per_inverter * number_of_inverters
        
        # More sophisticated clipping model
        clipping_threshold = max_total_ac_power * 0.99  # 95% of AC rating is a typical threshold
        hourly_energy_clipped = hourly_energy.copy()
        clipped_energy = hourly_energy[hourly_energy > clipping_threshold]
        clipping_losses = (clipped_energy - clipping_threshold).sum()
        
        # Clip hourly energy to inverter AC capacity
        hourly_energy = hourly_energy.clip(upper=max_total_ac_power)
    
    daily_energy = hourly_energy.resample('D').sum() * total_area
    monthly_energy = daily_energy.resample('M').sum()

    # Create DataFrames with error handling
    try:
        irradiation_daily = pd.DataFrame({
            "Date": daily_gii.index.tz_localize(None),
            "Daily Solar Irradiation (kWh/m²)": daily_gii.values
        })
        
        irradiation_monthly = pd.DataFrame({
            "Month": monthly_gii.index.strftime('%B'),
            "Monthly Solar Irradiation (kWh/m²)": monthly_gii.values
        })

        energy_daily = pd.DataFrame({
            "Date": daily_energy.index.tz_localize(None),
            "Daily Energy Production (kWh)": daily_energy.values
        })
        
        energy_monthly = pd.DataFrame({
            "Month": monthly_energy.index.strftime('%B'),
            "Monthly Energy Production (kWh)": monthly_energy.values
        })
    except Exception as e:
        raise ValueError(f"Error creating output DataFrames: {str(e)}")

    # Create figures with error handling
    try:
        irradiation_fig_daily = go.Figure()
        irradiation_fig_daily.add_trace(go.Scatter(
            x=irradiation_daily["Date"],
            y=irradiation_daily["Daily Solar Irradiation (kWh/m²)"],
            mode='lines',
            name='Daily Irradiation'
        ))
        irradiation_fig_daily.update_layout(
            title="Daily Solar Irradiation",
            xaxis_title="Date",
            yaxis_title="kWh/m²",
            showlegend=True
        )

        irradiation_fig_monthly = go.Figure()
        irradiation_fig_monthly.add_trace(go.Bar(
            x=irradiation_monthly["Month"],
            y=irradiation_monthly["Monthly Solar Irradiation (kWh/m²)"],
            marker_color='indianred',
            name='Monthly Irradiation'
        ))
        irradiation_fig_monthly.update_layout(
            title="Monthly Solar Irradiation",
            xaxis_title="Month",
            yaxis_title="kWh/m²",
            showlegend=True
        )

        energy_fig_daily = go.Figure()
        energy_fig_daily.add_trace(go.Scatter(
            x=energy_daily["Date"],
            y=energy_daily["Daily Energy Production (kWh)"],
            mode='lines',
            line=dict(color='blue'),
            name='Daily Energy'
        ))
        energy_fig_daily.update_layout(
            title="Daily Energy Production",
            xaxis_title="Date",
            yaxis_title="kWh",
            showlegend=True
        )

        energy_fig_monthly = go.Figure()
        energy_fig_monthly.add_trace(go.Bar(
            x=energy_monthly["Month"],
            y=energy_monthly["Monthly Energy Production (kWh)"],
            marker_color='blue',
            name='Monthly Energy'
        ))
        energy_fig_monthly.update_layout(
            title="Monthly Energy Production",
            xaxis_title="Month",
            yaxis_title="kWh",
            showlegend=True
        )
    except Exception as e:
        raise ValueError(f"Error creating figures: {str(e)}")

    # Return results with inverter information
    try:
        return {
            "irradiation": {
                "daily": irradiation_daily,
                "monthly": irradiation_monthly,
                "metrics": {
                    "max_daily": float(daily_gii.max()),
                    "min_daily": float(daily_gii.min()),
                    "total_yearly": float(daily_gii.sum())
                },
                "figures": {
                    "daily": irradiation_fig_daily,
                    "monthly": irradiation_fig_monthly
                }
            },
            "energy": {
                "daily": energy_daily,
                "monthly": energy_monthly,
                "metrics": {
                    "max_daily": float(daily_energy.max()),
                    "min_daily": float(daily_energy.min()),
                    "total_yearly": float(daily_energy.sum())
                },
                "figures": {
                    "daily": energy_fig_daily,
                    "monthly": energy_fig_monthly
                }
            },
            "system": {
                "total_modules": int(total_modules),
                "total_area": float(total_area),
                "calculated_capacity": float(calculated_capacity),
                "inverter_configuration": st.session_state.inverter_params if st.session_state.inverter_params else None,
                "effective_dc_ac_ratio": inverter_dc_ac_ratio,
                "number_of_inverters": number_of_inverters,
                "inverter_efficiency": inverter_efficiency
            }
        }
    except Exception as e:
        raise ValueError(f"Error creating return dictionary: {str(e)}")