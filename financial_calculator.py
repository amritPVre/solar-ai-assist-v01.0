# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:41:58 2025

@author: amrit
"""

# financial_calculator.py

import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Any
import json
import traceback
import math 
import plotly.graph_objects as go
# Currency and Regional Data
from financial_data import CURRENCIES, REGION_COSTS

class FinancialCalculator:
    def __init__(self):
        self.exchange_rates = {}
        self.current_settings = {}
    
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get real-time exchange rate between currencies"""
        try:
            if from_currency == to_currency:
                return 1.0
                
            # Using a free currency API
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                rate = data['rates'][to_currency]
                self.exchange_rates[(from_currency, to_currency)] = rate
                return rate
            return self.get_default_exchange_rate(from_currency, to_currency)
            
        except Exception as e:
            st.warning(f"Using default exchange rates: {str(e)}")
            return self.get_default_exchange_rate(from_currency, to_currency)

    def get_default_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Fallback exchange rates"""
        # Default exchange rates (update regularly)
        EXCHANGE_RATES = {
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
        
        if from_currency == to_currency:
            return 1.0
        elif from_currency == "USD":
            return EXCHANGE_RATES["USD"].get(to_currency, 1.0)
        elif to_currency == "USD":
            return 1.0 / EXCHANGE_RATES["USD"].get(from_currency, 1.0)
        else:
            usd_to_from = 1.0 / EXCHANGE_RATES["USD"].get(from_currency, 1.0)
            usd_to_to = EXCHANGE_RATES["USD"].get(to_currency, 1.0)
            return usd_to_from * usd_to_to

    def update_currency(self, new_currency: str, financial_inputs: Dict):
        """
        Update currency and convert all monetary values
        
        Parameters:
        -----------
        new_currency : str
            New currency code to convert to
        financial_inputs : Dict
            Dictionary containing all financial inputs to be converted
        """
        old_currency = self.current_settings['currency']
        
        if old_currency != new_currency:
            try:
                # Get exchange rate
                exchange_rate = self.get_exchange_rate(old_currency, new_currency)
                
                # Update project cost data
                if financial_inputs["project_cost"]:
                    project_cost = financial_inputs["project_cost"]
                    # Convert all monetary values in project cost
                    project_cost['cost_local'] *= exchange_rate
                    project_cost['base_cost_local'] *= exchange_rate
                    project_cost['cost_per_kw_local'] *= exchange_rate
                    if 'cost_per_kw_actual' in project_cost:
                        project_cost['cost_per_kw_actual'] *= exchange_rate
                    project_cost['currency'] = new_currency
                    project_cost['currency_symbol'] = CURRENCIES[new_currency]['symbol']
                    financial_inputs["project_cost"] = project_cost
                
                # Update O&M costs
                if financial_inputs["om_params"]:
                    om_params = financial_inputs["om_params"]
                    om_params['yearly_om_cost'] *= exchange_rate
                    financial_inputs["om_params"] = om_params
                
                # Update electricity tariffs
                if financial_inputs["electricity_data"]:
                    electricity_data = financial_inputs["electricity_data"]
                    if electricity_data["tariff"]["type"] == "flat":
                        electricity_data["tariff"]["rate"] *= exchange_rate
                    else:
                        for slab in electricity_data["tariff"]["slabs"]:
                            slab["rate"] *= exchange_rate
                    
                    electricity_data["yearly_cost"] *= exchange_rate
                    financial_inputs["electricity_data"] = electricity_data
                
                # Update regional data costs in current settings
                regional_data = self.current_settings['regional_data']
                regional_data['default_tariff'] *= exchange_rate
                
                # Update settings
                self.current_settings.update({
                    'currency': new_currency,
                    'currency_symbol': CURRENCIES[new_currency]['symbol'],
                    'exchange_rate': exchange_rate,
                    'regional_data': regional_data
                })
                
                # Debug information
                print(f"Currency updated from {old_currency} to {new_currency}")
                print(f"Exchange rate: {exchange_rate}")
                print("Updated project cost:", financial_inputs["project_cost"])
                print("Updated O&M params:", financial_inputs["om_params"])
                
            except Exception as e:
                st.error(f"Error updating currency: {str(e)}")
                st.error(traceback.format_exc())


    def initialize_financial_settings(self, default_country: str = None) -> Dict[str, Any]:
        """
        Initialize financial parameters with location integration
        
        Parameters:
        -----------
        default_country : str, optional
            Default country to use for initialization. If not provided, will prompt user to select.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing financial settings
        """
        if default_country:
            # Clean up country name if needed (remove leading/trailing spaces)
            country = default_country.strip()
            
            # Find the region for this country
            region = next(
                (reg for reg, data in REGION_COSTS.items() 
                 if country in data["countries"]),
                "North America"  # Default region if country not found
            )
            
            # Get regional defaults
            regional_data = REGION_COSTS[region]
            
            # Get local currency based on country
            local_currency = next(
                (code for code, data in CURRENCIES.items() 
                 if data["country"] == country),
                "USD"  # Default to USD if currency not found
            )
            
            # Display currency information
            st.info(f"Default currency for {country}: {CURRENCIES[local_currency]['name']} ({CURRENCIES[local_currency]['symbol']})")
            
        else:
            # If no default country provided, let user select
            country = st.selectbox(
                "Select your country:",
                [curr_data["country"] for curr_data in CURRENCIES.values()],
                help="Choose your country for cost estimation"
            )
            
            # Determine region based on country
            region = next(
                (reg for reg, data in REGION_COSTS.items() 
                 if country in data["countries"]),
                "North America"
            )
            
            # Get regional defaults
            regional_data = REGION_COSTS[region]
            
            # Get local currency based on country
            local_currency = next(
                (code for code, data in CURRENCIES.items() 
                 if data["country"] == country),
                "USD"
            )
            
            # Display currency information
            st.info(f"Default currency for {country}: {CURRENCIES[local_currency]['name']} ({CURRENCIES[local_currency]['symbol']})")
        
        settings = {
            "region": region,
            "country": country,
            "currency": local_currency,
            "currency_symbol": CURRENCIES[local_currency]['symbol'],
            "exchange_rate": 1.0,
            "regional_data": regional_data
        }
        
        self.current_settings = settings
        return settings
    
    def show_electricity_data(self) -> Dict[str, Any]:
        """Display electricity consumption and tariff data input interface"""
        if st.session_state.tab_states['om_settings_done']:
            st.subheader("Electricity Consumption & Tariff")
            
            # Initialize yearly_amount
            yearly_amount = 0.0
            consumption_data = None
            
            # System type selection
            system_type = st.radio(
                "Select system type:",
                ["Captive Consumption", "Grid Export Only"],
                help="Choose whether the system is for self-consumption or direct grid export"
            )
            
            # Only show consumption section for Captive Consumption
            if system_type == "Captive Consumption":
                consumption_enabled = st.checkbox(
                    "Include consumption data",
                    value=True,
                    help="Enable to include electricity consumption data in financial analysis"
                )
                
                if consumption_enabled:
                    # Consumption method selection
                    consumption_method = st.radio(
                        "How would you like to input electricity consumption?",
                        ["Monthly Average", "Month-wise Details"]
                    )
                    
                    if consumption_method == "Monthly Average":
                        with st.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                monthly_consumption = st.number_input(
                                    "Average monthly consumption (kWh):",
                                    min_value=0.0,
                                    value=1000.0,
                                    step=100.0,
                                    help="Enter your typical monthly electricity consumption"
                                )
                            with col2:
                                st.metric(
                                    "Yearly Consumption",
                                    f"{monthly_consumption * 12:,.0f} kWh",
                                    help="Estimated yearly consumption"
                                )
                            consumption_data = {"type": "average", "value": monthly_consumption}
                    else:
                        st.write("Enter month-wise consumption:")
                        monthly_data = {}
                        col1, col2, col3 = st.columns(3)
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        for idx, month in enumerate(months):
                            with [col1, col2, col3][idx % 3]:
                                monthly_data[month] = st.number_input(
                                    f"{month} (kWh)",
                                    min_value=0.0,
                                    value=1000.0,
                                    step=100.0,
                                    key=f"month_{month}_consumption"
                                )
                        
                        yearly_total = sum(monthly_data.values())
                        st.metric(
                            "Total Yearly Consumption",
                            f"{yearly_total:,.0f} kWh",
                            help="Total annual electricity consumption"
                        )
                        consumption_data = {"type": "detailed", "values": monthly_data}
            else:  # Grid Export Only
                st.info("💡 System configured for direct grid export. Only tariff information will be used for financial calculations.")
            
            # Tariff structure
            st.divider()
            st.subheader("Electricity Tariff Structure")
            
            tariff_type = st.radio(
                "Select tariff type:",
                ["Flat Rate", "Slab-based"],
                help="Choose how electricity is priced"
            )
            
            current_currency_symbol = self.current_settings['currency_symbol']
            
            if tariff_type == "Flat Rate":
                rate = st.number_input(
                    f"Electricity rate ({current_currency_symbol}/kWh):",
                    min_value=0.0,
                    value=float(self.current_settings['regional_data']['default_tariff']),
                    step=0.1,
                    help="Enter electricity rate for consumption/export"
                )
                tariff_data = {"type": "flat", "rate": rate}
                
                # Calculate yearly amount based on system type
                if system_type == "Grid Export Only":
                    yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
                    yearly_amount = yearly_generation * rate
                    st.metric(
                        "Estimated Yearly Revenue",
                        f"{current_currency_symbol}{yearly_amount:,.2f}",
                        help="Estimated annual revenue from grid export"
                    )
                elif consumption_data:
                    if consumption_data["type"] == "average":
                        yearly_amount = consumption_data["value"] * 12 * rate
                    else:
                        yearly_amount = sum(consumption_data["values"].values()) * rate
                    st.metric(
                        "Estimated Yearly Electricity Cost",
                        f"{current_currency_symbol}{yearly_amount:,.2f}",
                        help="Estimated annual electricity cost at current rates"
                    )
            else:
                # Slab-based tariff implementation
                st.write("Enter slab-wise rates:")
                slabs = []
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("First slab (0 to X units)")
                    first_slab_units = st.number_input("Units up to:", min_value=0)
                    first_slab_rate = st.number_input(
                        f"Rate ({current_currency_symbol}/kWh):",
                        min_value=0.0,
                        step=0.1
                    )
                    slabs.append({"units": first_slab_units, "rate": first_slab_rate})
                
                # Additional slabs
                add_more = st.checkbox("Add more slabs")
                slab_count = 1
                while add_more:
                    col1, col2 = st.columns(2)
                    with col1:
                        units = st.number_input(
                            "Units up to:",
                            min_value=slabs[-1]["units"],
                            key=f"slab_{slab_count}_units"
                        )
                    with col2:
                        rate = st.number_input(
                            f"Rate ({current_currency_symbol}/kWh):",
                            min_value=0.0,
                            step=0.1,
                            key=f"slab_{slab_count}_rate"
                        )
                    slabs.append({"units": units, "rate": rate})
                    slab_count += 1
                    add_more = st.checkbox("Add another slab?", key=f"add_slab_{slab_count}")
                
                tariff_data = {"type": "slab", "slabs": slabs}
                
                # Calculate yearly amount based on system type
                if system_type == "Grid Export Only":
                    yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
                    monthly_revenue = self._calculate_slab_cost(yearly_generation / 12, slabs)
                    yearly_amount = monthly_revenue * 12
                    st.metric(
                        "Estimated Yearly Revenue",
                        f"{current_currency_symbol}{yearly_amount:,.2f}",
                        help="Estimated annual revenue from grid export"
                    )
                elif consumption_data:
                    if consumption_data["type"] == "average":
                        monthly_cost = self._calculate_slab_cost(consumption_data["value"], slabs)
                        yearly_amount = monthly_cost * 12
                    else:
                        yearly_amount = sum(
                            self._calculate_slab_cost(consumption, slabs)
                            for consumption in consumption_data["values"].values()
                        )
                    st.metric(
                        "Estimated Yearly Electricity Cost",
                        f"{current_currency_symbol}{yearly_amount:,.2f}",
                        help="Estimated annual electricity cost at current rates"
                    )
            
            # Save button
            if st.button("Save Electricity Data"):
                electricity_data = {
                    "system_type": system_type,
                    "consumption": consumption_data,
                    "tariff": tariff_data,
                    "yearly_amount": yearly_amount
                }
                
                st.session_state.financial_inputs["electricity_data"] = electricity_data
                st.session_state.tab_states['electricity_done'] = True
                st.success("✅ Electricity data saved")
                st.rerun()
                
                return electricity_data
            
        else:
            st.warning("⚠️ Please complete O&M Settings first")
            return None
        
    def get_electricity_data(self, consumption_method: str = "Monthly Average", 
                          tariff_type: str = "Flat Rate") -> Dict[str, Any]:
        """Get electricity consumption and tariff data"""
        settings = self.current_settings
        
        st.subheader("Electricity Consumption & Tariff Details")
        
        # System type selection
        system_type = st.radio(
            "Select system type:",
            ["Captive Consumption", "Grid Export Only"],
            help="Choose whether the system is for self-consumption or direct grid export"
        )
        
        consumption_data = None
        yearly_amount = 0.0
        
        # Only show consumption section for Captive Consumption
        if system_type == "Captive Consumption":
            consumption_enabled = st.checkbox(
                "Include consumption data",
                value=True,
                help="Enable to include electricity consumption data in financial analysis"
            )
            
            if consumption_enabled:
                # Consumption method selection
                consumption_method = st.radio(
                    "How would you like to input electricity consumption?",
                    ["Monthly Average", "Month-wise Details"],
                    index=0 if consumption_method == "Monthly Average" else 1
                )
                
                if consumption_method == "Monthly Average":
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            monthly_consumption = st.number_input(
                                "Average monthly consumption (kWh):",
                                min_value=0.0,
                                value=1000.0,
                                step=100.0,
                                help="Enter your typical monthly electricity consumption"
                            )
                        with col2:
                            st.metric(
                                "Yearly Consumption",
                                f"{monthly_consumption * 12:,.0f} kWh",
                                help="Estimated yearly consumption"
                            )
                    consumption_data = {"type": "average", "value": monthly_consumption}
                else:
                    st.write("Enter month-wise consumption:")
                    monthly_data = {}
                    col1, col2, col3 = st.columns(3)
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    for idx, month in enumerate(months):
                        with [col1, col2, col3][idx % 3]:
                            monthly_data[month] = st.number_input(
                                f"{month} (kWh)",
                                min_value=0.0,
                                value=1000.0,
                                step=100.0,
                                key=f"month_{month}"
                            )
                    
                    yearly_total = sum(monthly_data.values())
                    st.metric(
                        "Total Yearly Consumption",
                        f"{yearly_total:,.0f} kWh",
                        help="Total annual electricity consumption"
                    )
                    consumption_data = {"type": "detailed", "values": monthly_data}
        else:
            st.info("💡 System configured for direct grid export. Only tariff information will be used for financial calculations.")
        
        # Tariff structure
        st.divider()
        st.subheader("Electricity Tariff Structure")
        
        tariff_type = st.radio(
            "Select tariff type:",
            ["Flat Rate", "Slab-based"],
            index=0 if tariff_type == "Flat Rate" else 1
        )
        
        if tariff_type == "Flat Rate":
            rate = st.number_input(
                f"Electricity rate ({settings['currency_symbol']}/kWh):",
                min_value=0.0,
                value=float(settings['regional_data']['default_tariff']),
                step=0.1,
                help="Enter your electricity rate"
            )
            tariff_data = {"type": "flat", "rate": rate}
            
            # Calculate yearly amount based on system type
            if system_type == "Grid Export Only":
                yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
                yearly_amount = yearly_generation * rate
                st.metric(
                    "Estimated Yearly Revenue",
                    f"{settings['currency_symbol']}{yearly_amount:,.2f}",
                    help="Estimated annual revenue from grid export"
                )
            elif consumption_data:
                if consumption_data["type"] == "average":
                    yearly_amount = consumption_data["value"] * 12 * rate
                else:
                    yearly_amount = yearly_total * rate
                st.metric(
                    "Estimated Yearly Electricity Cost",
                    f"{settings['currency_symbol']}{yearly_amount:,.2f}",
                    help="Estimated annual electricity cost at current rates"
                )
        else:
            st.write("Enter slab-wise rates:")
            slabs = []
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("First slab (0 to X units)")
                first_slab_units = st.number_input("Units up to:", min_value=0)
                first_slab_rate = st.number_input(
                    f"Rate ({settings['currency_symbol']}/kWh):",
                    min_value=0.0,
                    step=0.1
                )
                slabs.append({"units": first_slab_units, "rate": first_slab_rate})
            
            add_more = st.checkbox("Add more slabs")
            while add_more:
                col1, col2 = st.columns(2)
                with col1:
                    units = st.number_input(
                        "Units up to:",
                        min_value=slabs[-1]["units"],
                        key=f"slab_{len(slabs)}_units"
                    )
                with col2:
                    rate = st.number_input(
                        f"Rate ({settings['currency_symbol']}/kWh):",
                        min_value=0.0,
                        step=0.1,
                        key=f"slab_{len(slabs)}_rate"
                    )
                slabs.append({"units": units, "rate": rate})
                add_more = st.checkbox("Add another slab?", key=f"add_slab_{len(slabs)}")
            
            tariff_data = {"type": "slab", "slabs": slabs}
            
            # Calculate yearly amount based on system type
            if system_type == "Grid Export Only":
                yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
                monthly_revenue = self._calculate_slab_cost(yearly_generation / 12, slabs)
                yearly_amount = monthly_revenue * 12
                st.metric(
                    "Estimated Yearly Revenue",
                    f"{settings['currency_symbol']}{yearly_amount:,.2f}",
                    help="Estimated annual revenue from grid export"
                )
            elif consumption_data:
                if consumption_data["type"] == "average":
                    monthly_cost = self._calculate_slab_cost(consumption_data["value"], slabs)
                    yearly_amount = monthly_cost * 12
                else:
                    yearly_amount = sum(
                        self._calculate_slab_cost(consumption, slabs)
                        for consumption in consumption_data["values"].values()
                    )
                st.metric(
                    "Estimated Yearly Electricity Cost",
                    f"{settings['currency_symbol']}{yearly_amount:,.2f}",
                    help="Estimated annual electricity cost at current rates"
                )
        
        return {
            "system_type": system_type,
            "consumption": consumption_data,
            "tariff": tariff_data,
            "yearly_amount": yearly_amount
        }
    
    
    def _calculate_grid_export_revenue(self, energy_data: Dict) -> float:
        """Calculate revenue from grid export"""
        try:
            yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
            if energy_data['tariff']['type'] == 'flat':
                return yearly_generation * energy_data['tariff']['rate']
            else:
                # For slab-based tariff
                monthly_generation = yearly_generation / 12
                monthly_revenue = self._calculate_slab_cost(monthly_generation, energy_data['tariff']['slabs'])
                return monthly_revenue * 12
        except Exception as e:
            st.error(f"Error calculating grid export revenue: {str(e)}")
            return 0.0
    
    def _calculate_consumption_savings(self, energy_data: Dict) -> float:
        """Calculate savings from captive consumption"""
        try:
            if not energy_data['consumption']:
                # If no consumption data, treat as grid export
                return self._calculate_grid_export_revenue(energy_data)
                
            yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
            
            if energy_data['tariff']['type'] == 'flat':
                return yearly_generation * energy_data['tariff']['rate']
            else:
                monthly_generation = yearly_generation / 12
                monthly_savings = self._calculate_slab_cost(monthly_generation, energy_data['tariff']['slabs'])
                return monthly_savings * 12
        except Exception as e:
            st.error(f"Error calculating consumption savings: {str(e)}")
            return 0.0
    
            
    def calculate_project_cost(self, capacity_kw: float) -> Dict[str, float]:
        """Calculate project cost with currency conversion"""
        settings = self.current_settings
        regional_data = settings['regional_data']
        
        # Get base USD values
        base_cost_per_kw_usd = regional_data['cost_per_kw']
        base_cost_usd = capacity_kw * base_cost_per_kw_usd
        
        # Convert to selected currency
        exchange_rate = self.get_exchange_rate('USD', settings['currency'])
        base_cost_local = base_cost_usd * exchange_rate
        cost_per_kw_local = base_cost_per_kw_usd * exchange_rate
        
        return {
            "base_cost_usd": base_cost_usd,
            "base_cost_local": base_cost_local,
            "cost_per_kw_usd": base_cost_per_kw_usd,
            "cost_per_kw_local": cost_per_kw_local,
            "currency": settings['currency'],
            "currency_symbol": settings['currency_symbol']
        }
    
    def _calculate_slab_cost(self, consumption: float, slabs: List[Dict]) -> float:
        """Calculate electricity cost for slab-based tariff"""
        total_cost = 0
        remaining_units = consumption
        
        for i, slab in enumerate(slabs):
            if i == 0:
                units = min(remaining_units, slab['units'])
            else:
                prev_units = slabs[i-1]['units']
                units = min(remaining_units, slab['units'] - prev_units)
            
            total_cost += units * slab['rate']
            remaining_units -= units
            
            if remaining_units <= 0:
                break
                
            # If this is the last slab and there are remaining units
            if i == len(slabs) - 1 and remaining_units > 0:
                total_cost += remaining_units * slab['rate']
        
        return total_cost

    def get_project_cost_input(self, capacity_kw: float) -> Dict[str, Any]:
        """Get user input for project cost with consistent UI"""
        st.subheader("Project Cost Settings")
        
        # Display system capacity
        st.metric(
            "System Capacity",
            f"{capacity_kw:.1f} kW",
            help="Total system capacity"
        )
        
        # Calculate initial cost estimates
        cost_data = self.calculate_project_cost(capacity_kw)
        
        st.markdown("#### Enter Project Cost")
        
        # Per kW cost input
        col1, col2 = st.columns(2)
        with col1:
            new_cost_per_kw = st.number_input(
                f"Cost per kW ({cost_data['currency_symbol']}/kW):",
                min_value=0.0,
                value=float(cost_data['cost_per_kw_local']),
                step=100.0,
                format="%.0f",
                help="Enter your expected or quoted cost per kW"
            )
            
            # Calculate and display total cost
            new_total_cost = new_cost_per_kw * capacity_kw
            with col2:
                st.metric(
                    "Total Project Cost",
                    f"{cost_data['currency_symbol']}{new_total_cost:,.0f}",
                    help="Total project cost based on per kW cost"
                )
            
            # Add comparison with base cost
            st.markdown("#### Cost Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Market Reference Cost",
                    f"{cost_data['currency_symbol']}{cost_data['base_cost_local']:,.0f}",
                    help="Typical market cost for your region"
                )
            with col2:
                cost_difference = ((new_total_cost - cost_data['base_cost_local']) / cost_data['base_cost_local']) * 100
                st.metric(
                    "Cost Difference",
                    f"{cost_difference:+.1f}%",
                    help="Difference from market reference cost",
                    delta_color="inverse"
                )
            
            # Update cost data with new values
            cost_data.update({
                'cost_local': new_total_cost,
                'cost_per_kw_actual': new_cost_per_kw
            })
            
            return cost_data

    def get_om_parameters(self, project_cost: float) -> Dict[str, float]:
        """Get O&M cost parameters"""
        settings = self.current_settings
        regional_data = settings['regional_data']
        
        st.subheader("Operation & Maintenance Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            yearly_om_percent = st.number_input(
                "Yearly O&M cost (% of project cost):",
                min_value=0.0,
                max_value=5.0,
                value=regional_data['om_cost_percent'],
                step=0.1,
                help="Annual O&M cost as percentage of project cost"
            )
            
            yearly_om_cost = project_cost * (yearly_om_percent / 100)
            st.metric(
                "Yearly O&M Cost",
                f"{settings['currency_symbol']}{yearly_om_cost:,.0f}"
            )
        
        with col2:
            om_escalation = st.number_input(
                "Yearly O&M cost escalation (%):",
                min_value=0.0,
                max_value=10.0,
                value=regional_data['default_escalation'],
                step=0.5,
                help="Annual increase in O&M costs"
            )
            
            tariff_escalation = st.number_input(
                "Yearly tariff escalation (%):",
                min_value=0.0,
                max_value=10.0,
                value=regional_data['default_escalation'],
                step=0.5,
                help="Annual increase in electricity tariff"
            )
        
        return {
            "yearly_om_cost": yearly_om_cost,
            "om_escalation": om_escalation/100,  # Convert to decimal
            "tariff_escalation": tariff_escalation/100  # Convert to decimal
        }
    
    
    def _calculate_yearly_savings(self, energy_data: Dict) -> float:
        """
        Calculate yearly electricity cost savings based on consumption and tariff data
        """
        try:
            # Get total yearly energy production from system
            yearly_energy = st.session_state.results['energy']['metrics']['total_yearly']
            
            # Calculate savings based on tariff type
            if energy_data['tariff']['type'] == 'flat':
                # Simple calculation for flat rate
                savings = yearly_energy * energy_data['tariff']['rate']
            else:
                # For slab-based tariff, calculate month by month
                monthly_energy = yearly_energy / 12  # Average monthly production
                monthly_savings = self._calculate_slab_based_savings(
                    monthly_energy, 
                    energy_data['tariff']['slabs']
                )
                savings = monthly_savings * 12
            
            return savings
            
        except Exception as e:
            st.error(f"Error calculating yearly savings: {str(e)}")
            return 0.0
    
    def _calculate_slab_based_savings(self, monthly_energy: float, 
                                    slabs: List[Dict]) -> float:
        """
        Calculate monthly savings for slab-based tariff
        """
        try:
            total_savings = 0
            remaining_energy = monthly_energy
            
            for i, slab in enumerate(slabs):
                if i == 0:
                    # First slab
                    energy_in_slab = min(remaining_energy, slab['units'])
                else:
                    # Higher slabs
                    energy_in_slab = min(
                        remaining_energy,
                        slab['units'] - slabs[i-1]['units']
                    )
                
                # Calculate savings for this slab
                total_savings += energy_in_slab * slab['rate']
                remaining_energy -= energy_in_slab
                
                if remaining_energy <= 0:
                    break
                
                # If we're at the last slab and still have remaining energy
                if i == len(slabs) - 1 and remaining_energy > 0:
                    total_savings += remaining_energy * slab['rate']
            
            return total_savings
            
        except Exception as e:
            st.error(f"Error calculating slab-based savings: {str(e)}")
            return 0.0
    
    def _calculate_monthly_savings(self, monthly_energy: float, 
                               energy_data: Dict) -> float:
        """
        Calculate savings for a specific month
        """
        if energy_data['tariff']['type'] == 'flat':
            return monthly_energy * energy_data['tariff']['rate']
        else:
            return self._calculate_slab_based_savings(
                monthly_energy,
                energy_data['tariff']['slabs']
            )
    
    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        """
        Calculate Net Present Value
        
        Parameters:
        -----------
        cash_flows : List[float]
            List of cash flows, starting with initial investment (negative)
        discount_rate : float
            Annual discount rate (e.g., 0.08 for 8%)
            
        Returns:
        --------
        float
            Net Present Value
        """
        try:
            npv = 0
            for year, cf in enumerate(cash_flows):
                npv += cf / (1 + discount_rate) ** year
            return npv
        except Exception as e:
            st.error(f"Error calculating NPV: {str(e)}")
            return 0.0
    
    # In financial_calculator.py, improve IRR calculation:
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return with multiple methods"""
        try:
            from scipy import optimize
            
            def npv_function(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            
            # Try multiple methods
            methods = ['newton', 'secant', 'bisect', 'brentq']
            for method in methods:
                try:
                    if method in ['newton', 'secant']:
                        irr = optimize.root_scalar(npv_function, method=method, x0=0.1).root
                        return irr * 100
                    else:
                        # Bisection methods need brackets
                        irr = optimize.root_scalar(npv_function, method=method, 
                                                bracket=[-0.999, 0.999]).root
                        return irr * 100
                except:
                    continue
                    
            # If all methods fail
            st.warning("Could not calculate IRR - project may not have a valid IRR")
            return 0.0
        except Exception as e:
            st.error(f"Error calculating IRR: {str(e)}")
            return 0.0
    
    def _calculate_roi(self, cash_flows: List[float], initial_investment: float) -> float:
        """
        Calculate Return on Investment
        """
        try:
            total_returns = sum(cf for cf in cash_flows[1:])  # Exclude initial investment
            roi = (total_returns / abs(initial_investment)) * 100
            return roi / len(cash_flows[1:])  # Annualized ROI
        except Exception as e:
            st.error(f"Error calculating ROI: {str(e)}")
            return 0.0
    
    def _calculate_payback_period(self, cash_flows: List[float]) -> float:
        """
        Calculate Simple Payback Period
        """
        try:
            initial_investment = abs(cash_flows[0])
            cumulative = 0
            
            for i, cf in enumerate(cash_flows[1:], 1):
                cumulative += cf
                if cumulative >= initial_investment:
                    # Interpolate for fractional year
                    fraction = (initial_investment - (cumulative - cf)) / cf
                    return i - 1 + fraction
                    
            # If payback is never reached
            return float('inf')
        except Exception as e:
            st.error(f"Error calculating payback period: {str(e)}")
            return float('inf')
    
    def _calculate_levelized_cost(self, total_energy: float, cash_flows: List[float], 
                                discount_rate: float) -> float:
        """
        Calculate Levelized Cost of Energy (LCOE)
        """
        try:
            # Calculate present value of all costs
            pv_costs = abs(cash_flows[0])  # Initial investment
            for year, cost in enumerate(cash_flows[1:], 1):
                if cost < 0:  # Only consider costs (negative cash flows)
                    pv_costs += abs(cost) / (1 + discount_rate) ** year
            
            # Calculate present value of energy
            pv_energy = 0
            yearly_energy = total_energy / (len(cash_flows) - 1)  # Excluding initial year
            degradation = 0.006  # 0.6% annual degradation
            
            for year in range(1, len(cash_flows)):
                year_energy = yearly_energy * (1 - degradation) ** (year - 1)
                pv_energy += year_energy / (1 + discount_rate) ** year
            
            # Calculate LCOE
            lcoe = pv_costs / pv_energy
            return lcoe
        except Exception as e:
            st.error(f"Error calculating LCOE: {str(e)}")
            return 0.0
    
    def _calculate_profitability_index(self, cash_flows: List[float], 
                                     discount_rate: float) -> float:
        """
        Calculate Profitability Index (PI)
        """
        try:
            initial_investment = abs(cash_flows[0])
            pv_future_cash_flows = sum(
                cf / (1 + discount_rate) ** i 
                for i, cf in enumerate(cash_flows[1:], 1)
            )
            return pv_future_cash_flows / initial_investment
        except Exception as e:
            st.error(f"Error calculating Profitability Index: {str(e)}")
            return 0.0
    
    
    def calculate_financial_metrics(self, energy_data: Dict, project_cost: float, 
                              om_params: Dict) -> Dict[str, Any]:
        """Calculate financial metrics considering system type"""
        try:
            # Constants
            ANALYSIS_PERIOD = 25  # years
            DISCOUNT_RATE = 0.08  # 8% discount rate
            ANNUAL_DEGRADATION = 0.006  # 0.6% annual degradation
            
            # Get yearly solar generation
            yearly_generation = st.session_state.results['energy']['metrics']['total_yearly']
            
            # Calculate initial yearly savings/revenue
            if energy_data["system_type"] == "Grid Export Only":
                initial_yearly_revenue = yearly_generation * energy_data['tariff']['rate']
                revenue_type = "Revenue"
            else:  # Captive Consumption
                initial_yearly_revenue = yearly_generation * energy_data['tariff']['rate']
                revenue_type = "Savings"
            
            # Calculate cash flows with degradation
            cash_flows = [-project_cost]  # Initial investment (negative)
            current_revenue = initial_yearly_revenue
            current_om_cost = om_params['yearly_om_cost']
            
            yearly_details = []  # Store yearly details for display
            
            for year in range(ANALYSIS_PERIOD):
                # Calculate degraded generation and revenue/savings
                degradation_factor = (1 - ANNUAL_DEGRADATION) ** year
                degraded_amount = current_revenue * degradation_factor
                
                # Calculate net cash flow
                net_cash_flow = degraded_amount - current_om_cost
                cash_flows.append(net_cash_flow)
                
                # Store yearly details
                yearly_detail = {
                    "year": year + 1,
                    "degradation_factor": degradation_factor,
                    "energy_output": yearly_generation * degradation_factor,
                    revenue_type.lower(): degraded_amount,  # Use the appropriate key
                    "om_cost": current_om_cost,
                    "net_cash_flow": net_cash_flow
                }
                yearly_details.append(yearly_detail)
                
                # Apply escalation rates for next year
                current_revenue *= (1 + om_params['tariff_escalation'])
                current_om_cost *= (1 + om_params['om_escalation'])
            
            # Calculate NPV
            npv = self._calculate_npv(cash_flows, DISCOUNT_RATE)
            
            # Calculate IRR
            irr = self._calculate_irr(cash_flows)
            
            # Calculate ROI (exclude initial investment from total returns)
            total_returns = sum(cash_flows[1:])  # All cash flows except initial investment
            roi = (total_returns / abs(project_cost)) * 100 / ANALYSIS_PERIOD  # Annualized ROI
            
            # Calculate payback period
            payback = self._calculate_payback_period(cash_flows)
            
            # Calculate cumulative metrics
            total_energy = sum(year_data['energy_output'] for year_data in yearly_details)
            total_revenue = sum(year_data[revenue_type.lower()] for year_data in yearly_details)
            total_om_cost = sum(year_data['om_cost'] for year_data in yearly_details)
            
            return {
                "npv": npv,
                "irr": irr,
                "roi": roi,
                "payback_period": payback,
                "yearly_details": yearly_details,
                "cash_flows": cash_flows,
                "system_type": energy_data["system_type"],
                "summary": {
                    "total_energy_25yr": total_energy,
                    "total_revenue_25yr": total_revenue,
                    "total_om_cost_25yr": total_om_cost,
                    "net_revenue_25yr": total_revenue - total_om_cost,
                    "revenue_type": revenue_type
                }
            }
            
        except Exception as e:
            st.error(f"Error in financial metrics calculation: {str(e)}")
            st.error(traceback.format_exc())
            return None

    def display_financial_metrics(self, metrics: Dict[str, Any]):
        """Display financial metrics with proper error handling"""
        st.subheader("Financial Analysis Results (25 Year Period)")
        
        # Get system type and appropriate terminology
        system_type = metrics.get('system_type', 'Captive Consumption')
        revenue_type = metrics['summary'].get('revenue_type', 'Savings')  # Default to 'Savings'
        currency_symbol = self.current_settings['currency_symbol']
        
        # Create grid layout for main metrics
        col1, col2, col3 = st.columns(3)
        
        # NPV Card
        with col1:
            st.markdown(f"""
                <div class="financial-card">
                    <div class="metric-icon">💰</div>
                    <div class="metric-title">NPV</div>
                    <div class="metric-value">
                        {currency_symbol}{metrics['npv']:,.2f}
                    </div>
                    <div class="metric-subtitle">Net Present Value</div>
                </div>
            """, unsafe_allow_html=True)
        
        # IRR Card
        with col2:
            st.markdown(f"""
                <div class="financial-card">
                    <div class="metric-icon">📈</div>
                    <div class="metric-title">IRR</div>
                    <div class="metric-value">
                        {metrics['irr']:.2f}%
                    </div>
                    <div class="metric-subtitle">Internal Rate of Return</div>
                </div>
            """, unsafe_allow_html=True)
        
        # ROI Card
        with col3:
            st.markdown(f"""
                <div class="financial-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-title">Annual Avg ROI</div>
                    <div class="metric-value">
                        {metrics['roi']:.2f}%
                    </div>
                    <div class="metric-subtitle">Return on Investment</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Payback Period Card with error handling
        payback_text = (
            "Project does not reach payback within 25 years"
            if math.isinf(metrics['payback_period']) or metrics['payback_period'] > 25
            else f"{int(metrics['payback_period'])} years and {int((metrics['payback_period'] % 1) * 12)} months"
        )
        
        st.markdown(f"""
            <div class="financial-card">
                <div class="metric-icon">⏳</div>
                <div class="metric-title">Simple Payback Period</div>
                <div class="metric-value">
                    {payback_text}
                </div>
                <div class="metric-subtitle">Time to recover initial investment</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 25-Year Summary
        st.subheader("25-Year Performance Summary")
        summary = metrics['summary']
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class="financial-card">
                    <div class="metric-icon">⚡</div>
                    <div class="metric-title">Total Energy Generation</div>
                    <div class="metric-value">
                        {summary['total_energy_25yr']:,.0f} kWh
                    </div>
                    <div class="metric-subtitle">Including 0.6% annual degradation</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="financial-card">
                    <div class="metric-icon">💵</div>
                    <div class="metric-title">Net {revenue_type}</div>
                    <div class="metric-value">
                        {currency_symbol}{summary['net_revenue_25yr']:,.0f}
                    </div>
                    <div class="metric-subtitle">Total {revenue_type.lower()} minus O&M costs</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Add warning if payback period is infinite
        if math.isinf(metrics['payback_period']) or metrics['payback_period'] > 25:
            warning_text = ("⚠️ Note: The project does not reach payback within the 25-year analysis period. "
                          "This could be due to:\n"
                          "- High initial costs\n"
                          "- Low energy production\n"
                          f"- Low electricity {('tariff' if system_type == 'Grid Export Only' else 'savings')}\n"
                          "- High O&M costs\n"
                          "Consider adjusting these parameters to improve financial viability.")
            st.warning(warning_text)
        
        # Show yearly details in an expander
        with st.expander("📊 View Year-by-Year Details"):
            yearly_df = pd.DataFrame(metrics['yearly_details'])
            yearly_df['Performance Ratio'] = yearly_df['degradation_factor'] * 100
            yearly_df['Energy Output (kWh)'] = yearly_df['energy_output']
            
            # Handle both revenue and savings cases
            if 'revenue' in yearly_df:
                yearly_df['Revenue'] = yearly_df['revenue']
                value_column = 'Revenue'
            elif 'savings' in yearly_df:
                yearly_df['Savings'] = yearly_df['savings']
                value_column = 'Savings'
            else:
                # If neither exists, create a default column
                value_column = revenue_type
                yearly_df[value_column] = yearly_df['net_cash_flow'] + yearly_df['om_cost']
            
            yearly_df['O&M Cost'] = yearly_df['om_cost']
            yearly_df['Net Cash Flow'] = yearly_df['net_cash_flow']
            
            display_columns = [
                'year', 
                'Performance Ratio', 
                'Energy Output (kWh)', 
                value_column,  # Use the determined value column
                'O&M Cost', 
                'Net Cash Flow'
            ]
            
            number_format = {
                'Performance Ratio': '{:.1f}%',
                'Energy Output (kWh)': '{:,.0f}',
                value_column: '{:,.0f}',
                'O&M Cost': '{:,.0f}',
                'Net Cash Flow': '{:,.0f}'
            }
            
            st.dataframe(
                yearly_df[display_columns].style.format(number_format)
            )
            
        # Add this to your display_financial_metrics method in financial_calculator.py

        # After the existing financial metrics display, add charts section
        st.markdown("---")
        st.subheader("📊 Financial Charts")
        
        # Create tabs for charts
        chart_tab1, chart_tab2 = st.tabs(["Payback Analysis", "Revenue vs Expenses"])
        
        with chart_tab1:
            # Create payback period chart
            try:
                # Extract data from financial results
                years = list(range(26))  # 0 to 25 years
                cash_flows = metrics['cash_flows']  # Use 'metrics' instead of 'financial_metrics'
                cumulative_cash_flow = [sum(cash_flows[:i+1]) for i in range(len(cash_flows))]
                
                # Get currency symbol from current settings
                currency_symbol = self.current_settings['currency_symbol']
                
                # Determine break-even point
                break_even_year = None
                break_even_value = None
                
                for i in range(1, len(cumulative_cash_flow)):
                    if cumulative_cash_flow[i-1] < 0 and cumulative_cash_flow[i] >= 0:
                        # Interpolate to find exact break-even point
                        prev_year = i - 1
                        prev_value = cumulative_cash_flow[prev_year]
                        curr_value = cumulative_cash_flow[i]
                        
                        # Linear interpolation
                        if curr_value - prev_value != 0:  # Avoid division by zero
                            fraction = -prev_value / (curr_value - prev_value)
                            break_even_year = prev_year + fraction
                            break_even_value = 0
                        else:
                            break_even_year = i
                            break_even_value = 0
                        break
                
                # Payback chart using Plotly
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add cumulative cash flow line
                fig.add_trace(go.Scatter(
                    x=years,
                    y=cumulative_cash_flow,
                    mode='lines+markers',
                    name='Cumulative Cash Flow',
                    line=dict(color='#2ca02c', width=3),
                    hovertemplate='Year %{x}: %{y:,.2f} ' + currency_symbol
                ))
                
                # Add break-even point
                if break_even_year is not None:
                    fig.add_trace(go.Scatter(
                        x=[break_even_year],
                        y=[break_even_value],
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='star'),
                        name=f'Break-even: {break_even_year:.2f} years',
                        hovertemplate='Break-even: %{x:.2f} years'
                    ))
                    
                    # Add vertical line at break-even point
                    fig.add_shape(
                        type="line",
                        x0=break_even_year,
                        y0=min(cumulative_cash_flow) * 1.1,
                        x1=break_even_year,
                        y1=max(cumulative_cash_flow) * 0.2,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=25,
                    y1=0,
                    line=dict(color="black", width=1.5),
                )
                
                # Update layout
                fig.update_layout(
                    title="Payback Period & Cumulative Cash Flow",
                    xaxis_title="Years",
                    yaxis_title=f"Cash Flow ({currency_symbol})",
                    hovermode="x unified",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                # Add annotations for key points
                if break_even_year is not None:
                    fig.add_annotation(
                        x=break_even_year,
                        y=max(cumulative_cash_flow) * 0.3,
                        text=f"Payback: {break_even_year:.2f} years",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        font=dict(size=14)
                    )
                
                # Show net profit at 25 years
                fig.add_annotation(
                    x=25,
                    y=cumulative_cash_flow[-1],
                    text=f"25-yr Net: {currency_symbol}{cumulative_cash_flow[-1]:,.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#2ca02c",
                    font=dict(size=14)
                )
                
                # Display chart
                
                col1, col2, col3 = st.columns([1, 8, 1])
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                with st.expander("ℹ️ Understanding the Payback Chart"):
                    st.markdown("""
                    This chart visualizes your investment's break-even point and cumulative returns:
                    
                    - **Green Line**: Cumulative cash flow over project lifetime
                    - **Red Star**: Break-even point (payback period)
                    - **Horizontal Black Line**: Zero line (break-even threshold)
                    
                    The payback period is when your cumulative cash flow crosses from negative to positive, 
                    representing the time it takes to recover your initial investment.
                    
                    After break-even, all additional cash flow represents profit on your investment.
                    """)
                    
            except Exception as e:
                st.error(f"Error creating payback chart: {str(e)}")
        
        with chart_tab2:
            # Create revenue vs expenses chart
            try:
                # Extract yearly data
                yearly_details = metrics['yearly_details']  # Use 'metrics' instead of 'financial_metrics'
                years = [detail["year"] for detail in yearly_details]
                
                # Get currency symbol from current settings
                currency_symbol = self.current_settings['currency_symbol']
                
                # Determine which field to use (revenue or savings)
                revenue_type = metrics['summary']['revenue_type'].lower()  # Use 'metrics' instead of 'financial_metrics'
                if revenue_type in yearly_details[0]:
                    revenues = [detail[revenue_type] for detail in yearly_details]
                else:
                    # Fallback calculation
                    revenues = [detail["net_cash_flow"] + detail["om_cost"] for detail in yearly_details]
                    
                expenses = [detail["om_cost"] for detail in yearly_details]
                net_cash_flows = [detail["net_cash_flow"] for detail in yearly_details]
                
                # Create stacked bar chart
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add bars for revenue
                fig.add_trace(go.Bar(
                    x=years,
                    y=revenues,
                    name=metrics['summary']['revenue_type'],  # Use 'metrics' instead of 'financial_metrics'
                    marker_color='#3366cc',
                    hovertemplate='Year %{x}: %{y:,.2f} ' + currency_symbol
                ))
                
                # Add bars for expenses
                fig.add_trace(go.Bar(
                    x=years,
                    y=expenses,
                    name='O&M Expenses',
                    marker_color='#ff9900',
                    hovertemplate='Year %{x}: %{y:,.2f} ' + currency_symbol
                ))
                
                # Add line for net cash flow
                fig.add_trace(go.Scatter(
                    x=years,
                    y=net_cash_flows,
                    mode='lines+markers',
                    name='Net Cash Flow',
                    line=dict(color='#2ca02c', width=3),
                    hovertemplate='Year %{x}: %{y:,.2f} ' + currency_symbol
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"25-Year {metrics['summary']['revenue_type']} vs Expenses",  # Use 'metrics' instead of 'financial_metrics'
                    xaxis_title="Year",
                    yaxis_title=f"Amount ({currency_symbol})",
                    barmode='group',
                    hovermode="x unified",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                # Display chart
                # With:
                col1, col2, col3 = st.columns([1, 8, 1])
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add summary metrics below the chart
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_revenue = sum(revenues)
                    st.metric(
                        f"Total {metrics['summary']['revenue_type']}",  # Use 'metrics' instead of 'financial_metrics'
                        f"{currency_symbol}{total_revenue:,.0f}",
                        help=f"25-year total {revenue_type}"
                    )
                    
                with col2:
                    total_expenses = sum(expenses)
                    st.metric(
                        "Total O&M Expenses",
                        f"{currency_symbol}{total_expenses:,.0f}",
                        help="25-year total operation & maintenance costs"
                    )
                    
                with col3:
                    expense_ratio = (total_expenses / total_revenue) * 100 if total_revenue > 0 else 0
                    st.metric(
                        "Expense Ratio",
                        f"{expense_ratio:.1f}%",
                        help="O&M expenses as percentage of total revenue"
                    )
                    
                # Add explanation
                with st.expander("ℹ️ Understanding the Revenue vs Expenses Chart"):
                    st.markdown(f"""
                    This chart compares your annual {revenue_type} against operation & maintenance expenses:
                    
                    - **Blue Bars**: Annual {metrics['summary']['revenue_type']} from your solar system
                    - **Yellow Bars**: Annual O&M expenses
                    - **Green Line**: Net cash flow (Revenue minus Expenses)
                    
                    Key observations:
                    - {metrics['summary']['revenue_type']} typically decreases slightly over time due to module degradation
                    - O&M expenses increase over time due to the escalation rate
                    - The difference between these values represents your annual financial benefit
                    """)
                    
            except Exception as e:
                st.error(f"Error creating revenue vs expenses chart: {str(e)}")