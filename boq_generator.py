# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:15:20 2025
@author: amrit

BOQ Generator - Generate detailed Bill of Quantities for solar PV systems
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import math
import io
from financial_data import CURRENCIES, REGION_COSTS

class BOQGenerator:
    def __init__(self):
        # BOQ categories
        self.categories = {
            "PV Modules": {
                "description": "Solar photovoltaic modules",
                "unit": "Nos",
                "items": []
            },
            "Inverters": {
                "description": "Solar inverters (string/central)",
                "unit": "Nos",
                "items": []
            },
            "Mounting Structure": {
                "description": "Module mounting structures",
                "unit": "Set",
                "items": []
            },
            "DC System": {
                "description": "DC cables, connectors and accessories",
                "unit": "Lot",
                "items": []
            },
            "AC System": {
                "description": "AC cables, distribution boards and accessories",
                "unit": "Lot",
                "items": []
            },
            "Safety Equipment": {
                "description": "Safety and protection devices",
                "unit": "Lot",
                "items": []
            },
            "Monitoring System": {
                "description": "Data monitoring and communication equipment",
                "unit": "Set",
                "items": []
            },
            "Civil Works": {
                "description": "Foundation and civil construction works",
                "unit": "Lot",
                "items": []
            },
            "Installation Services": {
                "description": "Labor and installation services",
                "unit": "Service",
                "items": []
            }
        }
        
        # Standard BOQ components with typical costs
        self.components = {
            "PV Modules": {
                "Standard Tier-1 Modules": {"cost_per_wp": 0.28, "unit": "Wp"},
                "High-Efficiency Modules": {"cost_per_wp": 0.35, "unit": "Wp"},
                "Bifacial Modules": {"cost_per_wp": 0.32, "unit": "Wp"}
            },
            "Inverters": {
                "String Inverters": {"cost_per_kw": 80, "unit": "kW"},
                "Central Inverters": {"cost_per_kw": 65, "unit": "kW"},
                "Microinverters": {"cost_per_kw": 110, "unit": "kW"}
            },
            "Mounting Structure": {
                "Rooftop Mounting System": {"cost_per_kw": 45, "unit": "kW"},
                "Ground-mounted System": {"cost_per_kw": 70, "unit": "kW"},
                "Carport Structure": {"cost_per_kw": 120, "unit": "kW"},
                "Tracking System": {"cost_per_kw": 150, "unit": "kW"}
            },
            "DC System": {
                "DC Cables (Solar)": {"cost_per_kw": 25, "unit": "kW"},
                "MC4 Connectors": {"cost_per_pair": 2, "unit": "Pair"},
                "DC Distribution Box": {"cost_per_unit": 300, "unit": "Nos"}
            },
            "AC System": {
                "AC Cables": {"cost_per_kw": 30, "unit": "kW"},
                "AC Distribution Panel": {"cost_per_unit": 500, "unit": "Nos"},
                "Step-up Transformer": {"cost_per_kva": 50, "unit": "kVA"}
            },
            "Safety Equipment": {
                "Surge Protection Device": {"cost_per_unit": 150, "unit": "Nos"},
                "Earthing Kit": {"cost_per_unit": 200, "unit": "Set"},
                "Lightning Arrestor": {"cost_per_unit": 250, "unit": "Nos"}
            },
            "Monitoring System": {
                "Basic Monitoring": {"cost_per_kw": 15, "unit": "kW"},
                "Advanced Monitoring": {"cost_per_kw": 25, "unit": "kW"},
                "Weather Station": {"cost_per_unit": 1200, "unit": "Set"}
            },
            "Civil Works": {
                "Basic Foundation Works": {"cost_per_kw": 15, "unit": "kW"},
                "Elevated Structure": {"cost_per_kw": 30, "unit": "kW"}
            },
            "Installation Services": {
                "Installation & Commissioning": {"cost_per_kw": 70, "unit": "kW"},
                "Testing & Inspection": {"cost_per_kw": 10, "unit": "kW"}
            }
        }
        
        # Define a color palette for different BOQ sections
        self.section_colors = {
            "PV Modules": "#E3F2FD",  # Light blue
            "Inverters": "#E8F5E9",    # Light green
            "Mounting Structure": "#FFF3E0",  # Light orange
            "DC System": "#F3E5F5",    # Light purple
            "AC System": "#E0F7FA",    # Light cyan
            "Safety Equipment": "#FFEBEE",  # Light red
            "Monitoring System": "#F1F8E9",  # Light mint
            "Civil Works": "#FFF8E1",   # Light amber
            "Installation Services": "#E8EAF6"  # Light indigo
        }
    
    def calculate_quantities(self, system_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate required quantities based on system parameters
        
        Parameters:
        -----------
        system_params : Dict[str, Any]
            System parameters including capacity, modules, inverters, etc.
            
        Returns:
        --------
        Dict[str, Any]
            Calculated quantities for BOQ
        """
        quantities = {}
        
        # Get key system parameters
        dc_capacity = system_params.get('plant_capacity_kw', 0) * 1000  # Convert to Wp
        
        # Get total modules from the results structure if available
        if hasattr(st.session_state, 'results') and st.session_state.results:
            total_modules = st.session_state.results['system']['total_modules']
        else:
            # Fallback calculation
            total_modules = math.ceil(dc_capacity / system_params.get('module_watt_peak', 400))
        
        module_wp = system_params.get('module_watt_peak', 400)
        
        # Get inverter details if available from session state
        inverter_params = st.session_state.get('inverter_params', {})
        inverter_config = inverter_params.get('configuration', {})
        num_inverters = inverter_config.get('num_inverters', 1)
        inverter_model = inverter_params.get('model', 'Generic Inverter')
        
        # Get inverter capacity - use specifications if available
        if 'specifications' in inverter_params:
            inverter_capacity = inverter_params['specifications'].get('nominal_ac_power', dc_capacity/1000)
        else:
            inverter_capacity = dc_capacity / 1000 / num_inverters  # Fallback calculation
        
        # PV Modules
        quantities["PV Modules"] = {
            "quantity": total_modules,
            "capacity_wp": dc_capacity,
            "model": f"{module_wp}W Module",
            "module_wp": module_wp
        }
        
        # Inverters
        quantities["Inverters"] = {
            "quantity": num_inverters,
            "capacity_kw": inverter_capacity * num_inverters,
            "model": inverter_model,
            "inverter_capacity": inverter_capacity  # Added individual inverter capacity
        }
        
        # Mounting Structure
        quantities["Mounting Structure"] = {
            "capacity_kw": dc_capacity / 1000,
            "type": "Rooftop" if dc_capacity < 100000 else "Ground-mounted"
        }
        
        # DC System
        # Estimate cable quantities based on system size
        string_size = inverter_config.get('optimal_modules_per_string', 20)
        if string_size <= 0:  # Avoid division by zero
            string_size = 20
        
        num_strings = inverter_config.get('total_strings', math.ceil(total_modules / string_size))
        
        quantities["DC System"] = {
            "capacity_kw": dc_capacity / 1000,
            "cable_length": num_strings * 30,  # Estimated 30m per string
            "mc4_pairs": total_modules + num_strings,
            "dc_boxes": math.ceil(num_strings / 10)  # 1 DC box per 10 strings
        }
        
        # AC System
        quantities["AC System"] = {
            "capacity_kw": dc_capacity / 1000,
            "ac_panels": math.ceil(num_inverters / 4),  # 1 panel per 4 inverters
            "transformer": "Yes" if dc_capacity >= 100000 else "No"
        }
        
        # Safety Equipment
        quantities["Safety Equipment"] = {
            "spd_sets": num_inverters + 1,  # 1 per inverter + 1 for AC side
            "earthing_kits": math.ceil(dc_capacity / 20000)  # 1 per 20kW
        }
        
        # Monitoring System
        quantities["Monitoring System"] = {
            "capacity_kw": dc_capacity / 1000,
            "type": "Advanced" if dc_capacity >= 50000 else "Basic"
        }
        
        # Civil Works
        quantities["Civil Works"] = {
            "capacity_kw": dc_capacity / 1000,
            "type": "Elevated" if quantities["Mounting Structure"]["type"] == "Ground-mounted" else "Basic"
        }
        
        # Installation Services
        quantities["Installation Services"] = {
            "capacity_kw": dc_capacity / 1000
        }
        
        return quantities
    
    def generate_boq(self, system_params: Dict[str, Any], currency: str) -> pd.DataFrame:
        """
        Generate BOQ based on system parameters
        
        Parameters:
        -----------
        system_params : Dict[str, Any]
            System parameters including capacity, modules, inverters, etc.
        currency : str
            Currency code (e.g., 'USD', 'INR')
            
        Returns:
        --------
        pd.DataFrame
            BOQ as DataFrame
        """
        currency_symbol = CURRENCIES.get(currency, {}).get('symbol', '$')
        quantities = self.calculate_quantities(system_params)
        
        boq_items = []
        
        # PV Modules
        module_type = "Standard Tier-1 Modules"
        if system_params.get('module_efficiency', 0) > 0.2:
            module_type = "High-Efficiency Modules"
        
        # Get module quantity from calculated quantities
        module_qty = quantities["PV Modules"]["quantity"]
        module_wp = quantities["PV Modules"]["module_wp"]
        
        # If module quantity is still 0, try to calculate from capacity
        if module_qty <= 0:
            module_qty = math.ceil(quantities["PV Modules"]["capacity_wp"] / module_wp)
        
        module_cost = self.components["PV Modules"][module_type]["cost_per_wp"] * module_wp
        module_total = module_qty * module_cost
        
        boq_items.append({
            "Category": "PV Modules",
            "Item": f"{module_wp}W {module_type.replace('Modules', '').strip()} Solar Module",
            "Description": f"{module_wp}W Solar PV Module",
            "Unit": "Nos",
            "Quantity": module_qty,
            "Unit Price": f"{currency_symbol}{module_cost:.2f}",
            "Total": f"{currency_symbol}{module_total:.2f}"
        })
        
        # Inverters
        inverter_params = st.session_state.get('inverter_params', {})
        inverter_config = inverter_params.get('configuration', {})
        inverter_model = inverter_params.get('model', 'String Inverter')
        inverter_qty = inverter_config.get('num_inverters', 1)
        
        # Get inverter capacity directly from quantities
        inverter_capacity = quantities["Inverters"]["inverter_capacity"]
        
        inverter_type = "String Inverters"
        if "Central" in inverter_model:
            inverter_type = "Central Inverters"
        
        inverter_cost = self.components["Inverters"][inverter_type]["cost_per_kw"] * inverter_capacity
        inverter_total = inverter_qty * inverter_cost
        
        boq_items.append({
            "Category": "Inverters",
            "Item": f"{inverter_model} - {inverter_capacity}kW",
            "Description": f"{inverter_capacity}kW Solar Inverter",
            "Unit": "Nos",
            "Quantity": inverter_qty,
            "Unit Price": f"{currency_symbol}{inverter_cost:.2f}",
            "Total": f"{currency_symbol}{inverter_total:.2f}"
        })
        
        # Mounting Structure
        mount_type = "Rooftop Mounting System"
        if quantities["Mounting Structure"]["type"] == "Ground-mounted":
            mount_type = "Ground-mounted System"
        
        mount_capacity = quantities["Mounting Structure"]["capacity_kw"]
        mount_cost = self.components["Mounting Structure"][mount_type]["cost_per_kw"] * mount_capacity
        mount_total = mount_capacity * mount_cost
        
        boq_items.append({
            "Category": "Mounting Structure",
            "Item": mount_type,
            "Description": f"Module mounting structure for {mount_capacity:.1f}kW system",
            "Unit": "Set",
            "Quantity": 1,
            "Unit Price": f"{currency_symbol}{mount_total:.2f}",
            "Total": f"{currency_symbol}{mount_total:.2f}"
        })
        
        # DC System
        dc_capacity = quantities["DC System"]["capacity_kw"]
        cable_length = quantities["DC System"]["cable_length"]
        mc4_pairs = quantities["DC System"]["mc4_pairs"]
        dc_boxes = quantities["DC System"]["dc_boxes"]
        
        # DC Cables
        dc_cable_cost = 2.5 * cable_length  # Estimated $2.5 per meter
        boq_items.append({
            "Category": "DC System",
            "Item": "Solar DC Cable",
            "Description": "6mm² Solar DC Cable (Red/Black)",
            "Unit": "m",
            "Quantity": cable_length,
            "Unit Price": f"{currency_symbol}2.50",
            "Total": f"{currency_symbol}{dc_cable_cost:.2f}"
        })
        
        # MC4 Connectors
        mc4_cost = self.components["DC System"]["MC4 Connectors"]["cost_per_pair"] * mc4_pairs
        boq_items.append({
            "Category": "DC System",
            "Item": "MC4 Connectors",
            "Description": "MC4 Compatible Solar Connectors (Pair)",
            "Unit": "Pair",
            "Quantity": mc4_pairs,
            "Unit Price": f"{currency_symbol}{self.components['DC System']['MC4 Connectors']['cost_per_pair']:.2f}",
            "Total": f"{currency_symbol}{mc4_cost:.2f}"
        })
        
        # DC Distribution Box
        dc_box_cost = self.components["DC System"]["DC Distribution Box"]["cost_per_unit"] * dc_boxes
        boq_items.append({
            "Category": "DC System",
            "Item": "DC Distribution Box",
            "Description": "Solar DC Distribution Box with Surge Protection",
            "Unit": "Nos",
            "Quantity": dc_boxes,
            "Unit Price": f"{currency_symbol}{self.components['DC System']['DC Distribution Box']['cost_per_unit']:.2f}",
            "Total": f"{currency_symbol}{dc_box_cost:.2f}"
        })
        
        # Add remaining items...
        # AC System
        ac_capacity = quantities["AC System"]["capacity_kw"]
        ac_cable_cost = self.components["AC System"]["AC Cables"]["cost_per_kw"] * ac_capacity
        boq_items.append({
            "Category": "AC System",
            "Item": "AC Cables",
            "Description": "Copper AC Cables for interconnection",
            "Unit": "Lot",
            "Quantity": 1,
            "Unit Price": f"{currency_symbol}{ac_cable_cost:.2f}",
            "Total": f"{currency_symbol}{ac_cable_cost:.2f}"
        })
        
        ac_panels = quantities["AC System"]["ac_panels"]
        ac_panel_cost = self.components["AC System"]["AC Distribution Panel"]["cost_per_unit"] * ac_panels
        boq_items.append({
            "Category": "AC System",
            "Item": "AC Distribution Panel",
            "Description": "Solar AC Distribution Panel with Protection",
            "Unit": "Nos",
            "Quantity": ac_panels,
            "Unit Price": f"{currency_symbol}{self.components['AC System']['AC Distribution Panel']['cost_per_unit']:.2f}",
            "Total": f"{currency_symbol}{ac_panel_cost:.2f}"
        })
        
        if quantities["AC System"]["transformer"] == "Yes":
            transformer_size = ac_capacity * 1.2  # 20% larger than AC capacity
            transformer_cost = self.components["AC System"]["Step-up Transformer"]["cost_per_kva"] * transformer_size
            boq_items.append({
                "Category": "AC System",
                "Item": "Step-up Transformer",
                "Description": f"{transformer_size:.0f}kVA Step-up Transformer",
                "Unit": "Nos",
                "Quantity": 1,
                "Unit Price": f"{currency_symbol}{transformer_cost:.2f}",
                "Total": f"{currency_symbol}{transformer_cost:.2f}"
            })
        
        # Safety Equipment
        spd_sets = quantities["Safety Equipment"]["spd_sets"]
        spd_cost = self.components["Safety Equipment"]["Surge Protection Device"]["cost_per_unit"] * spd_sets
        boq_items.append({
            "Category": "Safety Equipment",
            "Item": "Surge Protection Devices",
            "Description": "SPDs for DC and AC systems",
            "Unit": "Set",
            "Quantity": spd_sets,
            "Unit Price": f"{currency_symbol}{self.components['Safety Equipment']['Surge Protection Device']['cost_per_unit']:.2f}",
            "Total": f"{currency_symbol}{spd_cost:.2f}"
        })
        
        earthing_kits = quantities["Safety Equipment"]["earthing_kits"]
        earthing_cost = self.components["Safety Equipment"]["Earthing Kit"]["cost_per_unit"] * earthing_kits
        boq_items.append({
            "Category": "Safety Equipment",
            "Item": "Earthing Kit",
            "Description": "Earthing and Grounding System",
            "Unit": "Set",
            "Quantity": earthing_kits,
            "Unit Price": f"{currency_symbol}{self.components['Safety Equipment']['Earthing Kit']['cost_per_unit']:.2f}",
            "Total": f"{currency_symbol}{earthing_cost:.2f}"
        })
        
        # Monitoring System
        monitoring_capacity = quantities["Monitoring System"]["capacity_kw"]
        monitoring_type = "Basic Monitoring" if quantities["Monitoring System"]["type"] == "Basic" else "Advanced Monitoring"
        monitoring_cost = self.components["Monitoring System"][monitoring_type]["cost_per_kw"] * monitoring_capacity
        boq_items.append({
            "Category": "Monitoring System",
            "Item": monitoring_type,
            "Description": f"{monitoring_type} System for {monitoring_capacity:.1f}kW Plant",
            "Unit": "Set",
            "Quantity": 1,
            "Unit Price": f"{currency_symbol}{monitoring_cost:.2f}",
            "Total": f"{currency_symbol}{monitoring_cost:.2f}"
        })
        
        # Civil Works
        civil_capacity = quantities["Civil Works"]["capacity_kw"]
        civil_type = "Basic Foundation Works" if quantities["Civil Works"]["type"] == "Basic" else "Elevated Structure"
        civil_cost = self.components["Civil Works"][civil_type]["cost_per_kw"] * civil_capacity
        boq_items.append({
            "Category": "Civil Works",
            "Item": civil_type,
            "Description": f"Civil works for {civil_capacity:.1f}kW system",
            "Unit": "Lot",
            "Quantity": 1,
            "Unit Price": f"{currency_symbol}{civil_cost:.2f}",
            "Total": f"{currency_symbol}{civil_cost:.2f}"
        })
        
        # Installation Services
        install_capacity = quantities["Installation Services"]["capacity_kw"]
        install_cost = self.components["Installation Services"]["Installation & Commissioning"]["cost_per_kw"] * install_capacity
        boq_items.append({
            "Category": "Installation Services",
            "Item": "Installation & Commissioning",
            "Description": f"Complete installation and commissioning for {install_capacity:.1f}kW system",
            "Unit": "Service",
            "Quantity": 1,
            "Unit Price": f"{currency_symbol}{install_cost:.2f}",
            "Total": f"{currency_symbol}{install_cost:.2f}"
        })
        
        testing_cost = self.components["Installation Services"]["Testing & Inspection"]["cost_per_kw"] * install_capacity
        boq_items.append({
            "Category": "Installation Services",
            "Item": "Testing & Inspection",
            "Description": "Testing, inspection and handover",
            "Unit": "Service",
            "Quantity": 1,
            "Unit Price": f"{currency_symbol}{testing_cost:.2f}",
            "Total": f"{currency_symbol}{testing_cost:.2f}"
        })
        
        return pd.DataFrame(boq_items)
    
    def display_boq_generator(self, system_params: Dict[str, Any]) -> None:
        """
        Display BOQ generator interface in Streamlit
        
        Parameters:
        -----------
        system_params : Dict[str, Any]
            System parameters including capacity, modules, inverters, etc.
        """
        st.subheader("📋 Bill of Quantities (BOQ) Generator")
        
        # Get complete system data from session state
        full_system_params = system_params.copy()
        
        # Add inverter data from session state if available
        if hasattr(st.session_state, 'inverter_params') and st.session_state.inverter_params:
            full_system_params['inverter_params'] = st.session_state.inverter_params
        
        # Add results data from session state if available
        if hasattr(st.session_state, 'results') and st.session_state.results:
            full_system_params['results'] = st.session_state.results
        
        # System overview - get values directly from session state
        dc_capacity = system_params.get('plant_capacity_kw', 0)
        module_type = st.session_state.get('selected_module', 'Standard Module')
        inverter_model = st.session_state.get('inverter_params', {}).get('model', 'Generic Inverter')
        
        st.info(f"""
        **System Overview:**
        - System Capacity: {dc_capacity:.2f} kWp
        - Module Type: {module_type}
        - Inverter Model: {inverter_model}
        """)
        
        # BOQ Configuration options
        st.markdown("### BOQ Configuration")
        
        # Get currency settings
        default_currency = st.session_state.financial_settings.get('currency', 'USD')
        currency_options = list(CURRENCIES.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select currency
            currency = st.selectbox(
                "Select Currency:",
                options=currency_options,
                index=currency_options.index(default_currency) if default_currency in currency_options else 0,
                format_func=lambda x: f"{CURRENCIES[x]['name']} ({CURRENCIES[x]['symbol']})"
            )
        
        with col2:
            # Select project type
            project_type = st.selectbox(
                "Project Type:",
                options=["Standard Grid-connected", "Off-grid with Battery", "Hybrid System"],
                index=0
            )
        
        # Optional equipment
        st.markdown("#### Optional Equipment")
        
        optional_cols = st.columns(3)
        with optional_cols[0]:
            weather_station = st.checkbox("Weather Station", value=False)
        
        with optional_cols[1]:
            cctv_system = st.checkbox("CCTV System", value=False)
        
        with optional_cols[2]:
            spare_parts = st.checkbox("Spare Parts Kit", value=False)
        
        # Generate BOQ button
        if st.button("Generate BOQ", type="primary"):
            with st.spinner("Generating Bill of Quantities..."):
                # Generate the BOQ dataframe
                boq_df = self.generate_boq(full_system_params, currency)
                
                # Add optional equipment if selected
                currency_symbol = CURRENCIES.get(currency, {}).get('symbol', '$')
                
                if weather_station:
                    weather_cost = self.components["Monitoring System"]["Weather Station"]["cost_per_unit"]
                    boq_df = pd.concat([boq_df, pd.DataFrame([{
                        "Category": "Monitoring System",
                        "Item": "Weather Station",
                        "Description": "Complete weather monitoring station",
                        "Unit": "Set",
                        "Quantity": 1,
                        "Unit Price": f"{currency_symbol}{weather_cost:.2f}",
                        "Total": f"{currency_symbol}{weather_cost:.2f}"
                    }])], ignore_index=True)
                
                if cctv_system:
                    cctv_cost = 1500  # Estimated cost
                    boq_df = pd.concat([boq_df, pd.DataFrame([{
                        "Category": "Monitoring System",
                        "Item": "CCTV System",
                        "Description": "CCTV surveillance system for plant security",
                        "Unit": "Set",
                        "Quantity": 1,
                        "Unit Price": f"{currency_symbol}{cctv_cost:.2f}",
                        "Total": f"{currency_symbol}{cctv_cost:.2f}"
                    }])], ignore_index=True)
                
                if spare_parts:
                    # Calculate 2% of total system cost for spares
                    total_cost = sum([float(item.replace(currency_symbol, '')) for item in boq_df["Total"]])
                    spare_cost = total_cost * 0.02
                    boq_df = pd.concat([boq_df, pd.DataFrame([{
                        "Category": "Safety Equipment",
                        "Item": "Spare Parts Kit",
                        "Description": "Essential spare parts for maintenance",
                        "Unit": "Set",
                        "Quantity": 1,
                        "Unit Price": f"{currency_symbol}{spare_cost:.2f}",
                        "Total": f"{currency_symbol}{spare_cost:.2f}"
                    }])], ignore_index=True)
                
                # Display the BOQ
                st.markdown("### Generated Bill of Quantities")
                
                # Group by category
                categories = boq_df["Category"].unique()
                
                for category in categories:
                    bg_color = self.section_colors.get(category, "#F5F5F5")  # Default to light gray
                    
                    # Create container with background color
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <h3>{category}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        category_df = boq_df[boq_df["Category"] == category]
                        st.table(category_df[["Item", "Description", "Unit", "Quantity", "Unit Price", "Total"]])
                
                # Calculate total BOQ cost
                total_cost = sum([float(item.replace(currency_symbol, '')) for item in boq_df["Total"]])
                
                st.markdown("---")
                st.markdown(f"#### Total BOQ Cost: {currency_symbol}{total_cost:,.2f}")
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    # Create Excel download button
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Write BOQ sheet
                        boq_df.to_excel(writer, sheet_name='BOQ', index=False)
                        
                        # Get workbook and add formats
                        workbook = writer.book
                        header_format = workbook.add_format({
                            'bold': True,
                            'bg_color': '#4B8BBE',
                            'font_color': 'white',
                            'border': 1
                        })
                        
                        # Get worksheet
                        worksheet = writer.sheets['BOQ']
                        
                        # Format headers
                        for col_num, value in enumerate(boq_df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        
                        # Format categories with colors
                        for category in categories:
                            category_format = workbook.add_format({
                                'bg_color': self.section_colors.get(category, "#F5F5F5").replace("#", ""),
                                'border': 1
                            })
                            
                            # Apply formatting to category rows
                            for i, row in enumerate(boq_df.itertuples()):
                                if row.Category == category:
                                    worksheet.set_row(i + 1, None, category_format)
                        
                        # Add total row
                        row_num = len(boq_df) + 2
                        total_format = workbook.add_format({
                            'bold': True,
                            'bg_color': '#E0E0E0',
                            'num_format': f'"{currency_symbol}"#,##0.00',
                            'top': 2,
                            'bottom': 2
                        })
                        
                        
                        worksheet.write(row_num, 0, "Total BOQ Cost", total_format)
                        worksheet.write(row_num, 6, total_cost, total_format)
                        
                        # Adjust column widths
                        worksheet.set_column(0, 0, 20)  # Category
                        worksheet.set_column(1, 1, 25)  # Item
                        worksheet.set_column(2, 2, 40)  # Description
                        worksheet.set_column(3, 3, 10)  # Unit
                        worksheet.set_column(4, 4, 10)  # Quantity
                        worksheet.set_column(5, 5, 15)  # Unit Price
                        worksheet.set_column(6, 6, 15)  # Total
                    
                    st.download_button(
                        label="📥 Download Excel BOQ",
                        data=buffer.getvalue(),
                        file_name=f"Solar_BOQ_{dc_capacity:.1f}kW.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # Create PDF download button (placeholder - actual PDF generation would require additional libraries)
                    st.download_button(
                        label="📥 Download PDF BOQ",
                        data=boq_df.to_csv(index=False),
                        file_name=f"Solar_BOQ_{dc_capacity:.1f}kW.pdf",
                        mime="application/pdf",
                        disabled=True
                    )
                    st.caption("PDF generation is currently unavailable")