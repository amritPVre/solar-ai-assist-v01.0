# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:30:12 2025
@author: amrit

Solar Icons Handler - Loads and manages SVG icons for the schematic
"""

def load_svg_content(filename: str) -> str:
    """Load SVG content from file"""
    try:
        with open(f"{filename}.svg", 'r') as file:
            content = file.read()
            # Extract just the path data and other elements inside the SVG
            start_content = content.find('>') + 1
            end_content = content.find('</svg>')
            inner_content = content[start_content:end_content].strip()
            return inner_content
    except Exception as e:
        print(f"Error loading SVG {filename}: {str(e)}")
        return ""

def get_all_svg_icons() -> dict:
    """Load all SVG icons and return as dictionary"""
    svg_files = {
        'solar_panel': 'solar_panel',
        'solar_inverter': 'solar-inverter',
        'accb': 'accb',
        'lv_panel': 'lv-panel',
        'solar_meter': 'solar-meter',
        'net_meter': 'net-meter',
        'grid': 'grid',
        'building': 'building',
        'monitor': 'monitor'
    }
    
    icons = {}
    for key, filename in svg_files.items():
        content = load_svg_content(filename)
        if content:
            icons[key] = content
            print(f"Successfully loaded {filename}.svg")
        else:
            print(f"Warning: Could not load {filename}.svg")
    
    return icons

# Load all icons when module is imported
ICONS = get_all_svg_icons()