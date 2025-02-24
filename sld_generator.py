# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:30:12 2025
@author: amrit

Solar PV System - Professional Schematic Generator with SVG Icons
"""

import math
from typing import Dict, Any
from solar_icons import ICONS

class EnergyFlowSchematicGenerator:
    def __init__(self):
        self.width = 1200
        self.height = 600
        
    def generate_svg(self, system_params: Dict[str, Any]) -> str:
        """Generate SVG for professional schematic"""
        # Use provided values if available, otherwise calculate them
        dc_capacity = system_params.get('dc_capacity', 
                                       system_params.get('calculated_capacity', 0) * 
                                       system_params.get('effective_dc_ac_ratio', 1.0))
                                       
        ac_capacity = system_params.get('ac_capacity',
                                      system_params.get('total_ac_capacity',
                                                      system_params.get('calculated_capacity', 0)))
        
        svg = f'''
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
             viewBox="0 0 {self.width} {self.height}">
            <!-- Definitions -->
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
                
                .title {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    font-weight: 600;
                }}
                .label {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    font-weight: 400;
                }}
                .value {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    font-weight: 400;
                    font-size: 12px;
                }}
            </style>
            <defs>
                <!-- Component Symbols -->
                <symbol id="solar-panel" viewBox="0 0 512 512">
                    {ICONS['solar_panel']}
                </symbol>
                <symbol id="solar-inverter" viewBox="0 0 512 512">
                    {ICONS['solar_inverter']}
                </symbol>
                <symbol id="accb" viewBox="0 0 512 512">
                    {ICONS['accb']}
                </symbol>
                <symbol id="lv-panel" viewBox="0 0 512 512">
                    {ICONS['lv_panel']}
                </symbol>
                <symbol id="solar-meter" viewBox="0 0 512 512">
                    {ICONS['solar_meter']}
                </symbol>
                <symbol id="net-meter" viewBox="0 0 512 512">
                    {ICONS['net_meter']}
                </symbol>
                <symbol id="grid" viewBox="0 0 512 512">
                    {ICONS['grid']}
                </symbol>
                <symbol id="building" viewBox="0 0 512 512">
                    {ICONS['building']}
                </symbol>
                <symbol id="monitor" viewBox="0 0 512 512">
                    {ICONS['monitor']}
                </symbol>
            </defs>
            
            <!-- Title -->
          <text x="{self.width/2}" y="40" text-anchor="middle" font-size="24" class="title">
              Solar PV System - Energy Flow Schematic
          </text>
          
          <!-- Components -->
          <!-- PV Array with right-aligned labels -->
          <g transform="translate(250,50)">
              <use href="#solar-panel" width="80" height="80"/>
              <text x="100" y="30" text-anchor="start" class="label">PV Array</text>
              <text x="100" y="50" text-anchor="start" class="value">{dc_capacity:.1f} kWp</text>
          </g>
          
          <!-- Distribution Grid with left-aligned labels -->
          <g transform="translate(920,50)">
              <use href="#grid" width="120" height="80"/>
              <text x="-10" y="30" text-anchor="end" class="label">Distribution</text>
              <text x="-10" y="50" text-anchor="end" class="label">Grid</text>
          </g>
          
          <!-- Solar Inverter -->
          <g transform="translate(225,250)">
              <use href="#solar-inverter" width="120" height="80"/>
              <text x="60" y="100" text-anchor="middle" class="label">Solar</text>
              <text x="60" y="120" text-anchor="middle" class="label">Inverter</text>
              <text x="60" y="140" text-anchor="middle" class="value">{ac_capacity:.1f} kW</text>
          </g>
          
          <!-- Solar Meter -->
          <g transform="translate(410,250)">
              <use href="#solar-meter" width="80" height="80"/>
              <text x="40" y="100" text-anchor="middle" class="label">Solar</text>
              <text x="40" y="120" text-anchor="middle" class="label">Meter</text>
          </g>
          
          <!-- AC Combiner Box with label above -->
          <g transform="translate(570,250)">
              <text x="60" y="-10" text-anchor="middle" class="label">AC Combiner Box</text>
              <use href="#accb" width="120" height="80"/>
          </g>
          
          <!-- Main LV Panel with label above -->
          <g transform="translate(770,250)">
              <text x="60" y="-10" text-anchor="middle" class="label">Main LV Panel</text>
              <use href="#lv-panel" width="120" height="80"/>
          </g>
          
          <!-- Net Meter -->
          <g transform="translate(930,250)">
              <use href="#net-meter" width="80" height="80"/>
              <text x="40" y="100" text-anchor="middle" class="label">Net</text>
              <text x="40" y="120" text-anchor="middle" class="label">Meter</text>
          </g>
          
          <!-- Monitoring System -->
          <g transform="translate(570,450)">
              <use href="#monitor" width="120" height="80"/>
              <text x="60" y="100" text-anchor="middle" class="label">Monitoring</text>
              <text x="60" y="120" text-anchor="middle" class="label">System</text>
          </g>
          
          <!-- Premises Load -->
          <g transform="translate(770,450)">
              <use href="#building" width="120" height="80"/>
              <text x="60" y="100" text-anchor="middle" class="label">Premise</text>
              <text x="60" y="120" text-anchor="middle" class="label">Loads</text>
          </g>
            
            <!-- Connection Lines -->
            <g stroke-width="2">
                <!-- Solar Energy Flow Lines (Green) -->
                <line x1="290" y1="130" x2="290" y2="250" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                
                <!-- Grid-Net Meter Bidirectional Flow (Changed) -->
                <line x1="980" y1="130" x2="980" y2="230" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                <line x1="960" y1="230" x2="960" y2="130" stroke="#f44336" marker-end="url(#arrowRed)"/>
                
                <line x1="320" y1="290" x2="410" y2="290" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                <line x1="490" y1="290" x2="570" y2="290" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                <line x1="690" y1="290" x2="770" y2="290" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                <line x1="890" y1="290" x2="930" y2="290" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                
                <!-- Grid Return Lines -->
                <line x1="930" y1="310" x2="890" y2="310" stroke="#f44336" marker-end="url(#arrowRed)"/>
                <line x1="770" y1="310" x2="690" y2="310" stroke="#f44336" marker-end="url(#arrowRed)"/>
                
                <!-- Vertical Lines to Monitoring and Loads -->
                <line x1="630" y1="330" x2="630" y2="450" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                <line x1="830" y1="330" x2="830" y2="450" stroke="#4caf50" marker-end="url(#arrowGreen)"/>
                <line x1="645" y1="330" x2="645" y2="450" stroke="#f44336" marker-end="url(#arrowRed)"/>
                <line x1="845" y1="330" x2="845" y2="450" stroke="#f44336" marker-end="url(#arrowRed)"/>
            </g>
            
            <!-- Arrow Markers -->
            <defs>
                <marker id="arrowGreen" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4caf50"/>
                </marker>
                
                <marker id="arrowRed" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#f44336"/>
                </marker>
            </defs>
        </svg>
        '''
        
        return svg

    def get_system_details(self, system_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system configuration details"""
        self.height = 600
        # Use provided values if available, otherwise calculate them
        dc_capacity = system_params.get('dc_capacity', 
                                       system_params.get('calculated_capacity', 0) * 
                                       system_params.get('effective_dc_ac_ratio', 1.0))
                                       
        ac_capacity = system_params.get('ac_capacity',
                                      system_params.get('total_ac_capacity',
                                                      system_params.get('calculated_capacity', 0)))
        
        # Calculate DC/AC ratio from the provided capacities
        dc_ac_ratio = dc_capacity / ac_capacity if ac_capacity > 0 else 1.0
        
        # Get actual module wattage
        module_wattage = system_params.get('module_watt_peak')
        
        return {
        "capacities": {
            "dc_capacity": dc_capacity,
            "ac_capacity": ac_capacity,
            "dc_ac_ratio": dc_ac_ratio
        },
        "modules": {
            "total_count": system_params.get('total_modules', 0),
            "wattage": module_wattage
        }
    }