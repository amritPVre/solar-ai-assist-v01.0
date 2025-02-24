# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:29:19 2025
@author: amrit

Solar Expert AI - A streamlined solar system design and calculation tool
"""

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from calculator import calculate_solar
import traceback
import pytz
from timezonefinder import TimezoneFinder
import json
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_folium import folium_static
import folium
import io
import math
from financial_calculator import FinancialCalculator
from sld_generator import EnergyFlowSchematicGenerator
# Currency and Regional Data
from financial_data import CURRENCIES, REGION_COSTS
from inverter_database import (
    get_suitable_inverters,
    get_inverter_details,
    calculate_inverter_configuration,
    COMMON_INVERTERS, CENTRAL_INVERTERS
)
from area_mapper import SolarAreaMapper
from boq_generator import BOQGenerator




# Define common solar module specifications
COMMON_MODULES = {
    "Tier 1 - High Efficiency (400W)": {
        "watt_peak": 400,
        "efficiency": 0.21,
        "area": 1.87,
        "manufacturer": "Premium Brand",
        "technology": "Mono PERC"
    },
    "Tier 1 - Standard (375W)": {
        "watt_peak": 375,
        "efficiency": 0.195,
        "area": 1.92,
        "manufacturer": "Standard Brand",
        "technology": "Mono PERC"
    },
    "Tier 1 - Large Format (540W)": {
        "watt_peak": 540,
        "efficiency": 0.21,
        "area": 2.56,
        "manufacturer": "Premium Brand",
        "technology": "Mono PERC"
    },
    "Tier 2 - Standard (330W)": {
        "watt_peak": 330,
        "efficiency": 0.185,
        "area": 1.78,
        "manufacturer": "Economic Brand",
        "technology": "Polycrystalline"
    }
}

# Pydantic models for data validation
class LocationInfo(BaseModel):
    """Model for location information"""
    city: str = Field(description="City name")
    country: str = Field(description="Country name")

class SystemParameters(BaseModel):
    """Model for solar system parameters"""
    plant_capacity_kw: float = Field(description="System capacity in kilowatts")  # Changed from capacity_kw
    tilt: Optional[float] = Field(default=20.0, description="Panel tilt angle")
    azimuth: Optional[float] = Field(default=180.0, description="Panel azimuth angle")
    module_efficiency: Optional[float] = Field(default=0.21, description="Module efficiency")
    module_watt_peak: Optional[float] = Field(default=400.0, description="Module watt peak")
    module_area: Optional[float] = Field(default=1.8, description="Module area in square meters")
    performance_ratio: Optional[float] = Field(default=0.8, description="System performance ratio")

class UserQuery(BaseModel):
    """Model for user query classification"""
    query_type: str = Field(description="Type of query (general/system_design)")
    content: str = Field(description="Query content")

# Custom tools for LangChain
class LocationTool(BaseTool):
    """Tool for geocoding locations"""
    name = "location_finder"
    description = "Find coordinates and timezone for a location"

    def _run(self, location: str) -> Dict[str, Any]:
        geolocator = Nominatim(user_agent="solar_ai", timeout=10)
        try:
            location_data = geolocator.geocode(location, exactly_one=True)
            if location_data:
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lat=location_data.latitude, lng=location_data.longitude)
                if not timezone_str:
                    timezone_str = 'UTC'
                return {
                    "success": True,
                    "latitude": location_data.latitude,
                    "longitude": location_data.longitude,
                    "timezone": timezone_str
                }
            return {"success": False, "error": "Location not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _arun(self, location: str):
        raise NotImplementedError("Async not implemented")
        
        


class SolarCalculationTool(BaseTool):
    """Tool for solar calculations"""
    name = "solar_calculator"
    description = "Calculate solar energy production"

    def _run(self, params_json: str) -> Dict[str, Any]:
        try:
            # Parse parameters
            params = json.loads(params_json)
            
            # Define parameter mapping (internal name -> calculator name)
            param_mapping = {
                'plant_capacity_kw': 'plant_capacity_kw',  # Map to correct parameter name
                'surface_tilt': 'surface_tilt',
                'surface_azimuth': 'surface_azimuth',
                'module_efficiency': 'module_efficiency',
                'performance_ratio': 'performance_ratio',
                'module_area': 'module_area',
                'module_watt_peak': 'module_watt_peak',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'tz_str': 'tz_str'
            }
            
            # Create clean params dict with mapped parameter names
            clean_params = {}
            for internal_name, calculator_name in param_mapping.items():
                if internal_name not in params:
                    return {
                        "success": False,
                        "error": f"Missing required parameter: {internal_name}"
                    }
                clean_params[calculator_name] = params[internal_name]
            
            # Print debug information
            st.write("Debug - Parameters being sent to calculator:", clean_params)
            
            # Calculate with clean parameters
            results = calculate_solar(**clean_params)
            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Calculation error: {str(e)}\nTraceback: {traceback.format_exc()}"
            }

    def _arun(self, params_json: str):
        raise NotImplementedError("Async not implemented")

def calculate_capacity_from_area(area: float, module_specs: dict) -> float:
    """
    Calculate system capacity based on available area and module specifications
    
    Parameters:
    -----------
    area : float
        Available area in square meters
    module_specs : dict
        Dictionary containing module specifications
        
    Returns:
    --------
    float
        Calculated system capacity in kW
    """
    # Calculate number of modules that can fit in the area
    # Adding 15% area for spacing and access
    usable_area = area * 0.85
    modules_possible = int(usable_area / module_specs["module_area"])
    
    # Calculate total capacity
    plant_capacity_kw = (modules_possible * module_specs["module_watt_peak"]) / 1000
    
    return plant_capacity_kw

# Parameter validation helper
def validate_calculation_params(params: dict) -> bool:
    """Validate that all required parameters are present for calculation"""
    required_params = [
        'latitude', 'longitude', 'tz_str', 'surface_tilt', 'surface_azimuth',
        'module_efficiency', 'performance_ratio', 'plant_capacity_kw', 'module_area',
        'module_watt_peak'
    ]
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        st.error(f"Missing required parameters: {', '.join(missing_params)}")
        return False
    return True

def create_panel_3d_viz(tilt, azimuth):
    """
    Create 3D visualization of solar panel orientation
    
    Parameters:
    -----------
    tilt : float
        Panel tilt angle in degrees (0-90)
    azimuth : float
        Panel azimuth angle in degrees (0-360, 180 = South)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D visualization
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Convert angles to radians
    tilt_rad = np.radians(tilt)
    azimuth_rad = np.radians(azimuth)
    
    # Panel dimensions
    width = 2  # panel width
    height = 1  # panel height
    
    # Calculate panel corners (initially in x-y plane)
    corners = np.array([
        [-width/2, -height/2, 0],  # bottom left
        [width/2, -height/2, 0],   # bottom right
        [width/2, height/2, 0],    # top right
        [-width/2, height/2, 0],   # top left
        [-width/2, -height/2, 0]   # close the loop
    ])
    
    # Rotate panel based on tilt and azimuth
    # First tilt up from horizontal
    tilt_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad), np.cos(tilt_rad)]
    ])
    
    # Then rotate for azimuth
    azimuth_matrix = np.array([
        [np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
        [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
        [0, 0, 1]
    ])
    
    # Apply rotations
    corners = np.dot(corners, tilt_matrix)
    corners = np.dot(corners, azimuth_matrix)
    
    # Create the 3D visualization
    fig = go.Figure()
    
    # Add panel surface
    fig.add_trace(go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        color='blue',
        opacity=0.7,
        hoverinfo='skip'
    ))
    
    # Add panel outline
    fig.add_trace(go.Scatter3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='skip'
    ))
    
    # Add direction indicators
    arrow_length = 1.5
    
    # North arrow (red)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, arrow_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=3),
        text=['', 'N'],
        textposition='top center'
    ))
    
    # South arrow (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, -arrow_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='blue', width=3),
        text=['', 'S'],
        textposition='bottom center'
    ))
    
    # Add panel normal vector
    normal = np.array([0, 0, 1])
    normal = np.dot(normal, tilt_matrix)
    normal = np.dot(normal, azimuth_matrix)
    normal *= 0.5  # Shorter than direction arrows
    
    fig.add_trace(go.Scatter3d(
        x=[0, normal[0]], y=[0, normal[1]], z=[0, normal[2]],
        mode='lines',
        line=dict(color='green', width=3),
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Panel Orientation (Tilt: {tilt}°, Azimuth: {azimuth}°)",
            x=0.5,
            y=0.95
        ),
        scene=dict(
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            xaxis=dict(range=[-2, 2], showgrid=True),
            yaxis=dict(range=[-2, 2], showgrid=True),
            zaxis=dict(range=[-2, 2], showgrid=True),
            aspectmode='cube'
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    
    return fig


def create_sunpath_diagram(latitude, longitude, tz_str, tilt, azimuth):
    """
    Create interactive sunpath diagram showing sun positions and panel orientation
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from pvlib.solarposition import get_solarposition
    from datetime import datetime, timedelta
    
    # Create timestamps for the whole year
    today = datetime.now()
    start = datetime(today.year, 1, 1, 0, 0, 0)
    end = datetime(today.year + 1, 1, 1, 0, 0, 0)
    times = pd.date_range(start=start, end=end, freq='H', tz=tz_str)[:-1]
    
    # Calculate solar position
    solpos = get_solarposition(times, latitude, longitude)
    
    # Create the figure
    fig = go.Figure()
    
    # Add sun paths for solstices and equinoxes
    special_dates = [
        (f"{today.year}-06-21", "Summer Solstice", "red"),     # Summer Solstice
        (f"{today.year}-12-21", "Winter Solstice", "blue"),    # Winter Solstice
        (f"{today.year}-03-21", "Spring Equinox", "green"),    # Spring Equinox
        (f"{today.year}-09-21", "Autumn Equinox", "orange")    # Autumn Equinox
    ]
    
    for date, name, color in special_dates:
        # Get solar position for this day
        day_times = pd.date_range(date, periods=24, freq='H', tz=tz_str)
        day_solpos = get_solarposition(day_times, latitude, longitude)
        
        # Convert coordinates
        day_azimuth = (180 - day_solpos.azimuth)
        day_theta = np.radians(day_azimuth)
        day_r = day_solpos.zenith
        
        # Add path
        fig.add_trace(go.Scatterpolar(
            r=day_r,
            theta=np.degrees(day_theta),
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            showlegend=True
        ))
    
    # Add hour lines (more subtle)
    for hour in range(6, 19, 1):  # 6 AM to 6 PM
        hour_data = solpos[solpos.index.hour == hour]
        hour_azimuth = (180 - hour_data.azimuth)
        hour_theta = np.radians(hour_azimuth)
        hour_r = hour_data.zenith
        
        fig.add_trace(go.Scatterpolar(
            r=hour_r,
            theta=np.degrees(hour_theta),
            mode='lines',
            name=f"{hour:02d}:00",
            line=dict(color='rgba(100,100,100,0.2)', width=1),
            showlegend=False
        ))
    
    # Add altitude circles
    for altitude in range(0, 90, 15):
        zenith = 90 - altitude
        fig.add_trace(go.Scatterpolar(
            r=[zenith] * 360,
            theta=list(range(360)),
            mode='lines',
            name=f"{altitude}° Altitude",
            line=dict(color='rgba(100,100,100,0.2)', width=1),
            showlegend=False
        ))
    
    # Add cardinal directions using scatter points with text
    directions = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    fig.add_trace(go.Scatterpolar(
        r=[95] * len(directions),  # Just outside the 90° circle
        theta=angles,
        mode='text',
        text=directions,
        textfont=dict(size=14, color="black"),
        showlegend=False
    ))
    
    # Add panel orientation marker
    panel_zenith = 90 - tilt
    panel_azimuth = (180 - azimuth)
    
    fig.add_trace(go.Scatterpolar(
        r=[panel_zenith],
        theta=[panel_azimuth],
        mode='markers+text',
        name='Panel Orientation',
        marker=dict(size=15, color='red', symbol='star'),
        text=['Panel'],
        textposition='top center'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"☀️ Annual Sunpath Diagram<br><sup>Latitude: {latitude:.2f}°, Longitude: {longitude:.2f}°</sup>",
            x=0.5,
            y=0.95
        ),
        polar=dict(
            radialaxis=dict(
                range=[0, 90],
                tickmode='array',
                tickvals=[0, 15, 30, 45, 60, 75, 90],
                ticktext=['90°', '75°', '60°', '45°', '30°', '15°', '0°'],
                tickfont=dict(size=10),
                title='Solar Altitude'
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE'],
                direction='clockwise',
            ),
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.1
        ),
        height=700,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig


def create_energy_heatmap(energy_data):
    """
    Create heatmap visualization of energy production patterns
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from datetime import datetime

    try:
        # Get daily data
        daily_df = pd.DataFrame({
            'Date': energy_data['energy']['daily']['Date'],
            'Energy': energy_data['energy']['daily']['Daily Energy Production (kWh)']
        })
        
        # Convert Date column to datetime if it's not already
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        
        # Add month and hour components
        daily_df['Month'] = daily_df['Date'].dt.month
        
        # Create a typical daily profile based on solar hours
        hours = range(24)
        hourly_weights = np.array([
            0, 0, 0, 0, 0, 0,  # 00-06: No production
            0.02, 0.05, 0.08,  # 06-09: Morning ramp
            0.10, 0.12, 0.14,  # 09-12: Late morning
            0.15, 0.15, 0.14,  # 12-15: Peak hours
            0.12, 0.10, 0.08,  # 15-18: Afternoon
            0.05, 0.02, 0,     # 18-21: Evening ramp
            0, 0, 0            # 21-00: No production
        ])
        
        # Create the hour x month matrix for the heatmap
        heatmap_data = np.zeros((24, 12))
        
        # Fill the matrix with weighted values
        for month in range(1, 13):
            monthly_avg = daily_df[daily_df['Month'] == month]['Energy'].mean()
            for hour in range(24):
                heatmap_data[hour, month-1] = monthly_avg * hourly_weights[hour]
        
        # Create month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=months,
            y=[f"{hour:02d}:00" for hour in hours],
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate="Month: %{x}<br>Time: %{y}<br>Energy: %{z:.2f} kWh<extra></extra>",
            colorbar=dict(
                title="Energy Production (kWh)",
                titleside="right"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="⚡ Daily Energy Production Pattern",
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="Month",
                tickangle=0
            ),
            yaxis=dict(
                title="Hour of Day",
                autorange="reversed"  # To show morning hours at top
            ),
            height=700,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Find peak production time safely
        max_weight_index = np.argmax(hourly_weights)
        max_month_index = np.argmax(np.max(heatmap_data, axis=0))
        
        # Add peak production annotation only if we have valid indices
        if max_weight_index < 24 and max_month_index < 12:
            fig.add_annotation(
                x=months[max_month_index],
                y=f"{max_weight_index:02d}:00",
                text="Peak Production",
                showarrow=True,
                arrowhead=1,
                font=dict(size=12, color="white")
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        st.write("Debug - Error details:")
        st.write(f"Error type: {type(e).__name__}")
        st.write(f"Error message: {str(e)}")
        st.write("Data structure:")
        st.write(energy_data['energy'].keys())
        return None

def create_location_map(latitude, longitude, location_name, system_capacity):
    """
    Create interactive map showing system location and solar data
    
    Parameters:
    -----------
    latitude : float
        System latitude
    longitude : float
        System longitude
    location_name : str
        Name of the location
    system_capacity : float
        System capacity in kW
    """
    import folium
    from folium import plugins
    from streamlit_folium import folium_static
    import requests
    
    try:
        # Create base map centered on system location
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=10,
            tiles='cartodbpositron'  # Light theme map
        )
        
        # Add system location marker with popup
        folium.Marker(
            [latitude, longitude],
            popup=folium.Popup(
                f"""
                <b>Solar PV System</b><br>
                Location: {location_name}<br>
                Capacity: {system_capacity:.1f} kW<br>
                Coordinates: {latitude:.4f}°, {longitude:.4f}°
                """,
                max_width=300
            ),
            icon=folium.Icon(color='red', icon='info-sign'),
            tooltip="Click for system details"
        ).add_to(m)
        
        # Add circle around the system location
        folium.Circle(
            radius=1000,  # 1 km radius
            location=[latitude, longitude],
            popup='Approximate system area',
            color='red',
            fill=True,
            fillOpacity=0.2
        ).add_to(m)
        
        # Add minimap
        minimap = plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Add fullscreen button
        plugins.Fullscreen(
            position='topright',
            title='Expand map',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)
        
        # Add custom legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    border-radius: 5px;">
            <p style="margin: 0;"><b>Legend</b></p>
            <p style="margin: 0;">
                🔴 System Location<br>
                ⭕ 1km Radius
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None


# Initialize LangChain components
def init_langchain():
    """Initialize LangChain with separate chains for general queries and system design"""
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=st.secrets["OPENAI_KEY"]
    )

    # System design message - strictly for parameter extraction
    system_design_message = SystemMessage(content="""You are a solar system design assistant. Extract either system capacity OR available area from user input.

ALWAYS respond in this exact format for system queries:
<conversation>
[Your acknowledgment of the input parameters]
</conversation>
<json>
{
    "query_type": "system_design",
    "location_info": {
        "city": "EXTRACTED_CITY",
        "country": "EXTRACTED_COUNTRY"
    },
    "system_params": {
        "plant_capacity_kw": EXTRACTED_CAPACITY_OR_NULL,
        "available_area": EXTRACTED_AREA_OR_NULL,
        "tilt": EXTRACTED_TILT,
        "azimuth": 180,
        "performance_ratio": 0.8
    },
    "next_action": "calculate"
}
</json>

Example inputs and responses:
1. "I have 100 sq meters available on my roof in Mumbai"
<conversation>
I'll analyze a solar system for your 100 sq meter roof area in Mumbai.
</conversation>
<json>
{
    "query_type": "system_design",
    "location_info": {"city": "Mumbai", "country": "India"},
    "system_params": {
        "plant_capacity_kw": null,
        "available_area": 100,
        "tilt": 20,
        "azimuth": 180,
        "performance_ratio": 0.8
    },
    "next_action": "calculate"
}
</json>

2. "Want to install 50kW system in Delhi"
<conversation>
I'll analyze a 50kW solar system in Delhi.
</conversation>
<json>
{
    "query_type": "system_design",
    "location_info": {"city": "Delhi", "country": "India"},
    "system_params": {
        "plant_capacity_kw": 50,
        "available_area": null,
        "tilt": 20,
        "azimuth": 180,
        "performance_ratio": 0.8
    },
    "next_action": "calculate"
}
</json>

IMPORTANT RULES:
1. Extract EITHER plant_capacity_kw OR available_area (set the other to null)
2. For area, accept square meters, sq.m, sqm, m2
3. For capacity, accept kW, KW, kw, kilowatt
4. Use default values: tilt=20, azimuth=180, performance_ratio=0.8 if not specified
5. Request clarification if neither capacity nor area is clear""")

    # General query message - for solar energy questions
    general_query_message = SystemMessage(content="""You are a solar energy expert. Provide informative responses about solar energy concepts, technology, and industry practices.

Guidelines:
1. Provide accurate, up-to-date information
2. Use clear, accessible language
3. Include technical details when relevant
4. Cite common industry standards
5. DO NOT provide system design calculations""")

    # Initialize memories
    design_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    general_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create prompts
    design_prompt = ChatPromptTemplate.from_messages([
        system_design_message,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    general_prompt = ChatPromptTemplate.from_messages([
        general_query_message,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Create conversation chains
    design_chain = ConversationChain(
        memory=design_memory,
        prompt=design_prompt,
        llm=llm,
        verbose=True
    )

    general_chain = ConversationChain(
        memory=general_memory,
        prompt=general_prompt,
        llm=llm,
        verbose=True
    )

    return design_chain, general_chain

def init_tools():
    """Initialize tools"""
    return [
        LocationTool(),
        SolarCalculationTool()
    ]

def parse_ai_response(response: str) -> tuple:
    """Parse AI response into conversation and structured data"""
    try:
        # Extract conversation part
        conv_start = response.find("<conversation>") + len("<conversation>")
        conv_end = response.find("</conversation>")
        conversation = response[conv_start:conv_end].strip()

        # Extract JSON part
        json_start = response.find("<json>") + len("<json>")
        json_end = response.find("</json>")
        json_str = response[json_start:json_end].strip()
        
        # Return just conversation for general queries
        if json_start == -1 or json_end == -1:
            return response, None
            
        structured_data = json.loads(json_str)
        return conversation, structured_data
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return response, None

def determine_query_type(user_input: str) -> str:
    """Determine if the query is general or system design related"""
    # Keywords suggesting system design query
    design_keywords = ['kw', 'capacity', 'system', 'calculate', 'design', 'tilt', 
                      'azimuth', 'angle', 'roof', 'install', 'output']
    
    # Convert to lowercase for comparison
    input_lower = user_input.lower()
    
    # Check for design keywords
    if any(keyword in input_lower for keyword in design_keywords) and any(char.isdigit() for char in input_lower):
        return "system_design"
    return "general"

def process_user_input(user_input: str, design_chain, general_chain) -> tuple:
    try:
        # Determine query type and get response
        if determine_query_type(user_input) == "system_design":
            response = design_chain.predict(input=user_input)
            conversation, structured_data = parse_ai_response(response)
            
            if structured_data:
                # Get location data
                location_tool = st.session_state.tools[0]
                location = f"{structured_data['location_info']['city']}, {structured_data['location_info']['country']}"
                location_data = location_tool._run(location)
                
                if location_data["success"]:
                    # Initialize system parameters with location data
                    st.session_state.system_params = {
                        "latitude": location_data["latitude"],
                        "longitude": location_data["longitude"],
                        "tz_str": location_data["timezone"],
                        "location": location,
                        "surface_tilt": structured_data["system_params"].get("tilt", 20.0),
                        "surface_azimuth": structured_data["system_params"].get("azimuth", 180.0),
                        "performance_ratio": structured_data["system_params"].get("performance_ratio", 0.8)
                    }
                    
                    # Store any provided capacity or area in session state for later use
                    if structured_data["system_params"].get("plant_capacity_kw"):
                        st.session_state.initial_capacity = structured_data["system_params"]["plant_capacity_kw"]
                        system_desc = f"{structured_data['system_params']['plant_capacity_kw']}kW"
                    elif structured_data["system_params"].get("available_area"):
                        st.session_state.initial_area = structured_data["system_params"]["available_area"]
                        system_desc = f"{structured_data['system_params']['available_area']} sq meters"
                    else:
                        system_desc = "solar system"
                        
                    # Initialize mapping state
                    if 'mapping_completed' not in st.session_state:
                        st.session_state.mapping_completed = False
                    
                    # Move to area mapping stage
                    st.session_state.current_stage = "area_mapping"
                    
                    # Return appropriate message based on whether initial parameters were provided
                    if hasattr(st.session_state, 'initial_capacity') or hasattr(st.session_state, 'initial_area'):
                        return (f"I'll help you design a {system_desc} in {location}. "
                               "First, let's verify or adjust the installation area using the mapping tool."), structured_data
                    else:
                        return (f"I'll help you design a solar system in {location}. "
                               "First, let's map out your installation area."), structured_data
            
            return conversation, structured_data
        else:
            # Handle general query
            response = general_chain.predict(input=user_input)
            return response, None
            
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def show_area_mapping():
    """Display area mapping interface and handle user input"""
    try:
        # Initial header and info display
        st.subheader("📍 Installation Area Mapping")
        
        # Show initial values if provided - keeping this here for context
        if hasattr(st.session_state, 'initial_capacity'):
            st.info(f"💡 Initial capacity provided: {st.session_state.initial_capacity:.1f} kW")
        elif hasattr(st.session_state, 'initial_area'):
            st.info(f"💡 Initial area provided: {st.session_state.initial_area:.1f} m²")
        
        # Create mapping interface - this now handles all analysis and navigation options
        map_data = st.session_state.area_mapper.create_installation_map(
            st.session_state.system_params.get('location')
        )
                    
    except Exception as e:
        st.error(f"Error in area mapping: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide fallback option
        if st.button("Continue without mapping", key="btn_error_fallback"):
            if hasattr(st.session_state, 'initial_capacity'):
                st.session_state.system_params["plant_capacity_kw"] = st.session_state.initial_capacity
            st.session_state.mapping_completed = True
            st.session_state.current_stage = "select_module"
            st.rerun()


def show_module_selection():
    """Display module selection interface and handle calculations"""
    st.subheader("🔋 Solar Module Selection")
    
    # Check if we're working with area or capacity
    is_area_based = any(param in st.session_state.system_params for param in ["available_area", "final_usable_area"])
    
    # Display current system parameters
    if is_area_based:
        # Check if we're using the new GCR-based areas
        if "final_usable_area" in st.session_state.system_params:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Area", 
                    f"{st.session_state.system_params['total_area']:.1f} m²",
                    help="Total drawn area"
                )
            with col2:
                st.metric(
                    "Initial Usable", 
                    f"{st.session_state.system_params['initial_usable_area']:.1f} m²",
                    help="After 85% usability factor"
                )
            with col3:
                st.metric(
                    "Final Usable", 
                    f"{st.session_state.system_params['final_usable_area']:.1f} m²",
                    help="After GCR application"
                )
            
            # Show project type info
            st.info(f"""
            📊 Installation Configuration:
            - Project Type: {st.session_state.system_params.get('project_category', 'N/A')}
            - Installation: {st.session_state.system_params.get('installation_type', 'N/A')}
            - GCR: {st.session_state.system_params.get('gcr_value', 0)*100:.1f}%
            """)
        else:
            # Legacy display for old data format
            st.info(f"📐 Available area: {st.session_state.system_params['available_area']} sq meters")
    else:
        st.info(f"⚡ System capacity: {st.session_state.system_params['plant_capacity_kw']} kW")
    
    selection_method = st.radio(
        "Choose module selection method:",
        ["Select from common modules", "Enter custom specifications"]
    )
    
    if selection_method == "Select from common modules":
        st.info("Common modules with typical specifications:")
        
        selected_module = st.selectbox(
            "Select a module type:",
            options=list(COMMON_MODULES.keys())
        )
        
        if selected_module:
            module = COMMON_MODULES[selected_module]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Power Rating", f"{module['watt_peak']} W")
                st.metric("Efficiency", f"{module['efficiency']*100:.1f}%")
            with col2:
                st.metric("Module Area", f"{module['area']:.2f} m²")
                st.metric("Technology", module['technology'])
            
            # Add electrical specifications display
            st.divider()
            st.markdown("#### ⚡ Electrical Specifications")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vmp", f"{module.get('vmp', 40):.1f} V")
                st.metric("Voc", f"{module.get('voc', 50):.1f} V")
            with col2:
                st.metric("Imp", f"{module.get('imp', 10):.2f} A")
                st.metric("Isc", f"{module.get('isc', 12):.2f} A")
            with col3:
                st.metric("Temp. Coeff. Voc", f"{module.get('temp_coeff_voc', -0.29):.2f} %/°C")
            
            if st.button("Use this module"):
                try:
                    # Update module specifications
                    module_params = {
                        "module_watt_peak": module['watt_peak'],
                        "module_efficiency": module['efficiency'],
                        "module_area": module['area'],
                        "vmp": module.get('vmp', 40),
                        "imp": module.get('imp', 10),
                        "voc": module.get('voc', 50),
                        "isc": module.get('isc', 12),
                        "temp_coeff_voc": module.get('temp_coeff_voc', -0.29)
                    }
                    
                    # If area-based, calculate capacity
                    if is_area_based:
                        # Check if using new GCR-based area calculations
                        if "final_usable_area" in st.session_state.system_params:
                            # Use the final usable area directly (already accounts for both 85% and GCR)
                            final_usable_area = st.session_state.system_params["final_usable_area"]
                            
                            # Calculate modules directly from final usable area
                            modules_possible = int(final_usable_area / module['area'])
                            
                            # Calculate system capacity
                            calculated_capacity = (modules_possible * module['watt_peak']) / 1000
                            
                            st.session_state.system_params["plant_capacity_kw"] = calculated_capacity
                            
                            # Show capacity calculation details with GCR context
                            st.success(f"""
                            📊 System Size Calculation:
                            - Total Area: {st.session_state.system_params['total_area']:.1f} m²
                            - Project Type: {st.session_state.system_params.get('project_category', 'N/A')}
                            - Installation: {st.session_state.system_params.get('installation_type', 'N/A')}
                            - GCR Applied: {st.session_state.system_params.get('gcr_value', 0)*100:.1f}%
                            - Final Usable Area: {final_usable_area:.1f} m²
                            - Number of Modules: {modules_possible}
                            - Total System Capacity: {calculated_capacity:.2f} kW
                            """)
                        else:
                            # Legacy calculation for backward compatibility
                            available_area = st.session_state.system_params["available_area"]
                            usable_area = available_area * 0.85
                            modules_possible = int(usable_area / module['area'])
                            calculated_capacity = (modules_possible * module['watt_peak']) / 1000
                            
                            st.session_state.system_params["plant_capacity_kw"] = calculated_capacity
                            
                            # Show capacity calculation details
                            st.success(f"""
                            📊 System Size Calculation:
                            - Available Area: {available_area:.1f} m²
                            - Usable Area: {usable_area:.1f} m² (85% of available area)
                            - Number of Modules: {modules_possible}
                            - Total System Capacity: {calculated_capacity:.2f} kW
                            """)
                    
                    # Update system parameters
                    st.session_state.system_params.update(module_params)
                    
                    # Save selected module
                    st.session_state.selected_module = selected_module
                    st.session_state.module_selection_done = True
                    
                    # Move to inverter selection
                    st.session_state.current_stage = "select_inverter"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during module selection: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")
    
    else:
        # Custom module specifications section
        st.info("Enter your module specifications:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            watt_peak = st.number_input(
                "Module Power (W)",
                min_value=100,
                max_value=1000,
                value=400,
                step=5,
                help="Rated power of the module in Watts"
            )
            
            efficiency = st.number_input(
                "Module Efficiency (%)",
                min_value=10.0,
                max_value=30.0,
                value=20.0,
                step=0.5,
                help="Module efficiency percentage"
            ) / 100
            
            vmp = st.number_input(
                "Voltage at MPP (V)",
                min_value=20.0,
                max_value=100.0,
                value=40.0,
                step=0.1,
                help="Maximum power point voltage"
            )
            
            voc = st.number_input(
                "Open Circuit Voltage (V)",
                min_value=20.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                help="Open circuit voltage"
            )
        
        with col2:
            area = st.number_input(
                "Module Area (m²)",
                min_value=1.0,
                max_value=3.0,
                value=1.8,
                step=0.1,
                help="Surface area of the module"
            )
            
            technology = st.selectbox(
                "Technology",
                options=["Mono PERC", "Polycrystalline", "Thin Film", "Other"],
                help="Module technology type"
            )
            
            imp = st.number_input(
                "Current at MPP (A)",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                help="Maximum power point current"
            )
            
            isc = st.number_input(
                "Short Circuit Current (A)",
                min_value=1.0,
                max_value=20.0,
                value=12.0,
                step=0.1,
                help="Short circuit current"
            )
        
        # Temperature coefficient
        temp_coeff_voc = st.number_input(
            "Temperature Coefficient of Voc (%/°C)",
            min_value=-1.0,
            max_value=0.0,
            value=-0.29,
            step=0.01,
            help="Temperature coefficient of open circuit voltage"
        )
        
        if st.button("Use custom specifications", type="primary"):
            try:
                # Update module specifications
                module_params = {
                    "module_watt_peak": watt_peak,
                    "module_efficiency": efficiency,
                    "module_area": area,
                    "vmp": vmp,
                    "imp": imp,
                    "voc": voc,
                    "isc": isc,
                    "temp_coeff_voc": temp_coeff_voc
                }
                
                # If area-based, calculate capacity
                if is_area_based:
                    # Check if using new GCR-based area calculations
                    if "final_usable_area" in st.session_state.system_params:
                        # Use the final usable area directly (already accounts for both 85% and GCR)
                        final_usable_area = st.session_state.system_params["final_usable_area"]
                        
                        # Calculate modules directly from final usable area
                        modules_possible = int(final_usable_area / area)
                        
                        # Calculate system capacity
                        calculated_capacity = (modules_possible * watt_peak) / 1000
                        
                        st.session_state.system_params["plant_capacity_kw"] = calculated_capacity
                        
                        # Show capacity calculation details with GCR context
                        st.success(f"""
                        📊 System Size Calculation:
                        - Total Area: {st.session_state.system_params['total_area']:.1f} m²
                        - Project Type: {st.session_state.system_params.get('project_category', 'N/A')}
                        - Installation: {st.session_state.system_params.get('installation_type', 'N/A')}
                        - GCR Applied: {st.session_state.system_params.get('gcr_value', 0)*100:.1f}%
                        - Final Usable Area: {final_usable_area:.1f} m²
                        - Number of Modules: {modules_possible}
                        - Total System Capacity: {calculated_capacity:.2f} kW
                        """)
                    else:
                        # Legacy calculation for backward compatibility
                        available_area = st.session_state.system_params["available_area"]
                        usable_area = available_area * 0.85
                        modules_possible = int(usable_area / area)
                        calculated_capacity = (modules_possible * watt_peak) / 1000
                        
                        st.session_state.system_params["plant_capacity_kw"] = calculated_capacity
                        
                        # Show capacity calculation details
                        st.success(f"""
                        📊 System Size Calculation:
                        - Available Area: {available_area:.1f} m²
                        - Usable Area: {usable_area:.1f} m² (85% of available area)
                        - Number of Modules: {modules_possible}
                        - Total System Capacity: {calculated_capacity:.2f} kW
                        """)
                
                # Update system parameters
                st.session_state.system_params.update(module_params)
                
                # Create custom module entry
                custom_module = {
                    "watt_peak": watt_peak,
                    "efficiency": efficiency,
                    "area": area,
                    "technology": technology,
                    "vmp": vmp,
                    "imp": imp,
                    "voc": voc,
                    "isc": isc,
                    "temp_coeff_voc": temp_coeff_voc
                }
                
                # Save selected module
                st.session_state.selected_module = "Custom Module"
                st.session_state.custom_module = custom_module
                st.session_state.module_selection_done = True
                
                # Move to inverter selection
                st.session_state.current_stage = "select_inverter"
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during module selection: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")

        with st.expander("Need help with specifications?"):
            st.markdown("""
            🔍 **Typical ranges for solar modules:**
            * **Power Rating**: 
                - Residential: 300W - 450W
                - Commercial: 400W - 600W
            * **Efficiency**:
                - Standard: 15% - 17%
                - High Efficiency: 18% - 22%
            * **Module Area**:
                - 60-cell: ~1.6-1.8 m²
                - 72-cell: ~2.0-2.2 m²
                - Large Format: ~2.4-2.8 m²
            * **Electrical Parameters**:
                - Vmp: 30V - 45V
                - Imp: 8A - 12A
                - Voc: 35V - 55V
                - Isc: 9A - 13A
                - Temp. Coeff.: -0.25%/°C to -0.35%/°C
            """)
    

def show_inverter_selection():
    """Display inverter selection interface and handle calculations"""
    st.subheader("🔌 Inverter Selection")
    
    # Get system capacity
    dc_capacity = st.session_state.system_params['plant_capacity_kw']
    
    # Display current system parameters
    st.info(f"📊 System Size: {dc_capacity:.1f} kW DC")
    
    # Determine if system is utility scale
    is_utility_scale = dc_capacity >= 1000
    
    # Manufacturer selection
    manufacturer = st.radio(
        "Select inverter manufacturer:",
        ["Sungrow", "Huawei"],
        help="Choose your preferred inverter manufacturer"
    )
    
    # Get suitable inverters
    suitable_inverters = get_suitable_inverters(
        dc_capacity=dc_capacity,
        manufacturer=manufacturer
    )
    
    if suitable_inverters:
        st.markdown("### Available Configurations")
        
        # Separate string and central inverter options
        string_options = []
        central_options = []
        inverter_configs = {}  # Store full configuration details
        
        for category, configs in suitable_inverters.items():
            for config in configs:
                display_text = (f"{config['num_inverters']}x {config['model']} "
                              f"(Total AC: {config['total_ac']:.1f}kW, "
                              f"DC/AC Ratio: {config['dc_ac_ratio']:.2f})")
                
                if "Central" in category:
                    central_options.append(display_text)
                else:
                    string_options.append(display_text)
                    
                inverter_configs[display_text] = config
        
        # Show appropriate options based on system size
        if is_utility_scale:
            st.subheader("Central Inverter Options")
            if central_options:
                selected_option = st.selectbox(
                    "Select central inverter configuration:",
                    options=central_options,
                    help="Choose a central inverter configuration suitable for your utility-scale system"
                )
            else:
                st.warning("No central inverters available for this size. Consider multiple string inverters.")
                selected_option = None
        else:
            st.subheader("String Inverter Options")
            if string_options:
                selected_option = st.selectbox(
                    "Select string inverter configuration:",
                    options=string_options,
                    help="Choose a string inverter configuration suitable for your system"
                )
            else:
                st.warning("No string inverters available for this size. Consider central inverters.")
                selected_option = None
        
        if selected_option:
            selected_config = inverter_configs[selected_option]
            inverter_specs = get_inverter_details(selected_config['model'])
            
            # Display inverter specifications
            st.markdown("#### Inverter Specifications")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nominal AC Power", f"{inverter_specs['nominal_ac_power']} kW")
                st.metric("Maximum Efficiency", f"{inverter_specs['max_efficiency']*100:.1f}%")
            
            with col2:
                st.metric("Maximum DC Power", f"{inverter_specs['max_dc_power']} kW")
                st.metric("Euro Efficiency", f"{inverter_specs['euro_efficiency']*100:.1f}%")
            
            with col3:
                st.metric("Max DC Voltage", f"{inverter_specs['max_dc_voltage']} V")
                st.metric("Number of MPPTs", str(inverter_specs['number_of_mppt']))
            
            # Get module specifications
            module_specs = COMMON_MODULES[st.session_state.selected_module]
            
            # Calculate string configuration
            config = calculate_inverter_configuration(
                dc_capacity=dc_capacity,
                inverter_model=selected_config['model'],
                module_vmp=module_specs.get('vmp', 40),
                module_imp=module_specs.get('imp', 10),
                module_voc=module_specs.get('voc', 50),
                module_isc=module_specs.get('isc', 12)
            )
            
            if config:
                st.divider()
                st.subheader("⚡ System Configuration")
                
                # Display configuration details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Inverters", str(selected_config['num_inverters']))
                    st.metric("DC/AC Ratio", f"{selected_config['dc_ac_ratio']:.2f}")
                
                with col2:
                    st.metric("Modules per String", str(config['optimal_modules_per_string']))
                    st.metric("Total Strings", str(config['total_strings']))
                
                with col3:
                    st.metric("Strings per Inverter", str(config['strings_per_inverter']))
                    st.metric("Strings per MPPT", str(config['strings_per_mppt']))
                
                # Show detailed configuration
                with st.expander("🔍 View Detailed Configuration"):
                    st.markdown(f"""
                    #### Voltage Specifications
                    - MPPT Voltage Range: {inverter_specs['mppt_range'][0]}V - {inverter_specs['mppt_range'][1]}V
                    - Maximum DC Voltage: {inverter_specs['max_dc_voltage']}V
                    - Rated MPPT Voltage: {inverter_specs['rated_mppt_voltage']}V
                    
                    #### Current Specifications
                    - Maximum Input Current: {inverter_specs['max_input_current']}A per MPPT
                    - Number of MPPTs: {inverter_specs['number_of_mppt']}
                    
                    #### String Details
                    - String Voltage (STC): {module_specs.get('vmp', 40) * config['optimal_modules_per_string']:.1f}V
                    - String Current: {module_specs.get('imp', 10):.1f}A
                    
                    #### Inverter Type
                    - Category: {config['category']}
                    - Type: {config['inverter_type'].title()}
                    """)
                
                # Add warnings about DC/AC ratio
                if selected_config['dc_ac_ratio'] > 1.15:
                    st.warning("⚠️ Higher DC/AC ratio may lead to some clipping during peak production.")
                elif selected_config['dc_ac_ratio'] < 1.0:
                    st.warning("⚠️ Low DC/AC ratio may result in inverter underutilization.")
                
                # Proceed button
                if st.button("Use this configuration", type="primary"):
                    try:
                        # Save inverter parameters
                        st.session_state.inverter_params = {
                            "model": selected_config['model'],
                            "manufacturer": manufacturer,
                            "specifications": inverter_specs,
                            "configuration": {
                                **config,
                                "num_inverters": selected_config['num_inverters'],
                                "dc_ac_ratio": selected_config['dc_ac_ratio']
                            }
                        }
                        
                        # Update system parameters
                        st.session_state.system_params.update({
                            "inverter_model": selected_config['model'],
                            "num_inverters": selected_config['num_inverters'],
                            "dc_ac_ratio": selected_config['dc_ac_ratio'],
                            "inverter_efficiency": inverter_specs['max_efficiency']
                        })
                        
                        st.session_state.inverter_selection_done = True
                        st.session_state.current_stage = "display_results"
                        
                        # Calculate results
                        calc_tool = st.session_state.tools[1]
                        results = calc_tool._run(json.dumps(st.session_state.system_params))
                        
                        if results["success"]:
                            st.session_state.results = results["results"]
                            st.success("✅ Inverter configuration saved and calculations completed!")
                            st.rerun()
                        else:
                            st.error(f"Calculation failed: {results.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"Error saving inverter configuration: {str(e)}")
                        st.error(f"Traceback: {traceback.format_exc()}")
    else:
        st.warning("""
        No standard inverter configurations found for this system size. Options:
        1. Consider splitting into multiple sub-arrays
        2. Contact manufacturer for custom solutions
        3. Explore other inverter options
        """)
        
        # Add option to proceed without inverter selection
        if st.button("Continue without inverter selection", key="skip_inverter"):
            st.session_state.inverter_selection_done = True
            st.session_state.current_stage = "display_results"
            st.rerun()
    

def show_3d_orientation():
    """Display 3D panel orientation in the results tab"""
    st.subheader("📐 Panel Orientation")
    
    # Get parameters from session state
    tilt = st.session_state.system_params.get('surface_tilt', 20)
    azimuth = st.session_state.system_params.get('surface_azimuth', 180)
    
    # Create and display the visualization
    fig = create_panel_3d_viz(tilt, azimuth)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("ℹ️ Understanding Panel Orientation"):
        st.markdown("""
        This 3D visualization shows the orientation of your solar panels:
        
        * **Tilt Angle**: The angle between the panel surface and horizontal ground
        * **Azimuth Angle**: The direction the panels face (180° = True South)
        * **Colored Arrows**:
            * 🔴 Red Arrow: Points North
            * 🔵 Blue Arrow: Points South
            * 🟢 Green Arrow: Panel normal (perpendicular to panel surface)
            
        You can:
        * Rotate the view by dragging
        * Zoom with the scroll wheel
        * Double-click to reset the view
        """)
        
def show_sunpath_analysis():
    """Display sunpath diagram in the results tab"""
    st.subheader("☀️ Solar Path Analysis")
    
    # Get parameters from session state
    params = st.session_state.system_params
    
    # Create and display the visualization
    fig = create_sunpath_diagram(
        latitude=params['latitude'],
        longitude=params['longitude'],
        tz_str=params['tz_str'],
        tilt=params['surface_tilt'],
        azimuth=params['surface_azimuth']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("ℹ️ Understanding the Sunpath Diagram"):
        st.markdown("""
        The sunpath diagram shows the sun's position throughout the year:
        
        * **Colored Paths**:
            * 🔴 Red: Summer Solstice (longest day)
            * 🔵 Blue: Winter Solstice (shortest day)
            * 🟢 Green: Spring Equinox
            * 🟡 Orange: Autumn Equinox
            
        * **Circles and Lines**:
            * Concentric circles: Solar altitude angles
            * Radial lines: Solar azimuth angles
            * Light gray lines: Hour markers
            
        * **⭐ Red Star**: Shows your panel orientation
        
        * **Reading the Diagram**:
            * Center: Directly overhead (90° altitude)
            * Outer edge: Horizon (0° altitude)
            * Cardinal directions marked on edge (S, SW, W, etc.)
        
        The diagram helps visualize optimal panel orientation and seasonal variations in solar access.
        """)
        
def show_hourly_analysis():
    """Display energy production pattern analysis"""
    st.subheader("⚡ Production Pattern Analysis")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Daily Pattern", "Monthly Analysis"])
    
    with tab1:
        try:
            # Create and display heatmap
            fig = create_energy_heatmap(st.session_state.results)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                with st.expander("ℹ️ Understanding the Production Pattern"):
                    st.markdown("""
                    This heatmap shows the typical energy production patterns:
                    
                    * **X-axis**: Months of the year
                    * **Y-axis**: Hours of the day (24-hour format)
                    * **Colors**: Energy production levels
                        * Darker colors = Higher production
                        * Lighter colors = Lower production
                    
                    Key Production Periods:
                    * **Peak Hours** (10:00-14:00): Maximum production
                    * **Morning Ramp** (06:00-10:00): Production increases
                    * **Afternoon Decline** (14:00-18:00): Production decreases
                    * **Low/No Production** (18:00-06:00): Minimal or no production
                    
                    Seasonal Variations:
                    * Summer months show longer production hours
                    * Winter months show shorter but still significant production
                    * Production peaks are higher in summer months
                    
                    You can:
                    * Hover over any point to see estimated values
                    * Click and drag to zoom
                    * Double-click to reset the view
                    """)
        
        except Exception as e:
            st.error(f"Error displaying pattern analysis: {str(e)}")
    
    with tab2:
        try:
            # Create monthly analysis
            monthly_data = pd.DataFrame({
                'Month': st.session_state.results['energy']['monthly']['Month'],
                'Energy': st.session_state.results['energy']['monthly']['Monthly Energy Production (kWh)']
            })
            
            # Calculate statistics
            total_yearly = st.session_state.results['energy']['metrics']['total_yearly']
            avg_daily = total_yearly / 365
            peak_month = monthly_data.loc[monthly_data['Energy'].idxmax()]
            low_month = monthly_data.loc[monthly_data['Energy'].idxmin()]
            
            # Display statistics
            st.markdown("#### 📊 Production Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Average Daily Production",
                    f"{avg_daily:.1f} kWh/day",
                    help="Average energy produced per day across the year"
                )
                st.metric(
                    "Peak Month",
                    f"{peak_month['Month']}",
                    f"{peak_month['Energy']:.0f} kWh"
                )
            
            with col2:
                st.metric(
                    "Average Monthly Production",
                    f"{total_yearly/12:.0f} kWh/month",
                    help="Average energy produced per month"
                )
                st.metric(
                    "Lowest Month",
                    f"{low_month['Month']}",
                    f"{low_month['Energy']:.0f} kWh"
                )
            
            # Show seasonal variation
            variation = (peak_month['Energy'] - low_month['Energy']) / (total_yearly/12) * 100
            st.metric(
                "Seasonal Variation",
                f"{variation:.1f}%",
                help="Percentage variation between highest and lowest producing months"
            )
            
        except Exception as e:
            st.error(f"Error displaying monthly analysis: {str(e)}")


def show_location_analysis():
    """Display location analysis with interactive map"""
    st.subheader("📍 Location Analysis")
    
    # Get system parameters
    params = st.session_state.system_params
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create and display map
        map_obj = create_location_map(
            latitude=params['latitude'],
            longitude=params['longitude'],
            location_name=params['location'],
            system_capacity=params['plant_capacity_kw']
        )
        
        if map_obj:
            folium_static(map_obj, width=700, height=500)
            
            with st.expander("ℹ️ Map Features"):
                st.markdown("""
                This interactive map shows:
                * 📍 Exact system location
                * ⭕ 1km radius around the site
                * 🗺️ Detailed terrain and satellite imagery
                
                You can:
                * Click markers for more information
                * Switch between map views
                * Zoom in/out for better detail
                * Expand to fullscreen
                """)
    
    with col2:
        st.markdown("#### 📊 Location Details")
        
        # Create metrics for location details
        st.metric(
            "Latitude",
            f"{params['latitude']:.4f}°",
            help="North-South position"
        )
        
        st.metric(
            "Longitude",
            f"{params['longitude']:.4f}°",
            help="East-West position"
        )
        
        st.metric(
            "Time Zone",
            params['tz_str'],
            help="Local timezone"
        )
        
        # Add elevation if available (placeholder for now)
        st.metric(
            "Elevation",
            "~100m",  # This could be fetched from an elevation API
            help="Height above sea level"
        )

def show_system_sld():
    """Display energy flow schematic diagram"""
    st.subheader("⚡ System Energy Flow Diagram")
    
    try:
        # Initialize schematic generator
        schematic_gen = EnergyFlowSchematicGenerator()
        
        # Create a comprehensive system_params dictionary with all needed values
        system_params = st.session_state.results['system'].copy()
        
        # Add key parameters from system_params
        if 'module_watt_peak' in st.session_state.system_params:
            system_params['module_watt_peak'] = st.session_state.system_params['module_watt_peak']
            
        # Add inverter configuration details
        if st.session_state.inverter_params:
            inverter_config = st.session_state.inverter_params['configuration']
            inverter_specs = st.session_state.inverter_params['specifications']
            
            # Add key inverter parameters
            system_params.update({
                'num_inverters': inverter_config['num_inverters'],
                'inverter_model': st.session_state.inverter_params['model'],
                'inverter_capacity': inverter_specs['nominal_ac_power'],
                'total_ac_capacity': inverter_config['num_inverters'] * inverter_specs['nominal_ac_power'],
                'effective_dc_ac_ratio': inverter_config['dc_ac_ratio'],
                'dc_capacity': system_params['calculated_capacity'] * inverter_config['dc_ac_ratio'],
                'ac_capacity': inverter_config['num_inverters'] * inverter_specs['nominal_ac_power']
            })
            
        # Generate schematic with enhanced parameters
        schematic_svg = schematic_gen.generate_svg(system_params)
        
        # Get system details with enhanced parameters
        system_details = schematic_gen.get_system_details(system_params)
        
        # Create HTML wrapper for SVG with proper styling
        html_content = f'''
            <div style="width: 100%; display: flex; justify-content: center; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="width: 1200px; max-width: 100%;">
                    {schematic_svg}
                </div>
            </div>
        '''
        
        # Display schematic
        st.components.v1.html(html_content, height=1000)
        
        # Display system specifications
        st.markdown("### System Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### System Capacities")
            capacities = system_details['capacities']
            st.metric("DC Capacity", f"{capacities['dc_capacity']:.1f} kWp")
            st.metric("AC Capacity", f"{capacities['ac_capacity']:.1f} kW")
            st.metric("DC/AC Ratio", f"{capacities['dc_ac_ratio']:.3f}")
        
        with col2:
            st.markdown("#### PV Array Details")
            modules = system_details['modules']
            st.metric("Total Modules", f"{modules['total_count']}")
            st.metric("Module Rating", f"{modules['wattage']} Wp")
            
        # Add download button
        st.download_button(
            label="📥 Download Schematic",
            data=schematic_svg,
            file_name=f"solar_system_schematic_{st.session_state.results['system']['calculated_capacity']:.1f}kW.svg",
            mime="image/svg+xml"
        )
            
    except Exception as e:
        st.error(f"Error generating schematic: {str(e)}")
        st.error(traceback.format_exc())



def show_financial_analysis():
    """Display financial analysis interface with sequential tabs"""
    try:
        # Get location from system parameters
        location = st.session_state.system_params.get('location', '')
        country = location.split(',')[-1].strip() if location else None
        
        # Initialize financial settings if not already done
        if not st.session_state.financial_settings:
            calculator = st.session_state.financial_calculator
            financial_settings = calculator.initialize_financial_settings(default_country=country)
            st.session_state.financial_settings = financial_settings
         
            
        # Add currency selector in sidebar
        with st.sidebar:
            st.markdown("### 💱 Currency Settings")
            current_currency = st.session_state.financial_settings['currency']
            
            if st.checkbox("Change Currency", value=False):
                new_currency = st.selectbox(
                    "Select currency:",
                    list(CURRENCIES.keys()),
                    index=list(CURRENCIES.keys()).index(current_currency),
                    format_func=lambda x: f"{CURRENCIES[x]['name']} ({CURRENCIES[x]['symbol']})"
                )
                
                if new_currency != current_currency:
                    # Update currency and convert values
                    st.session_state.financial_calculator.update_currency(
                        new_currency,
                        st.session_state.financial_inputs
                    )
                    st.rerun()
                    
                    
        # Initialize tab completion states if not exists
        if 'tab_states' not in st.session_state:
            st.session_state.tab_states = {
                'project_cost_done': False,
                'om_settings_done': False,
                'electricity_done': False
            }
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "💰 Project Cost",
            "⚙️ O&M Settings" + (" 🔒" if not st.session_state.tab_states['project_cost_done'] else ""),
            "⚡ Electricity Data" + (" 🔒" if not st.session_state.tab_states['om_settings_done'] else "")
        ])
        
        # Tab 1: Project Cost
        with tab1:
            st.subheader("Project Cost Settings")
            
            if st.session_state.financial_inputs["project_cost"] is None:
                project_cost_data = st.session_state.financial_calculator.get_project_cost_input(
                    st.session_state.system_params['plant_capacity_kw']
                )
                st.session_state.financial_inputs["project_cost"] = project_cost_data
            else:
                # Use the same layout for updates
                project_cost_data = st.session_state.financial_inputs["project_cost"]
                
                # System capacity display
                st.metric(
                    "System Capacity",
                    f"{st.session_state.system_params['plant_capacity_kw']:.1f} kW",
                    help="Total system capacity"
                )
                
                st.markdown("#### Enter Project Cost")
                # Per kW cost input
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Base Cost per kW:** {project_cost_data['currency_symbol']}{project_cost_data['cost_per_kw_local']:,.0f}")
                    new_cost_per_kw = st.number_input(
                        f"Cost per kW ({project_cost_data['currency_symbol']}/kW):",
                        min_value=0.0,
                        value=float(project_cost_data['cost_local'] / st.session_state.system_params['plant_capacity_kw']),
                        step=100.0,
                        format="%.0f",
                        help="Enter your expected or quoted cost per kW",
                        key="cost_per_kw_input"
                    )
                
                # Calculate and display total cost
                new_total_cost = new_cost_per_kw * st.session_state.system_params['plant_capacity_kw']
                with col2:
                    st.metric(
                        "Total Project Cost",
                        f"{project_cost_data['currency_symbol']}{new_total_cost:,.0f}",
                        help="Total project cost based on per kW cost"
                    )
                
                # Add comparison with base cost
                base_total = project_cost_data['cost_per_kw_local'] * st.session_state.system_params['plant_capacity_kw']
                cost_difference = ((new_total_cost - base_total) / base_total) * 100
                
                # Show cost comparison
                st.markdown("#### Cost Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Market Reference Cost",
                        f"{project_cost_data['currency_symbol']}{base_total:,.0f}",
                        help="Typical market cost for your region"
                    )
                with col2:
                    st.metric(
                        "Cost Difference",
                        f"{cost_difference:+.1f}%",
                        help="Difference from market reference cost",
                        delta_color="inverse"
                    )
                
                # Save/Update buttons
                if not st.session_state.tab_states['project_cost_done']:
                    if st.button("Save Project Cost and Continue", key="save_project_cost"):
                        project_cost_data['cost_local'] = new_total_cost
                        project_cost_data['cost_per_kw_actual'] = new_cost_per_kw
                        st.session_state.financial_inputs["project_cost"] = project_cost_data
                        st.session_state.tab_states['project_cost_done'] = True
                        st.rerun()
                else:
                    if st.button("Update Project Cost", key="update_project_cost"):
                        project_cost_data['cost_local'] = new_total_cost
                        project_cost_data['cost_per_kw_actual'] = new_cost_per_kw
                        st.session_state.financial_inputs["project_cost"] = project_cost_data
                        st.success("✅ Project cost updated")
        
        # Tab 2: O&M Settings
        with tab2:
            if st.session_state.tab_states['project_cost_done']:
                st.subheader("Operation & Maintenance Settings")
                
                if st.session_state.financial_inputs["om_params"] is None:
                    om_params = st.session_state.financial_calculator.get_om_parameters(
                        st.session_state.financial_inputs["project_cost"]['cost_local']
                    )
                    st.session_state.financial_inputs["om_params"] = om_params
                
                # Get current data and currency settings
                om_params = st.session_state.financial_inputs["om_params"]
                project_cost_data = st.session_state.financial_inputs["project_cost"]
                current_currency_symbol = st.session_state.financial_settings['currency_symbol']
                
                # Project cost reference
                st.info(f"Project Cost: {current_currency_symbol}{project_cost_data['cost_local']:,.0f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    om_percent = st.number_input(
                        "Yearly O&M cost (% of project cost):",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(om_params['yearly_om_cost'] / project_cost_data['cost_local'] * 100),
                        step=0.1,
                        help="Annual O&M cost as percentage of project cost",
                        key="om_percent_input"  # Added unique key
                    )
                    new_yearly_cost = project_cost_data['cost_local'] * (om_percent / 100)
                
                with col2:
                    st.metric(
                        "Yearly O&M Cost",
                        f"{current_currency_symbol}{new_yearly_cost:,.0f}",
                        help="Annual operation and maintenance cost"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    om_escalation = st.number_input(
                        "Yearly O&M cost escalation (%):",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(om_params['om_escalation'] * 100),
                        step=0.5,
                        help="Annual increase in O&M costs",
                        key="om_escalation_input"  # Added unique key
                    )
                
                with col2:
                    tariff_escalation = st.number_input(
                        "Yearly tariff escalation (%):",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(om_params['tariff_escalation'] * 100),
                        step=0.5,
                        help="Annual increase in electricity tariff",
                        key="tariff_escalation_input"  # Added unique key
                    )
                
                # Display projected costs
                if st.checkbox("Show future cost projections", key="show_projections_checkbox"):  # Added unique key
                    years = range(1, 6)
                    projection_data = []
                    current_om = new_yearly_cost
                    
                    for year in years:
                        current_om *= (1 + om_escalation/100)
                        projection_data.append({
                            "Year": year,
                            "Projected Cost": f"{current_currency_symbol}{current_om:,.0f}"
                        })
                    
                    st.markdown("##### 5-Year O&M Cost Projection")
                    st.table(pd.DataFrame(projection_data))
                
                if not st.session_state.tab_states['om_settings_done']:
                    if st.button("Save O&M Settings and Continue", key="save_om_settings"):  # Added unique key
                        om_params.update({
                            'yearly_om_cost': new_yearly_cost,
                            'om_escalation': om_escalation / 100,
                            'tariff_escalation': tariff_escalation / 100
                        })
                        st.session_state.financial_inputs["om_params"] = om_params
                        st.session_state.tab_states['om_settings_done'] = True
                        st.rerun()
                else:
                    if st.button("Update O&M Settings", key="update_om_settings"):  # Added unique key
                        om_params.update({
                            'yearly_om_cost': new_yearly_cost,
                            'om_escalation': om_escalation / 100,
                            'tariff_escalation': tariff_escalation / 100
                        })
                        st.session_state.financial_inputs["om_params"] = om_params
                        st.success("✅ O&M settings updated")
            else:
                st.warning("⚠️ Please complete Project Cost settings first")
        
        # Tab 3: Electricity Data
        # Inside show_financial_analysis()
        # Electricity Data Tab (tab3)
        # Tab 3: Electricity Data
        # Tab 3: Electricity Data
        with tab3:
            if st.session_state.tab_states['om_settings_done']:
                electricity_data = st.session_state.financial_calculator.show_electricity_data()
                
                if electricity_data:
                    # Save/Update Electricity Data
                    if not st.session_state.tab_states['electricity_done']:
                        if st.button("Save Electricity Data", key="save_electricity_data"):
                            st.session_state.financial_inputs["electricity_data"] = electricity_data
                            st.session_state.tab_states['electricity_done'] = True
                            st.success("✅ Electricity data saved")
                            st.rerun()
                    else:
                        if st.button("Update Electricity Data", key="update_electricity_data"):
                            st.session_state.financial_inputs["electricity_data"] = electricity_data
                            st.success("✅ Electricity data updated")
                
                # Add a divider
                st.divider()
                
                # Show calculate button when all sections are complete
                # Note: Moved this outside the electricity_data check
                if (st.session_state.tab_states['project_cost_done'] and 
                    st.session_state.tab_states['om_settings_done'] and 
                    st.session_state.tab_states['electricity_done']):
                    
                    if 'calculation_done' not in st.session_state:
                        st.session_state.calculation_done = False

                    if not st.session_state.calculation_done:
                        # In tab3, just before calculating financial metrics:
                        if st.button("Calculate Financial Metrics", type="primary", key="calculate_metrics"):
                            try:
                                project_cost_data = st.session_state.financial_inputs["project_cost"]
                                om_params = st.session_state.financial_inputs["om_params"]
                                electricity_data = st.session_state.financial_inputs["electricity_data"]
                                
                                # Debug prints
                                st.write("Debug - Input Data:")
                                st.write("Project Cost Data:", project_cost_data)
                                st.write("O&M Params:", om_params)
                                st.write("Electricity Data:", electricity_data)
                                
                                financial_metrics = st.session_state.financial_calculator.calculate_financial_metrics(
                                    electricity_data,
                                    project_cost_data['cost_local'],
                                    om_params
                                )
                                
                                # Debug print
                                st.write("Debug - Calculated Metrics:", financial_metrics)
                                
                                if financial_metrics:
                                    st.session_state.financial_results = financial_metrics
                                    st.session_state.calculation_done = True
                                    st.rerun()
                                else:
                                    st.error("No financial metrics were calculated")
                                    
                            except Exception as e:
                                st.error(f"Error calculating financial metrics: {str(e)}")
                                st.error(traceback.format_exc())
                                
                    # Add after O&M settings
                    #st.write("Debug - Session State:")
                    #st.write("Calculation Done:", st.session_state.get('calculation_done'))
                    #st.write("Financial Results:", st.session_state.get('financial_results'))

                    # Results display section
                    if st.session_state.calculation_done and st.session_state.financial_results:
                        st.markdown("### Financial Analysis Results")
                        st.markdown("---")
                        st.session_state.financial_calculator.display_financial_metrics(
                            st.session_state.financial_results
                        )
                        
                        # Add buttons in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Recalculate with Current Values", key="recalculate_metrics"):
                                st.session_state.calculation_done = False
                                st.rerun()
                        
                        with col2:
                            if st.button("Download Financial Report", key="download_report"):
                                generate_financial_report(st.session_state.financial_results)
            else:
                st.warning("⚠️ Please complete O&M Settings first")
                    
    except Exception as e:
        st.error(f"Error in financial analysis: {str(e)}")
        st.error(traceback.format_exc())

def generate_financial_report(financial_results):
    """Generate and download detailed financial report"""
    try:
        # Get current currency settings
        current_currency = st.session_state.financial_settings['currency']
        current_currency_symbol = st.session_state.financial_settings['currency_symbol']
        
        # Get initial project cost from session state
        project_cost = st.session_state.financial_inputs["project_cost"]["cost_local"]
        
        # Get revenue type
        revenue_type = financial_results['summary']['revenue_type']  # "Revenue" or "Savings"
        
        # Summary data
        report_data = {
            "System Details": {
                "Capacity": f"{st.session_state.system_params['plant_capacity_kw']} kW",
                "Location": st.session_state.system_params['location'],
                "Currency": f"{CURRENCIES[current_currency]['name']} ({current_currency_symbol})"
            },
            "Project Costs": {
                "Total Project Cost": f"{current_currency_symbol}{project_cost:,.2f}",
                "Cost per kW": f"{current_currency_symbol}{project_cost/st.session_state.system_params['plant_capacity_kw']:,.2f}"
            },
            "Financial Metrics": {
                "NPV": f"{current_currency_symbol}{financial_results['npv']:,.2f}",
                "IRR": f"{financial_results['irr']:.2f}%",
                "ROI": f"{financial_results['roi']:.2f}%",
                "Payback Period": (
                    "Project does not reach payback within 25 years"
                    if math.isinf(financial_results['payback_period']) or financial_results['payback_period'] > 25
                    else f"{int(financial_results['payback_period'])} years and {int((financial_results['payback_period'] % 1) * 12)} months"
                )
            },
            "25-Year Summary": {
                "Total Energy": f"{financial_results['summary']['total_energy_25yr']:,.0f} kWh",
                f"Total {revenue_type}": f"{current_currency_symbol}{financial_results['summary']['total_revenue_25yr']:,.0f}",
                "Total O&M Cost": f"{current_currency_symbol}{financial_results['summary']['total_om_cost_25yr']:,.0f}",
                f"Net {revenue_type}": f"{current_currency_symbol}{financial_results['summary']['net_revenue_25yr']:,.0f}"
            }
        }
        
        # Convert summary to DataFrame
        df = pd.DataFrame([
            [k1, k2, v] for k1, d in report_data.items() for k2, v in d.items()
        ], columns=['Category', 'Metric', 'Value'])
        
        # Create Excel buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write Summary sheet
            df.to_excel(writer, sheet_name='Financial Summary', index=False)
            
            # Process yearly details
            yearly_details = financial_results['yearly_details']
            yearly_df = pd.DataFrame(yearly_details)
            
            # Create final yearly details DataFrame with proper columns
            final_yearly_df = pd.DataFrame({
                'Year': range(1, len(yearly_details) + 1),
                'Performance Ratio (%)': [detail['degradation_factor'] * 100 for detail in yearly_details],
                'Energy Output (kWh)': [detail['energy_output'] for detail in yearly_details],
                f'{revenue_type} ({current_currency_symbol})': [detail[revenue_type.lower()] for detail in yearly_details],
                f'O&M Cost ({current_currency_symbol})': [detail['om_cost'] for detail in yearly_details],
                f'Net Cash Flow ({current_currency_symbol})': [detail['net_cash_flow'] for detail in yearly_details]
            })
            
            # Write yearly details to Excel
            final_yearly_df.to_excel(writer, sheet_name='Yearly Details', index=False)
            
            # Get workbook and add formats
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4B8BBE',
                'font_color': 'white',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            
            # Get worksheets
            summary_worksheet = writer.sheets['Financial Summary']
            details_worksheet = writer.sheets['Yearly Details']
            
            # Format headers
            for col_num, value in enumerate(df.columns.values):
                summary_worksheet.write(0, col_num, value, header_format)
                
            for col_num, value in enumerate(final_yearly_df.columns):
                details_worksheet.write(0, col_num, value, header_format)
            
            # Adjust column widths
            summary_worksheet.set_column(0, 0, 20)
            summary_worksheet.set_column(1, 1, 25)
            summary_worksheet.set_column(2, 2, 30)
            
            details_worksheet.set_column(0, 0, 10)  # Year
            details_worksheet.set_column(1, 1, 20)  # Performance Ratio
            details_worksheet.set_column(2, 2, 20)  # Energy Output
            details_worksheet.set_column(3, 5, 25)  # Financial columns
            
            # Apply number formatting to relevant columns
            for row in range(1, len(final_yearly_df) + 1):
                details_worksheet.write(row, 1, final_yearly_df.iloc[row-1]['Performance Ratio (%)'], number_format)
                details_worksheet.write(row, 2, final_yearly_df.iloc[row-1]['Energy Output (kWh)'], number_format)
                details_worksheet.write(row, 3, final_yearly_df.iloc[row-1][f'{revenue_type} ({current_currency_symbol})'], number_format)
                details_worksheet.write(row, 4, final_yearly_df.iloc[row-1][f'O&M Cost ({current_currency_symbol})'], number_format)
                details_worksheet.write(row, 5, final_yearly_df.iloc[row-1][f'Net Cash Flow ({current_currency_symbol})'], number_format)
        
        # Download button
        st.download_button(
            label="📥 Download Detailed Report",
            data=buffer.getvalue(),
            file_name=f"solar_financial_report_{st.session_state.system_params['plant_capacity_kw']}kW.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.error(f"Debug info: {traceback.format_exc()}")
        # Print the structure of yearly_details for debugging
        st.error("Yearly details structure:")
        st.write(financial_results['yearly_details'][0] if financial_results['yearly_details'] else "No yearly details")


# Session State Management
if 'area_mapper' not in st.session_state:
    st.session_state.area_mapper = SolarAreaMapper()
    
if "design_chain" not in st.session_state:
    st.session_state.design_chain, st.session_state.general_chain = init_langchain()
if "tools" not in st.session_state:
    st.session_state.tools = init_tools()
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "initial"
if "system_params" not in st.session_state:
    st.session_state.system_params = {}
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add new inverter-related session states
if "inverter_params" not in st.session_state:
    st.session_state.inverter_params = None
if "selected_module" not in st.session_state:
    st.session_state.selected_module = None
if "selected_inverter" not in st.session_state:
    st.session_state.selected_inverter = None

# Financial-related session states
if "financial_calculator" not in st.session_state:
    st.session_state.financial_calculator = FinancialCalculator()
if "financial_results" not in st.session_state:
    st.session_state.financial_results = None
if "financial_settings" not in st.session_state:
    st.session_state.financial_settings = {}
if "financial_inputs" not in st.session_state:
    st.session_state.financial_inputs = {
        "project_cost": None,
        "om_params": None,
        "electricity_data": None,
        "tariff_type": "Flat Rate",
        "consumption_method": "Monthly Average"
    }
    
if "boq_generator" not in st.session_state:
    st.session_state.boq_generator = BOQGenerator()
    
if 'mapping_completed' not in st.session_state:
    st.session_state.mapping_completed = False

if 'map_center' not in st.session_state:
    st.session_state.map_center = [18.9384791, 72.8252102]  # Default to India

if "inputs_ready" not in st.session_state:
    st.session_state.inputs_ready = False

# Tab states
if 'tab_states' not in st.session_state:
    st.session_state.tab_states = {
        'project_cost_done': False,
        'om_settings_done': False,
        'electricity_done': False
    }
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False

# Progress tracking for inverter selection
if 'module_selection_done' not in st.session_state:
    st.session_state.module_selection_done = False
if 'inverter_selection_done' not in st.session_state:
    st.session_state.inverter_selection_done = False


def reset_conversation():
    """Reset all conversation-related state"""
    try:
        st.session_state.design_chain, st.session_state.general_chain = init_langchain()
        st.session_state.current_stage = "initial"
        st.session_state.system_params = {}
        st.session_state.results = None
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"Error resetting conversation: {str(e)}")

# UI Layout
st.set_page_config(
    page_title="Solar Expert AI",
    page_icon="🌞",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .system-params {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ System Details")
    
    if st.session_state.system_params:
        st.subheader("Current Parameters")
        st.markdown('<div class="system-params">', unsafe_allow_html=True)
        params_display = {
            "Location": st.session_state.system_params.get("location", "Not set"),
            "Capacity": f"{st.session_state.system_params.get('plant_capacity_kw', 0):.1f} kW",
            "Tilt": f"{st.session_state.system_params.get('surface_tilt', 20):.1f}°",
            "Azimuth": f"{st.session_state.system_params.get('surface_azimuth', 180):.1f}°"
        }
        for key, value in params_display.items():
            st.text(f"{key}: {value}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mode selection
    st.subheader("🔄 Mode Selection")
    mode = st.radio(
        "Choose interaction mode:",
        ["System Design & Calculation", "General Solar Knowledge"],
        help="Select 'System Design' for calculations or 'General Knowledge' for information"
    )
    
    if st.button("Start New Session"):
        reset_conversation()
        st.rerun()

# Main Area
st.title("🌞 Solar Expert AI")
if mode == "System Design & Calculation":
    st.caption("Design and analyze your solar power system")
else:
    st.caption("Ask me anything about solar energy")

# Chat interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
input_placeholder = ("Describe your solar project (e.g., '50kW system in Mumbai')" 
                    if mode == "System Design & Calculation"
                    else "Ask anything about solar energy...")
prompt = st.chat_input(input_placeholder)

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Process input based on mode
    with st.chat_message("assistant"):
        with st.spinner("Processing..." if mode == "System Design & Calculation" else "Thinking..."):
            if mode == "System Design & Calculation":
                conversation, structured_data = process_user_input(
                    prompt, 
                    st.session_state.design_chain,
                    st.session_state.general_chain
                )
            else:
                conversation, structured_data = process_user_input(
                    prompt,
                    st.session_state.general_chain,
                    st.session_state.general_chain
                )
            
            if conversation:
                st.write(conversation)
                st.session_state.chat_history.append({"role": "assistant", "content": conversation})

# Workflow stages
if st.session_state.current_stage == "area_mapping":
    st.divider()
    show_area_mapping()
elif st.session_state.current_stage == "select_module":
    st.divider()
    show_module_selection()
elif st.session_state.current_stage == "select_inverter":
    st.divider()
    show_inverter_selection()

# Results Display
if st.session_state.results:
    st.divider()
    
    # System Overview
    st.header("📊 System Analysis")
    
    # First row of metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            "System Capacity", 
            f"{st.session_state.results['system']['calculated_capacity']:,.1f} kW",
            help="Total system capacity in kilowatts"
        )
    with metrics_col2:
        st.metric(
            "Annual Energy", 
            f"{st.session_state.results['energy']['metrics']['total_yearly']:,.0f} kWh",
            help="Total annual energy production"
        )
    with metrics_col3:
        st.metric(
            "Total Modules", 
            f"{st.session_state.results['system']['total_modules']}",
            help="Number of solar modules"
        )
    with metrics_col4:
        st.metric(
            "System Area", 
            f"{st.session_state.results['system']['total_area']:,.1f} m²",
            help="Total system area required"
        )
    
    # Inverter Configuration Section
    st.divider()
    st.subheader("🔌 Inverter Configuration")
    
    if st.session_state.inverter_params:
        inverter_config = st.session_state.inverter_params['configuration']
        inverter_specs = st.session_state.inverter_params['specifications']
        
        # Inverter details in two rows
        col1, col2, col3 = st.columns(3)
        
        # First row
        with col1:
            st.metric(
                "Inverter Model",
                st.session_state.inverter_params['model'],
                help="Selected inverter model"
            )
            st.metric(
                "Number of Inverters",
                str(inverter_config['num_inverters']),
                help="Total number of inverters in the system"
            )
        
        with col2:
            st.metric(
                "DC/AC Ratio",
                f"{inverter_config['dc_ac_ratio']:.2f}",
                help="Ratio of DC capacity to AC capacity"
            )
            st.metric(
                "Modules per String",
                str(inverter_config['optimal_modules_per_string']),
                help="Optimal number of modules in each string"
            )
        
        with col3:
            st.metric(
                "Total Strings",
                str(inverter_config['total_strings']),
                help="Total number of strings in the system"
            )
            st.metric(
                "Strings per Inverter",
                str(inverter_config['strings_per_inverter']),
                help="Number of strings connected to each inverter"
            )
        
        # String Configuration Details
        with st.expander("🔍 View Detailed String Configuration"):
            st.markdown(f"""
            #### String Configuration Details
            - **Voltage Specifications**:
                - MPPT Voltage Range: {inverter_specs['mppt_range'][0]}V - {inverter_specs['mppt_range'][1]}V
                - Maximum DC Voltage: {inverter_specs['max_dc_voltage']}V
                - Rated MPPT Voltage: {inverter_specs['rated_mppt_voltage']}V
            
            - **Current Specifications**:
                - Maximum Input Current: {inverter_specs['max_input_current']}A per MPPT
                - Number of MPPTs: {inverter_specs['number_of_mppt']}
            
            - **Efficiency**:
                - Maximum Efficiency: {inverter_specs['max_efficiency']*100:.1f}%
                - Euro Efficiency: {inverter_specs['euro_efficiency']*100:.1f}%
            """)
            
            # Add warnings if necessary
            if inverter_config['dc_ac_ratio'] > 1.3:
                st.warning("⚠️ The DC/AC ratio is higher than recommended (>1.3). This may lead to power clipping during peak production hours.")
            elif inverter_config['dc_ac_ratio'] < 1.1:
                st.warning("⚠️ The DC/AC ratio is lower than recommended (<1.1). The inverter capacity might be underutilized.")
    
    else:
        st.info("No inverter configuration data available. Please complete the inverter selection process.")
    
    # Add a note about clipping losses if applicable
    if st.session_state.inverter_params and inverter_config['dc_ac_ratio'] > 1.2:
        energy_loss_estimate = ((inverter_config['dc_ac_ratio'] - 1.2) * 100 * 0.5)  # Rough estimation
        st.info(f"""
        ℹ️ **Note on Energy Production**: 
        With a DC/AC ratio of {inverter_config['dc_ac_ratio']:.2f}, you might experience some clipping losses 
        during peak production hours. Estimated annual production impact: ~{energy_loss_estimate:.1f}%
        """)
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Energy Production",
    "☀️ Solar Irradiation",
    "🔧 System Details",
    "📐 Positioning & Orientation",
    "⚡ Hourly Analysis",
    "💰 Financial Analysis",
    "🔌 System SLD",
    "📋 BOQ Generator"
])
    
    with tab1:
        st.subheader("Energy Production Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                st.session_state.results['energy']['figures']['daily'],
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                st.session_state.results['energy']['figures']['monthly'],
                use_container_width=True
            )
            
        # Key metrics
        st.subheader("Key Performance Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Daily Average", 
                f"{st.session_state.results['energy']['metrics']['total_yearly']/365:,.1f} kWh"
            )
        with metric_col2:
            st.metric(
                "Maximum Daily", 
                f"{st.session_state.results['energy']['metrics']['max_daily']:,.1f} kWh"
            )
        with metric_col3:
            st.metric(
                "Minimum Daily", 
                f"{st.session_state.results['energy']['metrics']['min_daily']:,.1f} kWh"
            )
    
    with tab2:
        st.subheader("Solar Irradiation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                st.session_state.results['irradiation']['figures']['daily'],
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                st.session_state.results['irradiation']['figures']['monthly'],
                use_container_width=True
            )
            
        st.divider()
    
        # Add location map below irradiation charts
        st.subheader("📍 Location Analysis")
        show_location_analysis()
    

    with tab3:
        st.subheader("System Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("#### 📍 Location Details")
            st.markdown(f"""
            - **Location**: {st.session_state.system_params.get('location', 'Not specified')}
            - **Latitude**: {st.session_state.system_params.get('latitude', 0):.4f}°
            - **Longitude**: {st.session_state.system_params.get('longitude', 0):.4f}°
            - **Timezone**: {st.session_state.system_params.get('tz_str', 'UTC')}
            """)
            
            st.markdown("#### 🎯 System Orientation")
            st.markdown(f"""
            - **Tilt Angle**: {st.session_state.system_params.get('surface_tilt', 20)}°
            - **Azimuth**: {st.session_state.system_params.get('surface_azimuth', 180)}°
            """)
        
        with config_col2:
            st.markdown("#### ⚡ System Specifications")
            st.markdown(f"""
            - **System Capacity**: {st.session_state.system_params.get('plant_capacity_kw', 0):.2f} kW
            - **Performance Ratio**: {st.session_state.system_params.get('performance_ratio', 0.8)*100:.1f}%
            """)
            
            st.markdown("#### 🔋 Module Details")
            st.markdown(f"""
            - **Module Power**: {st.session_state.system_params.get('module_watt_peak', 0)} W
            - **Module Efficiency**: {st.session_state.system_params.get('module_efficiency', 0)*100:.1f}%
            - **Module Area**: {st.session_state.system_params.get('module_area', 0):.2f} m²
            """)
        
        # Add a box with key performance indicators
        st.markdown("---")
        st.markdown("#### 📊 Key System Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            total_modules = int(st.session_state.results['system']['total_modules'])
            st.metric(
                "Total Modules",
                f"{total_modules:,}",
                help="Number of modules in the system"
            )
        
        with metric_col2:
            total_area = st.session_state.results['system']['total_area']
            st.metric(
                "Total System Area",
                f"{total_area:.1f} m²",
                help="Total area required for installation"
            )
        
        with metric_col3:
            specific_yield = st.session_state.results['energy']['metrics']['total_yearly'] / st.session_state.system_params['plant_capacity_kw']
            st.metric(
                "Specific Yield",
                f"{specific_yield:.0f} kWh/kWp",
                help="Annual energy generation per kW of installed capacity"
            )
            
            
    with tab4:
        col1, col2 = st.columns(2)
    
        with col1:
            # Panel orientation visualization
            st.markdown("#### 3D Panel Orientation")
            show_3d_orientation()
        
        with col2:
            # Sunpath diagram
            st.markdown("#### Annual Sunpath Diagram")
            show_sunpath_analysis()
        
    with tab5:
        show_hourly_analysis()
        
    with tab6:
        show_financial_analysis()
        
    with tab7:
        show_system_sld()
        
    with tab8:
        st.session_state.boq_generator.display_boq_generator(st.session_state.system_params)