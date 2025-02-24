import streamlit as st
import folium
from folium import plugins
from folium.plugins import Draw, MeasureControl
from streamlit_folium import folium_static, st_folium
from geopy.geocoders import Nominatim
import json
from typing import Dict, Any, Tuple
import math
import traceback
from branca.element import Figure, JavascriptLink, CssLink
from folium.plugins import Fullscreen
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
from gcr import (
    get_gcr_categories,
    get_installation_types,
    get_gcr_value,
    get_gcr_range,
    get_installation_description
)

class SolarAreaMapper:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="solar_expert_ai")
        
    def _calculate_area(self, coordinates: list) -> float:
        """
        Calculate area of polygon in square meters using Shoelace formula
        """
        if len(coordinates) < 3:
            return 0
            
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)
            
            a = (math.sin(delta_phi/2) * math.sin(delta_phi/2) +
                 math.cos(phi1) * math.cos(phi2) *
                 math.sin(delta_lambda/2) * math.sin(delta_lambda/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
            
        area = 0
        j = len(coordinates) - 1
        
        for i in range(len(coordinates)):
            p1 = coordinates[i]
            p2 = coordinates[j]
            
            # Calculate trapezoid area using haversine distances
            lat_dist = haversine_distance(p1[0], p1[1], p2[0], p1[1])
            lon_dist = haversine_distance(p1[0], p1[1], p1[0], p2[1])
            area += p1[0] * p2[1] - p2[0] * p1[1]
            j = i
            
        return abs(area) / 2

    def create_installation_map(self, default_location: str = None) -> Dict[str, Any]:
        """
        Create interactive map for drawing solar installation areas
        """
        try:
            st.subheader("🗺️ Solar Installation Area Mapping")
            
            # Show initial values if provided
            if hasattr(st.session_state, 'initial_capacity'):
                st.info(f"💡 Initial capacity provided: {st.session_state.initial_capacity:.1f} kW")
            
            # Project Category and Installation Type Selection
            st.markdown("### Project Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                # Project Category Selection
                categories = get_gcr_categories()
                project_category = st.selectbox(
                    "Select Project Category:",
                    options=categories,
                    help="Choose the type of solar project"
                )
                # Store selection in session state
                st.session_state.project_category = project_category
            
            with col2:
                # Installation Type Selection (dependent on category)
                installation_types = get_installation_types(project_category)
                installation_type = st.selectbox(
                    "Select Installation Type:",
                    options=installation_types,
                    help="Choose specific installation configuration"
                )
                # Store selection in session state
                st.session_state.installation_type = installation_type
            
            # Show GCR information
            gcr_value = get_gcr_value(project_category, installation_type)
            gcr_range = get_gcr_range(project_category, installation_type)
            description = get_installation_description(project_category, installation_type)
            
            st.info(f"""
            📊 Installation Configuration:
            - Ground Coverage Ratio (GCR): {gcr_value*100:.1f}%
            - GCR Range: {gcr_range[0]*100:.1f}% - {gcr_range[1]*100:.1f}%
            - Description: {description}
            """)
            
            # Location search
            st.markdown("### Location Selection")
            col1, col2 = st.columns([3, 1])
            with col1:
                search_location = st.text_input(
                    "Search location:",
                    value=default_location if default_location else "",
                    help="Enter an address, city, or coordinates"
                )
            
            with col2:
                if st.button("Search", key="location_search"):
                    if search_location:
                        location_data = self.geolocator.geocode(search_location)
                        if location_data:
                            st.session_state.map_center = [
                                location_data.latitude,
                                location_data.longitude
                            ]
                            st.rerun()
                        else:
                            st.error("Location not found")
            
            # Initialize map center
            if 'map_center' not in st.session_state:
                if default_location:
                    location_data = self.geolocator.geocode(default_location)
                    if location_data:
                        st.session_state.map_center = [
                            location_data.latitude,
                            location_data.longitude
                        ]
                    else:
                        st.session_state.map_center = [18.9384791, 72.8252102]
                else:
                    st.session_state.map_center = [18.9384791, 72.8252102]
            
            # Create base map
            m = folium.Map(
                location=st.session_state.map_center,
                zoom_start=20,
                max_zoom=24,
                control_scale=True
            )
            
            # [All existing map configuration code remains exactly the same]
            plugins.Geocoder(collapsed=True).add_to(m)
            plugins.MousePosition().add_to(m)
            
            folium.TileLayer(
                'CartoDB positron',
                name='Light Map',
                max_zoom=18
            ).add_to(m)
            
            folium.TileLayer(
                'OpenStreetMap',
                name='Street Map',
                max_zoom=19
            ).add_to(m)
            
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite (Esri)',
                max_zoom=18
            ).add_to(m)
            
            folium.TileLayer(
                'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                name='Topographic',
                attr='Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
                max_zoom=17
            ).add_to(m)
            
            folium.TileLayer(
                'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                name='Google Satellite',
                attr='Google',
                max_zoom=20,
                overlay=False
            ).add_to(m)
    
            # Add custom tile switching logic
            custom_js = """
            <script>
            document.addEventListener('leaflet.tile.loaded', function(e) {
                var zoom = e.target._map.getZoom();
                var activeLayer = e.target;
                
                if (zoom > activeLayer.options.maxZoom) {
                    // Switch to a layer that supports higher zoom
                    var layers = Object.values(e.target._map._layers);
                    var highZoomLayer = layers.find(l => l.options && l.options.maxZoom >= zoom);
                    if (highZoomLayer) {
                        highZoomLayer.addTo(e.target._map);
                    }
                }
            });
            </script>
            """
            m.get_root().html.add_child(folium.Element(custom_js))
    
            # Add all existing controls
            draw = Draw(
                position='topleft',
                draw_options={
                    'polyline': False,
                    'polygon': {
                        'allowIntersection': False,
                        'drawError': {'color': '#e1e100', 'message': 'Polygon cannot intersect!'},
                        'shapeOptions': {'color': '#97009c', 'fillOpacity': 0.3}
                    },
                    'circle': True,
                    'rectangle': {
                        'shapeOptions': {'color': '#97009c', 'fillOpacity': 0.3}
                    },
                    'marker': True,
                    'circlemarker': False
                }
            )
            m.add_child(draw)
            
            measure = MeasureControl(
                position='topleft',
                primary_length_unit='meters',
                primary_area_unit='sqmeters',
                active_color='red',
                completed_color='green'
            )
            m.add_child(measure)
            
            plugins.Fullscreen(
                position='topleft',
                title='Expand map',
                title_cancel='Exit fullscreen',
                force_separate_button=True,
                fullscreen=True
            ).add_to(m)
            
            folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
            # Add existing custom CSS
            custom_css = """
            <style>
                .leaflet-draw-toolbar {
                    margin-top: 0 !important;
                }
                .leaflet-draw-toolbar .leaflet-draw-draw-polygon { 
                    background-position: -31px -2px;
                }
                .leaflet-draw-toolbar .leaflet-draw-draw-rectangle { 
                    background-position: -62px -2px;
                }
                .leaflet-touch .leaflet-bar {
                    border: 2px solid rgba(0,0,0,0.2);
                }
                .leaflet-control-zoom {
                    border: none !important;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.2) !important;
                }
                .leaflet-control-zoom a {
                    width: 30px !important;
                    height: 30px !important;
                    line-height: 30px !important;
                    font-size: 15px !important;
                    font-weight: bold !important;
                }
                .leaflet-control-zoom-in {
                    border-top-left-radius: 4px !important;
                    border-top-right-radius: 4px !important;
                }
                .leaflet-control-zoom-out {
                    border-bottom-left-radius: 4px !important;
                    border-bottom-right-radius: 4px !important;
                }
            </style>
            """
            m.get_root().html.add_child(folium.Element(custom_css))
    
            # Display map with specific height
            st_data = st_folium(
                m,
                width="100%",
                height=600,
                returned_objects=["last_active_drawing", "all_drawings"],
                key="solar_map"
            )
            
            # Process drawn features and show analysis
            has_drawn_area = False
            potential = None
            
            if st_data and st_data.get("last_active_drawing"):
                feature = st_data["last_active_drawing"]
                if feature and "geometry" in feature:
                    potential = self.calculate_installation_potential({
                        "type": "Feature",
                        "geometry": feature["geometry"]
                    })
                    if potential:
                        has_drawn_area = True
    
            # Display analysis and navigation options
            # Display analysis and navigation options
            if has_drawn_area and potential:
                # Analysis section
                with st.container():
                    st.markdown("### Installation Area Analysis")
                    
                    # Area metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Area",
                            f"{potential['total_area']:.1f} m²",
                            help="Total drawn area"
                        )
                    with col2:
                        st.metric(
                            "Initial Usable Area",
                            f"{potential['initial_usable_area']:.1f} m²",
                            help="After 85% usability factor"
                        )
                    with col3:
                        st.metric(
                            "Final Usable Area",
                            f"{potential['final_usable_area']:.1f} m²",
                            help="After GCR application"
                        )
                    
                    # Capacity and configuration
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"""
                            📊 Installation Area Analysis:
                            - Total Area: {potential['total_area']:.1f} m²
                            - GCR Applied: {potential['gcr']*100:.1f}%
                            - Final Usable Area: {potential['final_usable_area']:.1f} m²
                            - Estimated Modules: {potential['modules_estimate']}
                            - Potential Capacity: {potential['potential_capacity_low']:.1f} - {potential['potential_capacity_high']:.1f} kW
                            """)
                    
                    with col2:
                        if hasattr(st.session_state, 'initial_capacity'):
                            diff = ((potential['potential_capacity'] - st.session_state.initial_capacity) 
                                   / st.session_state.initial_capacity * 100)
                            st.metric(
                                "Capacity Difference",
                                f"{diff:+.1f}%",
                                help="Difference between mapped and initially provided capacity",
                                delta_color="inverse"
                            )
            
            # Navigation Options
            st.markdown("### Navigation Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if has_drawn_area and potential:
                    if st.button("Use Mapped Area", type="primary"):
                        st.session_state.system_params.update({
                            "total_area": potential['total_area'],
                            "initial_usable_area": potential['initial_usable_area'],
                            "final_usable_area": potential['final_usable_area'],  # This is what module selection will use
                            "plant_capacity_kw": potential['potential_capacity'],
                            "gcr_value": potential['gcr'],
                            "project_category": potential['project_category'],
                            "installation_type": potential['installation_type']
                        })
                        st.session_state.mapping_completed = True
                        st.session_state.current_stage = "select_module"
                        st.rerun()
            
            with col2:
                if st.button("Use Initial Capacity"):
                    if hasattr(st.session_state, 'initial_capacity'):
                        st.session_state.system_params["plant_capacity_kw"] = st.session_state.initial_capacity
                    st.session_state.mapping_completed = True
                    st.session_state.current_stage = "select_module"
                    st.rerun()
            
            with col3:
                if st.button("Skip Area Mapping"):
                    if hasattr(st.session_state, 'initial_capacity'):
                        st.session_state.system_params["plant_capacity_kw"] = st.session_state.initial_capacity
                    st.session_state.mapping_completed = True
                    st.session_state.current_stage = "select_module"
                    st.rerun()
    
            return {
                "map_center": st.session_state.map_center,
                "drawn_data": st_data
            }
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            st.error(traceback.format_exc())
            return None
            
    def calculate_area_from_feature(self, feature: dict) -> float:
        """Calculate area from GeoJSON feature"""
        try:
            if feature["geometry"]["type"] == "Polygon":
                # Extract coordinates in the correct order (lat, lon)
                coords = [(coord[1], coord[0]) for coord in feature["geometry"]["coordinates"][0]]
                return self._calculate_area_from_coords(coords)
            return 0
        except Exception as e:
            st.error(f"Error calculating area: {str(e)}")
            return 0
    
    def calculate_installation_potential(self, feature: Dict) -> Dict[str, float]:
        """Calculate installation potential based on drawn area and selected GCR"""
        try:
            if not feature or "geometry" not in feature:
                return None
                
            geometry = feature["geometry"]
            if geometry["type"] != "Polygon":
                return None
                
            # Extract coordinates
            coords = geometry["coordinates"][0]
            
            # Calculate total area
            total_area = self._calculate_area_from_coords(coords)
            
            if total_area > 0:
                # Get selected project category and installation type from session state
                project_category = st.session_state.get('project_category')
                installation_type = st.session_state.get('installation_type')
                
                # Get GCR information
                default_gcr = get_gcr_value(project_category, installation_type)
                gcr_range = get_gcr_range(project_category, installation_type)
                
                # Calculate initial usable area (85% of total)
                initial_usable_area = total_area * 0.85
                
                # Allow user to adjust GCR within recommended range
                st.markdown("### GCR Fine-tuning")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    gcr = st.slider(
                        "Ground Coverage Ratio (GCR):",
                        min_value=float(gcr_range[0]),
                        max_value=float(gcr_range[1]),
                        value=float(default_gcr),
                        step=0.01,
                        format="%.2f",
                        help=f"Recommended range: {gcr_range[0]*100:.1f}% - {gcr_range[1]*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Selected GCR",
                        f"{gcr*100:.1f}%",
                        delta=f"{(gcr-default_gcr)*100:+.1f}%",
                        help="Difference from recommended GCR"
                    )
                
                # Calculate final usable area with GCR
                final_usable_area = initial_usable_area * gcr
                
                # Estimating PV capacity)
                modules_estimate = int(final_usable_area / 2.16)  # Using a typical module size of 2.16m²/500W
                potential_capacity_low = modules_estimate * 450 / 1000  # Low-power modules (~450W)  
                potential_capacity_high = modules_estimate * 620 / 1000  # High-power modules (~620W)
                potential_capacity = (potential_capacity_low + potential_capacity_high) / 2
                # Store in the nested dictionary as before
                potential_capacity_range = {
                    "low": potential_capacity_low,
                    "high": potential_capacity_high
                }
                
                return {
                    "total_area": total_area,
                    "initial_usable_area": initial_usable_area,
                    "final_usable_area": final_usable_area,
                    "modules_estimate": modules_estimate,
                    "potential_capacity_range": potential_capacity_range,
                    "potential_capacity_low": potential_capacity_low,  # Add direct access keys
                    "potential_capacity_high": potential_capacity_high,  # Add direct access keys
                    "potential_capacity": potential_capacity,  # Add average value
                    "gcr": gcr,
                    "gcr_range": gcr_range,
                    "project_category": project_category,
                    "installation_type": installation_type
                }
            return None
                
        except Exception as e:
            st.error(f"Error calculating potential: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    # In area_mapper.py, improve area calculation:
    def _calculate_area_from_coords(self, coords: list) -> float:
        """Calculate area from coordinates using GeoPy"""
        try:
            if len(coords) < 3:
                return 0
                
            import numpy as np
            from geopy.distance import geodesic
            
            # Use Shoelace formula with geodesic distances
            # This is more accurate for geographic coordinates
            area = 0
            coords_array = np.array(coords)
            n = len(coords_array)
            
            # Convert to radians
            lat_rad = np.radians(coords_array[:, 1])
            lon_rad = np.radians(coords_array[:, 0])
            
            # Earth radius in meters
            R = 6371000
            
            # Apply spherical earth Shoelace formula
            for i in range(n):
                j = (i + 1) % n
                area += (lon_rad[j] - lon_rad[i]) * (2 + np.sin(lat_rad[i]) + np.sin(lat_rad[j]))
            
            area = abs(area * R * R / 2.0)
            return area
                
        except Exception as e:
            st.error(f"Error in area calculation: {str(e)}")
            return 0