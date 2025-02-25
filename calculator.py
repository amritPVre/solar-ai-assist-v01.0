def create_energy_heatmap(energy_data):
    """Create heatmap visualization of energy production patterns"""
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from datetime import datetime

    try:
        # Validate energy_data structure
        if not energy_data or 'energy' not in energy_data or 'daily' not in energy_data['energy']:
            st.warning("Energy data is not properly structured for heatmap visualization")
            return None
            
        # Get daily data
        daily_df = pd.DataFrame({
            'Date': pd.to_datetime(energy_data['energy']['daily']['Date']),
            'Energy': energy_data['energy']['daily']['Daily Energy Production (kWh)']
        })
        
        # Make sure Date is actually datetime
        if not pd.api.types.is_datetime64_any_dtype(daily_df['Date']):
            daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        
        # Add month component
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
            monthly_data = daily_df[daily_df['Month'] == month]
            if len(monthly_data) > 0:
                monthly_avg = monthly_data['Energy'].mean()
            else:
                monthly_avg = 0
                
            for hour in range(24):
                heatmap_data[hour, month-1] = monthly_avg * hourly_weights[hour]
        
        # Create month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create the heatmap - FIX: changed colorbar settings
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=months,
            y=[f"{hour:02d}:00" for hour in hours],
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate="Month: %{x}<br>Time: %{y}<br>Energy: %{z:.2f} kWh<extra></extra>",
            colorbar=dict(
                title="Energy Production (kWh)"
                # Removed 'titleside' property that was causing the error
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
        st.write(f"Error details: {traceback.format_exc()}")
        return None
