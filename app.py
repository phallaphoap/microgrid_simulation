# app.py

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import streamlit.components.v1 as components

# Load environment variables from .env file
load_dotenv()

# Access the OpenWeatherMap API key
# OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

# Access the OpenWeatherMap API key from secrets
OPENWEATHERMAP_API_KEY = st.secrets["OPENWEATHERMAP_API_KEY"]

# Streamlit page configuration
st.set_page_config(page_title="Microgrid Simulation", layout="wide")

# -------------------------------
# Function Definitions
# -------------------------------

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_weather_data(city):
    """
    Fetches current and forecast weather data for a specified city using OpenWeatherMap API.

    Args:
        city (str): Name of the city.

    Returns:
        tuple: Current weather data, forecast data, and fetch time.
    """
    if not OPENWEATHERMAP_API_KEY:
        st.error("OpenWeatherMap API Key is not set. Please set it in the .env file.")
        return None, None, None

    # Geocoding API to get latitude and longitude
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHERMAP_API_KEY}"
    geocode_response = requests.get(geocode_url).json()

    if not geocode_response:
        st.error("City not found. Please enter a valid city.")
        return None, None, None

    lat = geocode_response[0]['lat']
    lon = geocode_response[0]['lon']

    # Current Weather Data
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHERMAP_API_KEY}"
    current_data = requests.get(current_url).json()

    # Forecast Data (5-day/3-hour forecast)
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHERMAP_API_KEY}"
    forecast_data = requests.get(forecast_url).json()

    # Record the fetch time
    fetch_time = datetime.utcnow()

    return current_data, forecast_data, fetch_time


def calculate_current_solar_output(current_weather, is_daytime, solar_capacity, panel_efficiency):
    """
    Calculates the current solar output based on weather conditions, daytime status, and solar panel parameters.

    Args:
        current_weather (dict): Current weather data.
        is_daytime (bool): Flag indicating if it's daytime.
        solar_capacity (float): Installed solar capacity in kW.
        panel_efficiency (float): Panel efficiency as a percentage.

    Returns:
        float: Calculated solar output in kW.
    """
    if not is_daytime:
        return 0.0  # No solar output at night

    temp = current_weather['main']['temp']
    cloud_coverage = current_weather['clouds']['all']
    # Adjust base output with panel efficiency
    base_output = solar_capacity * (panel_efficiency / 100.0)
    cloud_factor = (100 - cloud_coverage) / 100
    temp_factor = 1 - abs(temp - 25) * 0.02  # Simplistic temperature factor
    solar_output = base_output * cloud_factor * temp_factor
    return round(max(0, solar_output), 2)


def calculate_solar_output_forecast(forecast_data, solar_capacity, panel_efficiency):
    """
    Calculates solar output forecast based on forecast data and solar panel parameters.

    Args:
        forecast_data (dict): Forecast weather data.
        solar_capacity (float): Installed solar capacity in kW.
        panel_efficiency (float): Panel efficiency as a percentage.

    Returns:
        pd.DataFrame: DataFrame containing timestamps and solar output.
    """
    forecast_list = forecast_data.get('list', [])
    if not forecast_list:
        st.warning("No forecast data available for solar output calculation.")
        return pd.DataFrame()

    solar_output_forecast = []
    base_output = solar_capacity * (panel_efficiency / 100.0)

    for item in forecast_list:
        pod = item['sys'].get('pod', 'd')  # 'd' for day, 'n' for night
        if pod == 'd':
            temp = item['main']['temp']
            cloud_coverage = item['clouds']['all']
            cloud_factor = (100 - cloud_coverage) / 100
            temp_factor = 1 - abs(temp - 25) * 0.02  # Simplistic temperature factor
            solar_output = base_output * cloud_factor * temp_factor
            solar_output = max(0, solar_output)  # Ensure non-negative
        else:
            solar_output = 0  # No solar output during night

        solar_output_forecast.append({
            'timestamp': datetime.utcfromtimestamp(item['dt']),
            'solar_output': round(solar_output, 2)
        })

    solar_df = pd.DataFrame(solar_output_forecast)
    return solar_df


def plot_weather_forecast(forecast_data):
    """
    Plots the weather forecast including temperature and cloud coverage.

    Args:
        forecast_data (dict): Forecast weather data.
    """
    forecast_list = forecast_data.get('list', [])
    if not forecast_list:
        st.warning("No forecast data available to plot.")
        return

    forecast_df = pd.DataFrame({
        'timestamp': [datetime.utcfromtimestamp(item['dt']) for item in forecast_list],
        'temperature': [item['main']['temp'] for item in forecast_list],
        'weather': [item['weather'][0]['description'] for item in forecast_list],
        'wind_speed': [item['wind']['speed'] for item in forecast_list],
        'clouds': [item['clouds']['all'] for item in forecast_list],
        'pod': [item['sys'].get('pod', 'd') for item in forecast_list]
    })

    # Adjust timestamps to local time using timezone offset
    timezone_offset = forecast_data['city'].get('timezone', 0)  # in seconds
    forecast_df['timestamp'] = forecast_df['timestamp'].apply(lambda x: x + timedelta(seconds=timezone_offset))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'],
        y=forecast_df['temperature'],
        mode='lines+markers',
        name='Temperature (°C)',
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'],
        y=forecast_df['clouds'],
        mode='lines+markers',
        name='Cloud Coverage (%)',
        yaxis='y2'
    ))
    fig.update_layout(
        title='Here is the forecast for the next 5 days:',
        xaxis_title='Time (Local)',
        yaxis=dict(
            title='Temperature (°C)',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title='Cloud Coverage (%)',
            titlefont=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_solar_output_forecast(solar_df):
    """
    Plots the solar output forecast.

    Args:
        solar_df (pd.DataFrame): DataFrame containing solar output forecast.
    """
    if solar_df.empty:
        st.warning("No solar output forecast data available to plot.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=solar_df['timestamp'],
        y=solar_df['solar_output'],
        mode='lines+markers',
        name='Solar Output (kW)',
        line=dict(color='orange')
    ))
    fig.update_layout(
        title='From our weather forecast above, we predict the following solar output for the next 5 days:',
        xaxis_title='Time (Local)',
        yaxis_title='Solar Output (kW)',
        legend=dict(x=0.01, y=0.99),
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_energy_balance(energy_df):
    """
    Plots the energy balance overview including solar output, load consumption, and battery SoC.

    Args:
        energy_df (pd.DataFrame): DataFrame containing energy balance data.
    """
    if energy_df.empty:
        st.warning("No energy balance data available to plot.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=energy_df['timestamp'],
        y=energy_df['solar_output'],
        name='Solar Output (kW)',
        marker_color='orange'
    ))
    fig.add_trace(go.Bar(
        x=energy_df['timestamp'],
        y=energy_df['load_consumption'],
        name='Load Consumption (kW)',
        marker_color='blue'
    ))
    fig.add_trace(go.Scatter(
        x=energy_df['timestamp'],
        y=energy_df['state_of_charge'],
        name='Battery SoC (kWh)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='green')
    ))
    fig.update_layout(
        title='Energy Production vs. Consumption',
        xaxis_title='Time (Local)',
        yaxis=dict(
            title='Power (kW)',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            side='left'
        ),
        yaxis2=dict(
            title='Battery SoC (kWh)',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_live_clock(timezone_offset, is_daytime):
    """
    Renders a styled live clock that updates every second based on the timezone offset and theming.

    Args:
        timezone_offset (int): Timezone offset in seconds from UTC.
        is_daytime (bool): Flag indicating if it's daytime.
    """
    # Define CSS styles for dark and light modes
    if is_daytime:
        theme = """
            #live-clock {
                color: #000000;
                background-color: #FFFFFF;
                margin-left: -8px;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                display: inline-block;
            }
        """
    else:
        theme = """
            #live-clock {
                color: #FFFFFF;
                background-color: #333333;
                margin-left: -8px;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                display: inline-block;
            }
        """

    # JavaScript code to display live clock with desired format
    clock_html = f"""
    <style>
        {theme}
    </style>
    <div id="live-clock"></div>
    <script>
        const timezoneOffset = {timezone_offset}; // in seconds

        function updateClock() {{
            const now = new Date();
            // Convert current time to UTC milliseconds
            const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
            // Apply timezone offset
            const localTime = new Date(utc + (timezoneOffset * 1000));

            const options = {{
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: 'numeric',
                minute: 'numeric',
                second: 'numeric',
                hour12: true
            }};

            const formattedTime = localTime.toLocaleString('en-US', options);
            document.getElementById('live-clock').innerHTML = formattedTime;
        }}

        // Update clock every second
        setInterval(updateClock, 1000);
        updateClock(); // Initial call
    </script>
    """
    # Embed the HTML and JavaScript into the Streamlit app
    components.html(clock_html, height=60)


def generate_energy_description(row):
    """
    Generates a description of the energy flow based on the energy transfers.

    Args:
        row (pd.Series): The row of the DataFrame at the selected time.

    Returns:
        str: A descriptive text of the energy flow.
    """
    solar_output = row['solar_output']
    battery_soc = row['state_of_charge']
    solar_to_load = row['solar_to_load']
    solar_to_battery = row['solar_to_battery']
    battery_to_load = row['battery_to_load']
    load_shedding = row['load_shedding']

    description_lines = []

    if solar_output > 0:
        description_lines.append(f"☀️ The sun is out. Solar panels are generating {solar_output:.2f} kW.")
        if solar_to_load > 0:
            description_lines.append(f"- {solar_to_load:.2f} kW is being used to power the load.")
        if solar_to_battery > 0:
            description_lines.append(f"- {solar_to_battery:.2f} kW is charging the battery.")
    else:
        description_lines.append("🌙 The sun is down. Solar panels are not generating power.")

    if battery_to_load > 0:
        description_lines.append(f"The battery is discharging {battery_to_load:.2f} kW to power the load.")

    if load_shedding > 0:
        description_lines.append(f"⚠️ There is a load shedding of {load_shedding:.2f} kW due to insufficient energy.")

    description_lines.append(f"The battery state of charge is {battery_soc:.2f} kWh.")

    # Join the lines with line breaks
    description = "\n".join(description_lines)

    return description


def plot_energy_transfers(energy_balance_df, selected_time, battery_capacity, solar_capacity, panel_efficiency):
    """
    Plots energy transfers between Solar Panels, Battery, and Loads at a selected timestamp.

    Args:
        energy_balance_df (pd.DataFrame): DataFrame containing energy balance and transfer data.
        selected_time (datetime): The selected timestamp to visualize.
        battery_capacity (float): The maximum capacity of the battery.
        solar_capacity (float): Installed solar capacity in kW.
        panel_efficiency (float): Panel efficiency as a percentage.
    """
    # Filter the DataFrame for the selected time
    row = energy_balance_df[energy_balance_df['timestamp'] == selected_time]

    if row.empty:
        st.warning("No data available for the selected time.")
        return

    row = row.iloc[0]

    # Define component positions
    components_pos = {
        'Solar Panels': {'x': 0, 'y': 1},
        'Battery': {'x': 1, 'y': 0},
        'Loads': {'x': 2, 'y': 1}
    }

    # Calculate current over maximum values
    solar_output = row['solar_output']
    max_solar_output = solar_capacity * (panel_efficiency / 100.0)
    solar_label = f"Solar Panels<br>{solar_output:.2f} kW / {max_solar_output:.2f} kW"

    battery_soc = row['state_of_charge']
    battery_label = f"Battery<br>{battery_soc:.2f} kWh / {battery_capacity:.2f} kWh"

    # Initialize Plotly figure
    fig = go.Figure()

    # Add component markers with additional labels
    for name, pos in components_pos.items():
        if name == 'Solar Panels':
            text = solar_label
        elif name == 'Battery':
            text = battery_label
        else:
            text = name  # For Loads, keep the name as is

        fig.add_trace(go.Scatter(
            x=[pos['x']],
            y=[pos['y']],
            mode='markers+text',
            marker=dict(
                size=50,
                color='lightblue' if name == 'Solar Panels' else 'lightgreen' if name == 'Battery' else 'salmon',
                symbol='circle'  # Changed from 'square' to 'circle'
            ),
            text=[text],
            textposition="top center",
            hoverinfo='text',
            showlegend=False
        ))

    # Define energy transfers
    transfers = [
        {
            'from': 'Solar Panels',
            'to': 'Loads',
            'value': row['solar_to_load'],
            'color': 'green',
            'path': 'curved'  # Use 'curved' for non-straight paths
        },
        {
            'from': 'Solar Panels',
            'to': 'Battery',
            'value': row['solar_to_battery'],
            'color': 'blue',
            'path': 'straight'
        },
        {
            'from': 'Battery',
            'to': 'Loads',
            'value': row['battery_to_load'],
            'color': 'red',
            'path': 'straight'
        }
    ]

    # Add lines for each transfer
    for transfer in transfers:
        if transfer['value'] > 0:
            from_pos = components_pos[transfer['from']]
            to_pos = components_pos[transfer['to']]

            if transfer['path'] == 'straight':
                # Draw straight line
                fig.add_trace(go.Scatter(
                    x=[from_pos['x'], to_pos['x']],
                    y=[from_pos['y'], to_pos['y']],
                    mode='lines',
                    line=dict(color=transfer['color'], width=2),
                    hoverinfo='none',
                    showlegend=False,
                ))
                # Add label at the midpoint
                mid_x = (from_pos['x'] + to_pos['x']) / 2
                mid_y = (from_pos['y'] + to_pos['y']) / 2
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y + 0.1,  # Slightly above the line
                    text=f"{transfer['value']:.2f} kW",
                    showarrow=False,
                    font=dict(color=transfer['color'], size=12),
                    align='center',
                )
            elif transfer['path'] == 'curved':
                # Draw curved line by adding intermediate points
                control_x = (from_pos['x'] + to_pos['x']) / 2
                control_y = max(from_pos['y'], to_pos['y']) + 0.5  # Adjust the control point to curve the line
                fig.add_trace(go.Scatter(
                    x=[from_pos['x'], control_x, to_pos['x']],
                    y=[from_pos['y'], control_y, to_pos['y']],
                    mode='lines',
                    line=dict(color=transfer['color'], width=2),
                    hoverinfo='none',
                    showlegend=False,
                ))
                # Add label at the approximate midpoint of the curve
                label_x = control_x
                label_y = control_y + 0.1  # Slightly above the curve
                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=f"{transfer['value']:.2f} kW",
                    showarrow=False,
                    font=dict(color=transfer['color'], size=12),
                    align='center',
                )

    # Update layout
    fig.update_layout(
        title=f"⚡ Energy Transfers at {selected_time.strftime('%Y-%m-%d %H:%M:%S')}",
        showlegend=False,
        xaxis=dict(range=[-0.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 2.0], showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Generate and display the description
    description = generate_energy_description(row)
    st.info(description)

# -------------------------------
# Main Application Function
# -------------------------------

def main():
    st.title("🔌 Microgrid Simulation")

    # Initialize session state for city
    if 'city' not in st.session_state:
        st.session_state['city'] = 'Lagos'

    # Define load profiles
    load_profiles = {
        "Residential": {
            "description": "Typical household energy consumption.",
            "power_consumption": [1.2, 1.0, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.2]  # in kW for each 3-hour interval
        },
        "Commercial": {
            "description": "Office building energy consumption.",
            "power_consumption": [3.0, 3.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.0, 4.5, 4.0, 3.5]
        },
        "Industrial": {
            "description": "Factory energy consumption.",
            "power_consumption": [5.0, 5.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.0, 6.5, 6.0, 5.5]
        }
    }

    # Sidebar for location selection
    st.sidebar.header("📍 Location Selection")
    city = st.sidebar.text_input("Enter a city:", st.session_state['city'])
    st.session_state['city'] = city

    # Sidebar for load profile selection
    st.sidebar.header("⚡ Load Profile Selection")
    selected_profile = st.sidebar.selectbox(
        "Choose a Load Profile:",
        options=list(load_profiles.keys()),
        index=0
    )

    # Display selected load profile description
    st.sidebar.write(load_profiles[selected_profile]["description"])

    # Sidebar for battery parameters
    st.sidebar.header("🔋 Battery Parameters")
    battery_capacity = st.sidebar.number_input("Battery Capacity (kWh):", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    battery_max_charge = st.sidebar.number_input("Max Charge Rate (kW):", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    battery_max_discharge = st.sidebar.number_input("Max Discharge Rate (kW):", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    initial_soc = st.sidebar.number_input("Initial State of Charge (kWh):", min_value=0.0, max_value=battery_capacity, value=5.0, step=0.1)

    # Sidebar for solar panel parameters
    st.sidebar.header("🔆 Solar Panel Parameters")
    solar_capacity = st.sidebar.number_input("Installed Capacity (kW):", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
    panel_efficiency = st.sidebar.slider("Panel Efficiency (%):", min_value=10.0, max_value=100.0, value=100.0, step=5.0)

    if city:
        with st.spinner('Fetching weather data...'):
            current_data, forecast_data, fetch_time = get_weather_data(city)

        if current_data and forecast_data:
            # Retrieve selected load profile data
            load_consumption = load_profiles[selected_profile]["power_consumption"]

            # Calculate solar output forecast
            solar_df = calculate_solar_output_forecast(forecast_data, solar_capacity, panel_efficiency)
            solar_output = solar_df['solar_output'].tolist()

            # Ensure load_consumption length matches solar_output
            min_length = min(len(load_consumption), len(solar_output))
            load_consumption = load_consumption[:min_length]
            solar_output = solar_output[:min_length]

            # Initialize battery SoC
            battery_soc = [initial_soc]

            # Simulate energy flow and battery SoC
            solar_to_load = []
            solar_to_battery = []
            battery_to_load = []
            load_shedding = []

            for i in range(min_length):
                available_solar = solar_output[i]
                required_load = load_consumption[i]
                surplus = available_solar - required_load

                # Solar to Load
                stl = min(available_solar, required_load)
                solar_to_load.append(stl)

                # Solar to Battery or Battery to Load
                if surplus > 0:
                    # Charge battery with surplus
                    charge_amount = min(surplus, battery_max_charge, battery_capacity - battery_soc[-1])
                    solar_to_battery.append(charge_amount)
                    battery_soc.append(battery_soc[-1] + charge_amount)
                    battery_to_load.append(0)
                    load_shedding.append(0)
                else:
                    solar_to_battery.append(0)
                    # Discharge battery to meet deficit
                    deficit = abs(surplus)
                    discharge_amount = min(deficit, battery_max_discharge, battery_soc[-1])
                    battery_to_load.append(discharge_amount)
                    battery_soc.append(battery_soc[-1] - discharge_amount)
                    remaining_deficit = deficit - discharge_amount
                    load_shedding.append(remaining_deficit if remaining_deficit > 0 else 0)

            # Create DataFrames
            timestamps = solar_df['timestamp'][:min_length]
            soc_df = pd.DataFrame({
                'timestamp': timestamps,
                'state_of_charge': battery_soc[1:]  # Exclude initial SoC
            })

            energy_balance_df = pd.DataFrame({
                'timestamp': timestamps,
                'solar_output': solar_output,
                'load_consumption': load_consumption,
                'state_of_charge': soc_df['state_of_charge'],
                'solar_to_load': solar_to_load,
                'solar_to_battery': solar_to_battery,
                'battery_to_load': battery_to_load,
                'load_shedding': load_shedding
            })

            # Adjust timestamps to local time using timezone offset
            timezone_offset = current_data.get('timezone', 0)  # in seconds
            energy_balance_df['timestamp'] = energy_balance_df['timestamp'] + timedelta(seconds=timezone_offset)
            soc_df['timestamp'] = soc_df['timestamp'] + timedelta(seconds=timezone_offset)
            solar_df['timestamp'] = solar_df['timestamp'] + timedelta(seconds=timezone_offset)

            # Convert 'timestamp' columns to native datetime.datetime objects
            energy_balance_df['timestamp'] = energy_balance_df['timestamp'].apply(lambda x: x.to_pydatetime() if isinstance(x, pd.Timestamp) else x)
            soc_df['timestamp'] = soc_df['timestamp'].apply(lambda x: x.to_pydatetime() if isinstance(x, pd.Timestamp) else x)
            solar_df['timestamp'] = solar_df['timestamp'].apply(lambda x: x.to_pydatetime() if isinstance(x, pd.Timestamp) else x)

            # Calculate current local time
            current_time_utc = datetime.utcnow()
            current_time_local = current_time_utc + timedelta(seconds=timezone_offset)

            # Determine if it's daytime
            sunrise_time = datetime.utcfromtimestamp(current_data['sys']['sunrise'] + timezone_offset)
            sunset_time = datetime.utcfromtimestamp(current_data['sys']['sunset'] + timezone_offset)
            is_daytime = sunrise_time <= current_time_local <= sunset_time

            # Display Live Local Time with enhanced styling
            render_live_clock(timezone_offset, is_daytime)

            # Display Last Data Refresh Time
            st.markdown(f"**Last Data Refresh (Local Time):** {(fetch_time + timedelta(seconds=timezone_offset)).strftime('%B %d, %Y %I:%M:%S %p')}")

            # Display Current Weather in Columns
            st.header("🌤️ Current Weather")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="Temperature (°C)", value=f"{current_data['main']['temp']:.2f}")
                st.metric(label="Humidity (%)", value=f"{current_data['main']['humidity']:.2f}")

            with col2:
                st.metric(label="Wind Speed (m/s)", value=f"{current_data['wind']['speed']:.2f}")
                st.metric(label="Pressure (hPa)", value=f"{current_data['main']['pressure']:.2f}")

            with col3:
                st.metric(label="Cloud Coverage (%)", value=f"{current_data['clouds']['all']:.2f}")
                current_solar_output = calculate_current_solar_output(current_data, is_daytime, solar_capacity, panel_efficiency)
                st.metric(label="Solar Output (kW)", value=f"{current_solar_output:.2f}")

            # Consolidate Weather Details Below the Metrics
            st.markdown(f"**Condition:** {current_data['weather'][0]['description'].title()}")
            # Adjust sunrise and sunset to local time using timezone offset
            sunrise_time_local = sunrise_time.strftime('%I:%M:%S %p')
            sunset_time_local = sunset_time.strftime('%I:%M:%S %p')
            st.markdown(f"**Sunrise (Local Time):** {sunrise_time_local}")
            st.markdown(f"**Sunset (Local Time):** {sunset_time_local}")

            # Plot Weather Forecast
            st.header("📈 Upcoming Weather Conditions")
            plot_weather_forecast(forecast_data)

            # Plot Solar Output Forecast
            st.header("🔆 Predicted Solar Output")
            plot_solar_output_forecast(solar_df)

            # Plot Energy Balance
            st.header("⚖️ Energy Production vs. Consumption")
            plot_energy_balance(energy_balance_df)

            # Display Load Shedding Information
            total_load_shedding = energy_balance_df['load_shedding'].sum()
            if total_load_shedding > 0:
                st.warning(f"⚠️ **Total Load Shed:** {total_load_shedding:.2f} kW over the forecast period.")
            else:
                st.success("✅ No load shedding occurred during the forecast period.")

            # -------------------------------
            # Energy Transfers Visualization
            # -------------------------------
            st.header("🔄 Energy Transfers Predictions")

            # Ensure 'timestamp' column is a list of datetime.datetime objects
            timestamps_list = list(energy_balance_df['timestamp'])

            # Determine the time range for the slider
            start_time = min(timestamps_list)
            end_time = max(timestamps_list)

            # Ensure start_time and end_time are datetime.datetime objects
            if isinstance(start_time, pd.Timestamp):
                start_time = start_time.to_pydatetime()
            if isinstance(end_time, pd.Timestamp):
                end_time = end_time.to_pydatetime()

            # Set the default value for the slider
            slider_value = start_time

            # Create the slider with improved time format
            selected_time = st.slider(
                "Select Time:",
                min_value=start_time,
                max_value=end_time,
                value=slider_value,
                step=timedelta(hours=3),  # Use 3-hour steps to match data intervals
                format="YYYY-MM-DD HH:mm:ss"  # Improved time format
            )

            # Plot energy transfers based on selected time
            plot_energy_transfers(energy_balance_df, selected_time, battery_capacity, solar_capacity, panel_efficiency)

        else:
            st.error("Failed to retrieve weather data. Please check the city name or try again later.")

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    main()
