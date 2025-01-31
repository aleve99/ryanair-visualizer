import dash
from dash import dcc, html, dash_table
import pandas as pd
from datetime import date, timedelta
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from data_loader import load_data
import numpy as np
from dash.exceptions import PreventUpdate
from functools import lru_cache

# Constants
CARD_COLORS = {
    'flight': {'bg': '#e3f2fd', 'border': '#2196f3'},
    'stay': {'bg': '#f1f8e9', 'border': '#4caf50'},
    'marker': {
        'default': '#7f8c8d',
        'stay': '#2ecc71',
        'route': '#3498db',
        'selected': '#f4d006'
    }
}

MAP_SETTINGS = {
    'style': 'basic',
    'curve_smoothness': 30,
    'curve_height_factor': 0.06,
    'marker_sizes': {'default': 8, 'highlighted': 12}
}

TABLE_SETTINGS = {
    'page_size': 14,
    'table_height': '51vh'
}

PAGE_SIZE = 14

def get_one_way_link(from_date: date, origin: str, destination: str) -> str:     
    return "https://www.ryanair.com/en/us/trip/flights/select?" + "&".join([
        "adults=1",
        "teens=0",
        "children=0",
        "infants=0",
        f"dateOut={from_date}",
        "isConnectedFlight=false",
        "discount=0",
        "promoCode=",
        "isReturn=false",
        f"originIata={origin}",
        f"destinationIata={destination}"
    ])

def format_timedelta(td: timedelta) -> str:
    """Convert timedelta to readable string"""
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    return f"{days} days {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"

def calculate_duration_days(row):
    """Calculate duration in days"""
    return (row['return_time'] - row['departure_time']).days

def create_curved_path(start_lon, start_lat, end_lon, end_lat):
    """Create a smooth curved path between two points"""
    t = np.linspace(0, 1, MAP_SETTINGS['curve_smoothness'])
    
    mid_lon = (start_lon + end_lon) / 2
    mid_lat = (start_lat + end_lat) / 2
    
    curve_height = MAP_SETTINGS['curve_height_factor'] * abs(end_lat - start_lat)
    mid_lat += curve_height
    
    lons = (1-t)**2 * start_lon + 2*(1-t)*t * mid_lon + t**2 * end_lon
    lats = (1-t)**2 * start_lat + 2*(1-t)*t * mid_lat + t**2 * end_lat
    
    return lons, lats

def calculate_map_bounds(selected_trip, selected_airports, airports_df):
    """Calculate map bounds and zoom level"""
    points_to_include = []
    
    # Include route points if trip is selected
    if selected_trip is not None:
        route_airports = selected_trip['route'].split('-')
        points_to_include.extend(route_airports)
    
    # Include selected airports
    if selected_airports:
        points_to_include.extend(selected_airports)
    
    # If no points to include, return default Europe view
    if not points_to_include:
        return 48.8566, 2.3522, 4  # Paris-centered default view
    
    # Get unique airports
    points_to_include = list(set(points_to_include))
    
    # Get coordinates for all points
    coords = airports_df[airports_df['code'].isin(points_to_include)]
    
    lat_min, lat_max = coords['lat'].min(), coords['lat'].max()
    lon_min, lon_max = coords['lng'].min(), coords['lng'].max()
    
    # Add padding (10%)
    padding = 0.1
    lat_pad = (lat_max - lat_min) * padding
    lon_pad = (lon_max - lon_min) * padding
    
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    # Calculate zoom level to fit all points
    lon_zoom = np.log2(360 / (lon_max - lon_min + 2 * lon_pad)) + 1
    lat_zoom = np.log2(180 / (lat_max - lat_min + 2 * lat_pad)) + 1
    zoom = min(lon_zoom, lat_zoom) - 0.5  # Slightly zoom out for better view
    
    return center_lat, center_lon, zoom

def create_map_figure(selected_trip, selected_airports=[], current_view=None):
    """Create map figure for a given trip with optimizations"""
    # Create base traces that will be modified
    base_traces = []
    
    # Pre-calculate marker properties for each state
    marker_props = {
        'default': dict(size=MAP_SETTINGS['marker_sizes']['default'], color=CARD_COLORS['marker']['default']),
        'selected': dict(size=MAP_SETTINGS['marker_sizes']['highlighted'], color=CARD_COLORS['marker']['selected']),
        'stay': dict(size=MAP_SETTINGS['marker_sizes']['highlighted'], color=CARD_COLORS['marker']['stay']),
        'route': dict(size=MAP_SETTINGS['marker_sizes']['highlighted'], color=CARD_COLORS['marker']['route'])
    }
    
    # Pre-process trip data if available
    trip_stays = set()
    route_airports = []  # Keep as list to maintain order
    if selected_trip is not None:
        route_airports = selected_trip['route'].split('-')  # Keep as list for order
        # Get stays using index
        trip_id = selected_trip.name if hasattr(selected_trip, 'name') else selected_trip['trip_id']
        if trip_id in stays.index.get_level_values(0):
            trip_stays = set(stays.loc[trip_id]['location'].tolist())
    
    selected_airports = set(selected_airports)  # Convert to set for O(1) lookups
    route_airports_set = set(route_airports)  # For O(1) lookups
    
    # Create a single trace for all airports with different marker properties
    all_lons, all_lats = [], []
    all_texts = []
    all_customdata = []
    all_markers = []
    
    for code, coords in AIRPORT_COORDS.items():
        all_lons.append(coords['lng'])
        all_lats.append(coords['lat'])
        all_texts.append(f"{AIRPORT_LOCATIONS[code]} ({code})")
        all_customdata.append(code)
        
        # Determine marker properties based on airport state
        if code in selected_airports:
            marker = marker_props['selected']
        elif code in trip_stays:
            marker = marker_props['stay']
        elif code in route_airports_set:
            marker = marker_props['route']
        else:
            marker = marker_props['default']
        all_markers.append(marker)
    
    # Create a single trace for all airports
    base_traces.append(go.Scattermap(
        lon=all_lons,
        lat=all_lats,
        mode='markers',
        marker=dict(
            size=[m['size'] for m in all_markers],
            color=[m['color'] for m in all_markers]
        ),
        text=all_texts,
        hoverinfo='text',
        customdata=all_customdata,
        name='airports'
    ))
    
    # Add route lines if trip is selected
    if selected_trip is not None and len(route_airports) > 1:
        # Create route segments
        for i in range(len(route_airports) - 1):
            start_code = route_airports[i]
            end_code = route_airports[i + 1]
            
            start_coords = AIRPORT_COORDS[start_code]
            end_coords = AIRPORT_COORDS[end_code]
            
            lons, lats = create_curved_path(
                start_coords['lng'], start_coords['lat'],
                end_coords['lng'], end_coords['lat']
            )
            
            base_traces.append(go.Scattermap(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=2, color=CARD_COLORS['marker']['route']),
                showlegend=False,
                hoverinfo='skip',
                name='route'
            ))
    
    fig = go.Figure(data=base_traces)
    
    # Use current view if available, otherwise calculate bounds
    if current_view:
        center_lat, center_lon, zoom = current_view
    else:
        # Calculate bounds based on selected trip and/or airports
        center_lat, center_lon, zoom = calculate_map_bounds(selected_trip, selected_airports, AIRPORTS_DF)
    
    fig.update_layout(
        map=dict(
            style=MAP_SETTINGS['style'],
            zoom=zoom,
            center=dict(lat=center_lat, lon=center_lon)
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event',
        uirevision=True  # Keep the view state constant during updates
    )
    
    return fig

# Load all data - Add caching
@lru_cache(maxsize=1)
def load_cached_data():
    """Cache the data loading to avoid reloading on every refresh in debug mode"""
    return load_data()

# Replace direct load_data() call
airports, fares, stays, summary, trips = load_cached_data()

# Set indexes for faster lookups
summary.set_index('trip_id', inplace=True)
stays.set_index(['trip_id', 'position'], inplace=True)
trips.set_index(['trip_id', 'position'], inplace=True)
fares.set_index('key', inplace=True)
airports.set_index('code', inplace=True)

# Pre-calculate commonly used data
AIRPORT_LOCATIONS = airports['location'].to_dict()
AIRPORT_COORDS = airports[['lat', 'lng']].to_dict('index')
AIRPORTS_DF = airports.reset_index()  # Keep a copy with the index as a column

# Initialize the Dash app with performance settings
app = dash.Dash(
    __name__,
    title="Flight Trip Explorer",
    update_title=None,  # Disable "Updating..." title
    compress=True,  # Enable gzip compression
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

@lru_cache(maxsize=32)
def create_trip_details(trip_id):
    """Create trip details content for a given trip ID with optimizations"""
    # Get trip data efficiently using indexes
    trip_stays = stays.loc[trip_id].sort_values('position') if trip_id in stays.index.get_level_values(0) else pd.DataFrame()
    trip_flights = trips.loc[trip_id].sort_values('position')
    
    details_content = []
    
    # Pre-fetch airport locations to avoid repeated lookups
    airport_locations = {}
    all_airports = set()
    for _, flight in trip_flights.iterrows():
        flight_key = flight['key']
        flight_fare = fares.loc[flight_key]
        all_airports.add(flight_fare['origin'])
        all_airports.add(flight_fare['destination'])
    
    # Use the pre-calculated AIRPORT_LOCATIONS instead of querying the DataFrame
    airport_locations = {code: AIRPORT_LOCATIONS[code] for code in all_airports}
    
    # Create flight and stay cards
    for i in range(len(trip_flights)):
        flight_key = trip_flights.iloc[i]['key']
        flight_fare = fares.loc[flight_key]
        
        origin_name = airport_locations[flight_fare['origin']]
        dest_name = airport_locations[flight_fare['destination']]
        
        # Create flight booking link
        booking_link = get_one_way_link(
            flight_fare['dep_time'].date(),
            flight_fare['origin'],
            flight_fare['destination']
        )
        
        # Create flight card with clickable price
        details_content.extend([
            html.Div([
                html.H4(f"Flight {i+1}"),
                html.P(f"{origin_name} ({flight_fare['origin']}) → {dest_name} ({flight_fare['destination']})"),
                html.P(f"Departure: {flight_fare['dep_time']}"),
                html.P(f"Arrival: {flight_fare['arr_time']}"),
                html.P([
                    "Cost: ",
                    html.A(
                        f"${flight_fare['fare']}", 
                        href=booking_link,
                        target="_blank",
                        style={
                            'color': 'var(--flight-border)',
                            'textDecoration': 'none',
                            'fontWeight': 'bold',
                            'cursor': 'pointer'
                        }
                    )
                ])
            ], className='detail-card flight-card')
        ])
        
        # Add stay card if exists and we have stays data
        if not trip_stays.empty and i < len(trip_stays):
            stay = trip_stays.iloc[i]
            stay_location_name = AIRPORT_LOCATIONS.get(stay['location'], stay['location'])
            details_content.extend([
                html.Div([
                    html.H4(f"Stay in {stay_location_name}"),
                    html.P(f"Location: {stay['location']}"),
                    html.P(f"Duration: {stay['duration']}")
                ], className='detail-card stay-card')
            ])
    
    return details_content

@lru_cache(maxsize=128)
def get_filtered_trips(filter_airports_tuple):
    """Cache filtered trips results with optimized filtering"""
    if not filter_airports_tuple:
        return summary
    
    # Convert filter_airports to set for O(1) lookups
    filter_airports = set(filter_airports_tuple)
    
    # Use vectorized operations for filtering
    mask = summary['route'].apply(lambda x: all(airport in x for airport in filter_airports))
    return summary[mask]

def get_table_data(page=0, filter_airports=None):
    """Optimized table data retrieval"""
    # Convert filter_airports list to tuple for caching
    filter_airports_tuple = tuple(sorted(filter_airports)) if filter_airports else ()
    
    # Get filtered data using cached function
    filtered_trips = get_filtered_trips(filter_airports_tuple)
    
    # If no data matches the filter, return empty data
    if filtered_trips.empty:
        return [], 0

    # Calculate total number of rows
    total_rows = len(filtered_trips)

    # Get slice for current page
    start_idx = page * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_data = filtered_trips.iloc[start_idx:end_idx]

    # Reset index to get trip_id as a column and use pandas operations for formatting
    table_data = (
        page_data
        .reset_index()  # This brings trip_id back as a column
        [['route', 'total_cost', 'departure_time', 'return_time', 'trip_id']]
        .assign(
            departure_date=lambda df: df['departure_time'].dt.strftime('%Y-%m-%d'),
            return_date=lambda df: df['return_time'].dt.strftime('%Y-%m-%d'),
            duration_days=lambda df: (df['return_time'] - df['departure_time']).dt.days
        )
        .drop(['departure_time', 'return_time'], axis=1)
        .to_dict('records')
    )

    return table_data, total_rows

# Initial data for first page
initial_data, total_rows = get_table_data(page=0)

# Get the first trip for initial display
initial_trip_id = summary.index[0]  # Get first trip_id from index
initial_trip = summary.loc[initial_trip_id]  # Get trip data using index
initial_details = create_trip_details(initial_trip_id)
initial_summary = html.Div([
    html.H3("Trip Summary"),
    html.P(f"Total Cost: ${initial_trip['total_cost']:.2f}"),
    html.P(f"Total Duration: {format_timedelta(initial_trip['total_duration'])}"),
    html.P(f"Departure: {initial_trip['departure_time'].strftime('%Y-%m-%d %H:%M')}"),
    html.P(f"Return: {initial_trip['return_time'].strftime('%Y-%m-%d %H:%M')}")
])

# Pre-calculate initial view
initial_map = create_map_figure(initial_trip)

# Define the layout with initial values
app.layout = html.Div([
    dcc.Store(id='selected-airports', data=[]),
    dcc.Store(id='selected-trip-id', data=initial_trip_id),  # Set initial trip_id

    # Title Container
    html.Div([
        html.H1("Flight Trip Explorer", className='dashboard-title')
    ], className='title-container'),
    
    # Main Content Container
    html.Div([
        # Top Row with Map and Table
        html.Div([
            # Map Container
            html.Div([
                dcc.Store(id='keyboard-state', data={'ctrl': False}),
                html.Div(id='key-listener', style={'display': 'none'}),  # Hidden div for key events
                dcc.Graph(
                    id='route-map',
                    figure=initial_map,
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'doubleClick': False
                    },
                    clear_on_unhover=True,
                    style={'height': '100%'}
                ),
                dcc.Store(id='last-click-state', data={'airport': None, 'n_clicks': 0})
            ], className='map-box'),
            
            # Table Container
            html.Div([
                html.Div(id='filter-info', children="Showing all trips", className='filter-info'),
                dash_table.DataTable(
                    id='summary-table',
                    columns=[
                        {'name': 'Route', 'id': 'route'},
                        {'name': 'Total Cost', 'id': 'total_cost', 'type': 'numeric', 
                         'format': {'specifier': '$.2f'}},
                        {'name': 'Departure Date', 'id': 'departure_date'},
                        {'name': 'Return Date', 'id': 'return_date'},
                        {'name': 'Duration (Days)', 'id': 'duration_days', 'type': 'numeric'}
                    ],
                    data=initial_data,
                    row_selectable='single',
                    selected_rows=[0],
                    page_action='custom',
                    page_current=0,
                    page_size=PAGE_SIZE,
                    page_count=-(-total_rows // PAGE_SIZE),
                    style_table={
                        'height': TABLE_SETTINGS['table_height'],
                        'overflowY': 'auto'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '4px 8px',
                        'whiteSpace': 'normal',
                        'fontSize': '13px'
                    },
                    style_data_conditional=[{
                        'if': {'state': 'selected'},
                        'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                        'border': '1px solid rgb(0, 116, 217)',
                    }]
                )
            ], className='table-box')
        ], className='top-row-container'),
        
        # Bottom Row with Trip Details
        html.Div([
            # Centered Title at the top
            html.Div([
                html.H2(id='trip-title', children=initial_trip['route'])
            ], className='trip-title-container'),
            
            # Container for Summary and Carousel (side by side)
            html.Div([
                # Left side - Summary Card
                html.Div(id='trip-summary', children=initial_summary, className='summary-box'),
                
                # Right side - Carousel Container
                html.Div([
                    html.Button('←', id='trip-carousel-left', className='carousel-arrow'),
                    html.Div([
                        html.Div(initial_details, id='trip-details', className='trip-details-carousel')
                    ], className='trip-details-container'),
                    html.Button('→', id='trip-carousel-right', className='carousel-arrow')
                ], className='carousel-container')
            ], className='content-row-container')
        ], className='bottom-row-container')

    ], className='main-container', style={'minHeight': '100vh'})
], className='dashboard-container')

@app.callback(
    [
        Output('selected-airports', 'data'),
        Output('filter-info', 'children'),
        Output('last-click-state', 'data'),
        Output('route-map', 'clickData'),
    ],
    [Input('route-map', 'clickData')],
    [
        State('keyboard-state', 'data'),
        State('selected-airports', 'data'),
        State('last-click-state', 'data')
    ],
    prevent_initial_call=True
)
def update_selected_airports(click_data, keyboard_state, current_selection, last_click):
    # If the user clicked outside any point (no customdata), do nothing:
    if not click_data or "points" not in click_data or not click_data["points"]:
        raise PreventUpdate

    clicked_airport = click_data["points"][0]["customdata"]
    if not clicked_airport:
        raise PreventUpdate

    # Get ctrl state from keyboard state
    ctrl_pressed = keyboard_state.get('ctrl', False) if keyboard_state else False

    # Copy the current selections if not None:
    if current_selection is None:
        current_selection = []

    # We keep track of the last-clicked airport to reuse your existing logic:
    last_click = last_click or {'airport': None, 'n_clicks': 0}

    if ctrl_pressed:
        # --- CTRL + CLICK mode: Toggle that airport ---
        if clicked_airport in current_selection:
            current_selection.remove(clicked_airport)
        else:
            current_selection.append(clicked_airport)
    else:
        # --- NORMAL CLICK mode: Replace with only that airport ---
        current_selection = [clicked_airport]

    # Update last-click info
    new_click_state = {
        'airport': clicked_airport,
        'n_clicks': last_click['n_clicks'] + 1
    }

    # Create filter text to display using pre-calculated AIRPORT_LOCATIONS:
    if not current_selection:
        filter_text = "Showing all trips"
    else:
        airport_names = [
            f"{AIRPORT_LOCATIONS[code]} ({code})"
            for code in current_selection
        ]
        filter_text = "Filtered by: " + ", ".join(airport_names)

    # Return the updated selection info.
    # We set map clickData to None if you want subsequent identical clicks to register again
    return current_selection, filter_text, new_click_state, None

@app.callback(
    [Output('summary-table', 'data'),
     Output('summary-table', 'page_count'),
     Output('summary-table', 'page_current')],
    [Input('selected-airports', 'data'),
     Input('summary-table', 'page_current')],
    prevent_initial_call=True
)
def update_table_data(selected_airports, page_current):
    # Get the trigger that caused this callback
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    # If the trigger was selected_airports changing, reset page to 0
    reset_page = trigger == 'selected-airports.data'
    page_to_use = 0 if reset_page else (page_current or 0)
    
    # Get the data
    table_data, total_rows = get_table_data(
        page=page_to_use,
        filter_airports=selected_airports
    )
    page_count = max(1, -(-total_rows // PAGE_SIZE))
    
    # Only update what changed
    if trigger == 'summary-table.page_current':
        return table_data, dash.no_update, dash.no_update
    else:
        return table_data, page_count, 0 if reset_page else dash.no_update

@app.callback(
    [Output('summary-table', 'selected_rows'),
     Output('selected-trip-id', 'data')],
    [Input('summary-table', 'selected_rows'),
     Input('summary-table', 'data'),
     Input('selected-airports', 'data'),
     Input('summary-table', 'page_current')],
    [State('selected-trip-id', 'data')],
    prevent_initial_call=False
)
def maintain_selection(selected_rows, current_data, selected_airports, page_current, stored_trip_id):
    if not current_data:
        return [], None
        
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    # Handle page change explicitly
    if trigger == 'summary-table.page_current':
        if stored_trip_id is not None:
            # Find the row index in current page data
            for i, row in enumerate(current_data):
                if row['trip_id'] == stored_trip_id:
                    return [i], stored_trip_id
            # If trip not found in current page, keep trip_id but clear selection
            return [], stored_trip_id
    
    # Reset selection when filtering or when data becomes available after being empty
    if trigger == 'selected-airports.data' or (trigger == 'summary-table.data' and stored_trip_id is None):
        if current_data:  # Only if we have data
            return [0], current_data[0]['trip_id']
        return [], None
    
    # Handle explicit row selection
    if trigger == 'summary-table.selected_rows' and selected_rows:
        return selected_rows, current_data[selected_rows[0]]['trip_id']
    
    # Default case: maintain stored_trip_id if possible
    if stored_trip_id is not None and current_data:
        for i, row in enumerate(current_data):
            if row['trip_id'] == stored_trip_id:
                return [i], stored_trip_id
    
    # Fallback to first row if we have data
    if current_data:
        return [0], current_data[0]['trip_id']
    
    return [], None

@app.callback(
    [Output('route-map', 'figure'),
     Output('trip-title', 'children'),
     Output('trip-summary', 'children'),
     Output('trip-details', 'children')],
    [Input('selected-trip-id', 'data'),
     Input('selected-airports', 'data'),
     Input('summary-table', 'data')],  # Add table data as input to detect when data becomes available
    [State('route-map', 'figure')],
    prevent_initial_call=False
)
def update_display(selected_trip_id, selected_airports, table_data, current_figure):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    # Get current view if available
    current_view = None
    if current_figure:
        try:
            center = current_figure['layout']['map']['center']
            zoom = current_figure['layout']['map']['zoom']
            current_view = (center['lat'], center['lon'], zoom)
        except (KeyError, TypeError):
            current_view = None
    
    # Handle no selection, no matching trips, or empty table
    if selected_trip_id is None or not table_data:
        fig = create_map_figure(None, selected_airports or [], current_view)
        return fig, "No route found within the current selection", "", []
    
    # Find selected trip
    try:
        selected_trip = summary.loc[selected_trip_id]
    except KeyError:
        fig = create_map_figure(None, selected_airports or [], current_view)
        return fig, "Selected trip not found", "", []
    
    # Always update everything if:
    # 1. Trip ID changed
    # 2. Airports changed
    # 3. Table data changed (meaning filters changed)
    if trigger in ['selected-trip-id.data', 'selected-airports.data', 'summary-table.data']:
        fig = create_map_figure(selected_trip, selected_airports or [], current_view)
        summary_content = html.Div([
            html.H3("Trip Summary"),
            html.P(f"Total Cost: ${selected_trip['total_cost']:.2f}"),
            html.P(f"Total Duration: {format_timedelta(selected_trip['total_duration'])}"),
            html.P(f"Departure: {selected_trip['departure_time'].strftime('%Y-%m-%d %H:%M')}"),
            html.P(f"Return: {selected_trip['return_time'].strftime('%Y-%m-%d %H:%M')}")
        ])
        
        details_content = create_trip_details(selected_trip_id)
        
        return fig, selected_trip['route'], summary_content, details_content
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

app.clientside_callback(
    """
    function(left_clicks, right_clicks, current_style) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered.length) return {'transform': 'translateX(0px)'};
        
        const button_clicked = ctx.triggered[0].prop_id.split('.')[0];
        let current_transform = 0;
        
        if (current_style && current_style.transform) {
            try {
                current_transform = parseInt(current_style.transform.split('(')[1]);
            } catch(e) {
                current_transform = 0;
            }
        }
        
        const new_transform = button_clicked === 'trip-carousel-left' 
            ? Math.min(0, current_transform + 220)
            : Math.max(-1000, current_transform - 220);
            
        return {
            'transform': `translateX(${new_transform}px)`,
            'transition': 'transform 0.3s ease-in-out'
        };
    }
    """,
    Output('trip-details', 'style'),
    [Input('trip-carousel-left', 'n_clicks'),
     Input('trip-carousel-right', 'n_clicks')],
    [State('trip-details', 'style')]
)

def no_data_figure():
    """Create a map figure with all airports when no trip is selected"""
    fig = go.Figure()
    
    # Add all airports with default styling
    for code, coords in AIRPORT_COORDS.items():
        fig.add_trace(go.Scattermap(
            lon=[coords['lng']],
            lat=[coords['lat']],
            mode='markers',
            marker=dict(
                size=MAP_SETTINGS['marker_sizes']['default'],
                color=CARD_COLORS['marker']['default']
            ),
            text=f"{AIRPORT_LOCATIONS[code]} ({code})",
            hoverinfo='text',
            customdata=[code]
        ))
    
    # Set default view of Europe
    fig.update_layout(
        map=dict(
            style=MAP_SETTINGS['style'],
            zoom=4,
            center=dict(lat=48.8566, lon=2.3522)  # Center on Paris
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event'
    )
    
    return fig

# Add this to improve initial load performance
app.config.suppress_callback_exceptions = True

# Replace the interval-based callback with these two callbacks
app.clientside_callback(
    """
    function(n_clicks) {
        if (!window.keyListenersAdded) {
            window.ctrlPressed = false;
            window.keyTrigger = null;
            
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Control' && !window.ctrlPressed) {
                    window.ctrlPressed = true;
                    if (window.keyTrigger) window.keyTrigger();
                }
            });
            
            document.addEventListener('keyup', (e) => {
                if (e.key === 'Control' && window.ctrlPressed) {
                    window.ctrlPressed = false;
                    if (window.keyTrigger) window.keyTrigger();
                }
            });
            
            window.addEventListener('blur', () => {
                if (window.ctrlPressed) {
                    window.ctrlPressed = false;
                    if (window.keyTrigger) window.keyTrigger();
                }
            });
            
            window.keyListenersAdded = true;
        }
        
        // Return a dummy value to satisfy Dash
        return window.performance.now();
    }
    """,
    Output('key-listener', 'n_clicks'),
    [Input('route-map', 'figure')]  # Initialize when the map is created
)

app.clientside_callback(
    """
    function(n_clicks) {
        // Set up the trigger for keyboard state updates
        window.keyTrigger = () => {
            if (document.getElementById('key-listener')) {
                document.getElementById('key-listener').click();
            }
        };
        
        return {'ctrl': window.ctrlPressed || false};
    }
    """,
    Output('keyboard-state', 'data'),
    [Input('key-listener', 'n_clicks')]
)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8050", debug=False)