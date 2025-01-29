import pandas as pd

# Define dtypes for each CSV
airport_dtypes = {
    'code': str,
    'location': str,
    'lng': float,
    'lat': float
}

fares_dtypes = {
    'origin': str,
    'destination': str,
    'fare': float,
    'left': int,
    'currency': str,
    'key': str
}

stays_dtypes = {
    'trip_id': int,
    'position': int,
    'location': str
}

summary_dtypes = {
    'trip_id': int,
    'total_cost': float,
    'num_flights': int,
    'route': str
}

trips_dtypes = {
    'trip_id': int,
    'position': int,
    'key': str
}

def load_data():
    """Load all CSV files with proper data types"""
    airports = pd.read_csv('data/airports.csv', dtype=airport_dtypes)
    
    fares = pd.read_csv(
        'data/fares.csv', 
        dtype=fares_dtypes,
        parse_dates=['dep_time', 'arr_time'],
        date_format='%Y-%m-%d %H:%M:%S'
    )
    
    stays = pd.read_csv(
        'data/stays.csv', 
        dtype=stays_dtypes,
        converters={'duration': pd.to_timedelta}
    )
    
    summary = pd.read_csv(
        'data/summary.csv', 
        dtype=summary_dtypes,
        parse_dates=['departure_time', 'return_time'],
        date_format='%Y-%m-%d %H:%M:%S',
        converters={'total_duration': pd.to_timedelta}
    )
    
    trips = pd.read_csv('data/trips.csv', dtype=trips_dtypes)
    
    return airports, fares, stays, summary, trips 