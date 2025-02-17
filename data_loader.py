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
        dtype=fares_dtypes
    )
    # Convert timestamp columns after loading
    fares['dep_time'] = pd.to_datetime(fares['dep_time'], unit='s')
    fares['arr_time'] = pd.to_datetime(fares['arr_time'], unit='s')
    
    stays = pd.read_csv(
        'data/stays.csv', 
        dtype=stays_dtypes,
        converters={'duration': lambda x: pd.Timedelta(seconds=float(x))}
    )
    
    summary = pd.read_csv(
        'data/summary.csv', 
        dtype=summary_dtypes
    )
    # Convert timestamp columns after loading
    summary['departure_time'] = pd.to_datetime(summary['departure_time'], unit='s')
    summary['return_time'] = pd.to_datetime(summary['return_time'], unit='s')
    summary['total_duration'] = summary['total_duration'].apply(lambda x: pd.Timedelta(seconds=float(x)))
    
    trips = pd.read_csv('data/trips.csv', dtype=trips_dtypes)
    
    return airports, fares, stays, summary, trips 