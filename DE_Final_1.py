import time
import pymongo
import pandas as pd
from polygon import RESTClient
from datetime import datetime
from pycaret.regression import setup, compare_models, pull, save_model
from sklearn.linear_model import LinearRegression
import numpy as np

# Forex pairs to track
forexList = [("EUR", "USD"), ("GBP", "CHF"), ("USD", "CAD"), ("EUR", "CHF"), ("EUR", "CAD"),
               ("GBP", "EUR"), ("GBP", "USD"), ("GBP", "CAD"), ("USD", "CHF"), ("USD", "JPY")]

# Polygon REST client setup
client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")

# Function to fetch forex data from Polygon API
def get_currency_exchange_rate(currency_pair):
    base_curr, quote_curr = currency_pair
    response = client.get_real_time_currency_conversion(base_curr, quote_curr, precision=7)
    timestamp = datetime.utcfromtimestamp(response.last.timestamp / 1000.0)
    if quote_curr == "JPY":
        exchange_rate = response.converted / 1000  # Adjust for JPY's higher value
    else:
        exchange_rate = response.converted
    currency_format = f"{base_curr},{quote_curr}"
    return currency_format, timestamp, exchange_rate

# MongoDB setup
client_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
database_name = "Forex_Analysis"
collection_EURUSD = "collection_EURUSD"
collection_GBPCHF = "collection_GBPCHF"
collection_USDCAD = "collection_USDCAD"
collection_EURCHF = "collection_EURCHF"
collection_EURCAD = "collection_EURCAD"
collection_GBPEUR = "collection_GBPEUR"
collection_GBPUSD = "collection_GBPUSD"
collection_GBPCAD = "collection_GBPCAD"
collection_USDCHF = "collection_USDCHF"
collection_USDJPY = "collection_USDJPY"


collection_EURUSD_stats = "collection_EURUSD_stats"
collection_GBPCHF_stats  = "collection_GBPCHF_stats"
collection_USDCAD_stats  = "collection_USDCAD_stats"
collection_EURCHF_stats  = "collection_EURCHF_stats"
collection_EURCAD_stats  = "collection_EURCAD_stats"
collection_GBPEUR_stats  = "collection_GBPEUR_stats"
collection_GBPUSD_stats  = "collection_GBPUSD_stats"
collection_GBPCAD_stats  = "collection_GBPCAD_stats"
collection_USDCHF_stats  = "collection_USDCHF_stats"
collection_USDJPY_stats = "collection_USDJPY_stats"
 
# Function to save foreign exchange data and perform statistical analysis
def store_and_analyze_forex_data(col_name, currency_pair, timestamp, exchange_rate):
    database = client_mongo[database_name]
    forex_collection = database[col_name]
    record = {
        "currency_pair": currency_pair,
        "exchange_timestamp": timestamp,
        "exchange_rate": exchange_rate,
        "recorded_at": datetime.now()
    }
    forex_collection.insert_one(record)

    # Gather data for statistical analysis
    rates = [record['exchange_rate'] for record in forex_collection.find()]
    highest_rate = max(rates)
    lowest_rate = min(rates)
    average_rate = sum(rates) / len(rates)
    volatility = (highest_rate - lowest_rate) / average_rate

    return highest_rate, lowest_rate, average_rate, volatility

# Function to record calculated statistics of forex data into a MongoDB collection
def record_forex_statistics(stats_collection, rate_timestamp, high, low, average, volatility, fractal_dim, euro_usd_corr, usd_jpy_corr):
    database = client_mongo[database_name]
    target_collection = database[stats_collection]
    statistic_record = {
        "rate_timestamp": rate_timestamp,
        "recorded_timestamp": datetime.now(),
        "highest_value": high,
        "lowest_value": low,
        "average_value": average,
        "volatility_index": volatility,
        "fractal_dimension": fractal_dim,
        "correlation_EURUSD": euro_usd_corr,
        "correlation_USDJPY": usd_jpy_corr
    }
    target_collection.insert_one(statistic_record)

# Function to generate Keltner Bands based on mean and volatility
def generate_keltner_channels(average_value, volatility):
    upper_band = [average_value + index * 0.025 * volatility for index in range(1, 101)]
    lower_band = [average_value - index * 0.025 * volatility for index in range(1, 101)]
    return upper_band, lower_band

# Function to compute the fractal dimension based on price movements and Keltner Bands
def compute_fractal_dimension(price_list, upper_band, lower_band):
    if not price_list or not upper_band or not lower_band:
        return None

    cross_count = 0

    for position in range(1, len(price_list)):
        previous_price, current_price = price_list[position-1], price_list[position]

        # Check price crossing against Keltner Bands
        for band_index in range(len(upper_band)-1):
            crossed_above_upper = previous_price <= upper_band[band_index] and current_price > upper_band[band_index+1]
            crossed_below_upper = previous_price > upper_band[band_index+1] and current_price <= upper_band[band_index]
            crossed_below_lower = previous_price >= lower_band[band_index] and current_price < lower_band[band_index+1]
            crossed_above_lower = previous_price < lower_band[band_index+1] and current_price >= lower_band[band_index]

            # Sum up all crossings
            cross_count += (crossed_above_upper + crossed_below_upper + crossed_below_lower + crossed_above_lower)

    overall_price_range = max(price_list) - min(price_list)
    if overall_price_range == 0:
        return 0

    fractal_dimension = cross_count / overall_price_range
    return fractal_dimension         

import pandas as pd
import pymongo
from pycaret.regression import setup, compare_models, pull, save_model

def conduct_pycaret_analysis():
    # Access the database and retrieve data from the specified MongoDB collection
    database = client_mongo[database_name]
    analytics_collection = database["synth_pair_stats"]
    raw_data = list(analytics_collection.find())
    
    # Transform the data into a pandas DataFrame and remove the MongoDB specific '_id' field
    analysis_df = pd.DataFrame(raw_data)
    analysis_df.drop('_id', axis=1, inplace=True)
    
    # Initialize the PyCaret environment for regression analysis
    setup(data=analysis_df, target='mean_value', session_id=123, silent=True, verbose=False)
    
    # Evaluate various models and select the best performing model based on Mean Absolute Error (MAE)
    optimal_model = compare_models(sort='MAE', n_select=1)
    
    # Persist the optimal model to disk for later use or deployment
    model_alias = "optimal_forex_model"
    save_model(optimal_model, model_alias)
    
    # Extract performance metrics of the best model
    performance_metrics = pull()
    minimal_mae = performance_metrics.iloc[0]['MAE']
    
    print(f"Optimal model '{model_alias}' has been saved with an MAE of: {minimal_mae}")
    
    # Return the MAE for external use or further analysis
    return minimal_mae

# Define the collections for stats
collections_stats = {
    "EURUSD": "collection_EURUSD_stats",
    "GBPCHF": "collection_GBPCHF_stats",
    "USDCAD": "collection_USDCAD_stats",
    "EURCHF": "collection_EURCHF_stats",
    "EURCAD": "collection_EURCAD_stats",
    "GBPEUR": "collection_GBPEUR_stats",
    "GBPUSD": "collection_GBPUSD_stats",
    "GBPCAD": "collection_GBPCAD_stats",
    "USDCHF": "collection_USDCHF_stats",
    "USDJPY": "collection_USDJPY_stats"
}

import pandas as pd
import pymongo
from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_absolute_error

def assess_model_accuracy_per_currency(model_identifier):
    database = client_mongo[database_name]
    accuracy_results = {}
    
    # Retrieve the pre-trained model
    trained_model = load_model(model_identifier)
    
    for currency_pair in collections_stats:
        stats_collection = collections_stats[currency_pair]
        forex_data_collection = database[stats_collection]
        
        # Retrieve the most recent 20 entries from the collection
        recent_entries = list(forex_data_collection.find().sort('_id', pymongo.DESCENDING).limit(20))
        data_frame = pd.DataFrame(recent_entries)
        
        if not data_frame.empty:
            # Clean the DataFrame by removing irrelevant columns
            data_frame.drop(['_id', 'fx_timestamp', 'entry_timestamp'], axis=1, inplace=True)

            # Isolate features and the target variable from the DataFrame
            input_features = data_frame.drop('mean_value', axis=1)
            actual_values = data_frame['mean_value']

            # Make predictions with the loaded model
            prediction_results = predict_model(trained_model, data=input_features)

            # Insert prediction results back into the DataFrame
            data_frame['predicted_values'] = prediction_results['Label']

            # Compute MAE between actual and predicted values
            mae = mean_absolute_error(actual_values, data_frame['predicted_values'])
            accuracy_results[currency_pair] = mae
        else:
            accuracy_results[currency_pair] = None  # Indicates no data was found for analysis

    return accuracy_results

# Function to fetch the last 20 mean values from each collection
def fetch_last_20_mean_values(collection_name):
    db = client_mongo[database_name]
    collection = db[collection_name]
    data_points = collection.find().sort("entry_timestamp", -1).limit(20)
    return [data['mean_value'] for data in data_points]

# Function to fetch the last 20 entries from a collection
def fetch_last_20_entries(collection_name, lim):
    db = client_mongo[database_name]
    collection = db[collection_name]
    return list(collection.find().sort("entry_timestamp", -1).limit(lim))

def classify_currency_pair(data):
    volatility = np.std([entry['rate'] for entry in data])
    if volatility < 0.005:
        return 'FORECASTABLE'
    elif volatility < 0.010:
        return 'UNDEFINED'
    else:
        return 'NON-FORECASTABLE'
    
def decide_position(data):
    if not data:
        return 'N/A'
    model = LinearRegression()
    x = np.array(range(len(data))).reshape(-1, 1)
    y = np.array([entry['rate'] for entry in data]).reshape(-1, 1)
    model.fit(x, y)
    slope = model.coef_[0][0]
    return 'long' if slope > 0 else 'short'

def create_synth_pair(cycle):
    if cycle == 40:
        lim = 20
    else:
        lim = 30

    # Fetch data from each collection
    eurusd_data = fetch_last_20_entries(collections_stats["EURUSD"], lim)
    gbpchf_data = fetch_last_20_entries(collections_stats["GBPCHF"], lim)
    usdcad_data = fetch_last_20_entries(collections_stats["USDCAD"], lim)

    # Prepare the new collection for aggregated data
    synth_collection_name = "synth_pair_stats"
    db = client_mongo[database_name]

    # Check if the collection exists
    if synth_collection_name in db.list_collection_names():
        # Drop the collection if it exists
        db[synth_collection_name].drop()
        print(f"Dropped existing collection: {synth_collection_name}")

    # Create the collection again
    synth_collection = db[synth_collection_name]
    print(f"Created new collection: {synth_collection_name}")

    # Calculate averages and insert into new collection
    for i in range(lim):
        avg_max = (eurusd_data[i]['max_value'] + gbpchf_data[i]['max_value'] + usdcad_data[i]['max_value']) / 3
        avg_min = (eurusd_data[i]['min_value'] + gbpchf_data[i]['min_value'] + usdcad_data[i]['min_value']) / 3
        avg_mean = (eurusd_data[i]['mean_value'] + gbpchf_data[i]['mean_value'] + usdcad_data[i]['mean_value']) / 3
        avg_vol = (eurusd_data[i]['vol_value'] + gbpchf_data[i]['vol_value'] + usdcad_data[i]['vol_value']) / 3
        avg_fd = (eurusd_data[i]['fd_value'] + gbpchf_data[i]['fd_value'] + usdcad_data[i]['fd_value']) / 3

        # Inserting aggregated data into the new collection
        synth_collection.insert_one({
            "max_value": avg_max,
            "min_value": avg_min,
            "mean_value": avg_mean,
            "vol_value": avg_vol,
            "fd_value": avg_fd
        })

    print("Aggregated data has been successfully stored in the collection", synth_collection_name)

def extract_currency_pair_name(collection_name):
    # Extracts the currency pair name from the Collection string
    collection_prefix = "collection_"
    stats_suffix = "_stats"
    start = collection_name.find(collection_prefix) + len(collection_prefix)
    end = collection_name.rfind(stats_suffix)
    return collection_name[start:end]

def transform_mae_results(mae_results):
    transformed_results = {}
    for collection, value in mae_results.items():
        # Convert the Collection object to string
        collection_str = str(collection)
        # Extract the currency pair name
        currency_pair = extract_currency_pair_name(collection_str)
        # Map the extracted name to its corresponding value
        transformed_results[currency_pair] = value
    return transformed_results

def execute_ls_strategy(data_gbpusd, data_usdjpy, cycle):
    if cycle not in [5, 6, 7]:  # Only run at specified hours
        return None
    
    position_gbpusd = decide_position(data_gbpusd)
    position_usdjpy = decide_position(data_usdjpy)
    
    # If positions are opposite, execute L/S
    if position_gbpusd == 'long' and position_usdjpy == 'short':
        # Calculate P/L: Assuming each position involves an investment of $100
        entry_price_gbpusd = data_gbpusd[-1]['rate']
        entry_price_usdjpy = data_usdjpy[-1]['rate'] * 100  # Adjusted ratio
        
        # Store initial prices for P/L calculation later
        if cycle == 5:
            global initial_prices
            initial_prices = {'GBPUSD': entry_price_gbpusd, 'USDJPY': entry_price_usdjpy}
        
        # Close positions and calculate P/L at hour 8
        if cycle == 8:
            closing_price_gbpusd = data_gbpusd[-1]['rate']
            closing_price_usdjpy = data_usdjpy[-1]['rate'] * 100  # Adjusted ratio
            pl_gbpusd = (closing_price_gbpusd - initial_prices['GBPUSD']) * 100
            pl_usdjpy = (initial_prices['USDJPY'] - closing_price_usdjpy) * 100
            total_pl = pl_gbpusd + pl_usdjpy
            return total_pl
    
    return None


# Main
def main():
    import os
    import numpy as np  # Ensure NumPy is imported if not already at the top of your script

    keltner_prev = {}
    for baseCurrency, quoteCurrency in forexList:
        pair_key = f"{baseCurrency}{quoteCurrency}"
        keltner_prev[pair_key] = {'upper': None, 'lower': None}

    for i in range(1, 51):  # Assuming 20 cycles; adjust as necessary
        print(f"Processing cycle {i}/50")
        
        results_df = pd.DataFrame(columns=['Cycle', 'Pair', 'Classification', 'Position'])

        data = {}
        stats = {}
        for baseCurrency, quoteCurrency in forexList:
            pair_key = f"{baseCurrency}{quoteCurrency}"
            data[pair_key] = []
            stats[pair_key] = {'max': None, 'min': None, 'mean': None, 'vol': None, 'fd_val': 0.0}

        for m in range(360):  # Fetch data every second for 6 minutes
            start = time.time()
            for baseCurrency, quoteCurrency in forexList:
                if (baseCurrency, quoteCurrency) in [("GBP", "USD"), ("USD", "JPY")]:  # Adjust data fetching for specific pairs
                    pair, fx_timestamp, rate = fetch_adjusted_forex_data((baseCurrency, quoteCurrency))
                else:
                    pair, fx_timestamp, rate = get_currency_exchange_rate((baseCurrency, quoteCurrency))
                pair_key = f"{baseCurrency}{quoteCurrency}"
                collection = "collection_" + pair_key
                print(f"pair: {pair}, fx_timestamp: {fx_timestamp}, rate = {rate}")
                stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'] = store_and_analyze_forex_data(collection, pair, fx_timestamp, rate)
                data[pair_key].append(rate)
                
            end = time.time()
            elapsed = end - start 
            if elapsed < 1:
                time.sleep(1 - elapsed)
            else:
                print("Warning: iteration took more than 1 sec")

        # After 6 minutes, calculate Keltner Bands and FD, then store all data in SQL database
        for pair_key in data.keys():
            if i > 1:
                stats[pair_key]['fd_val'] = compute_fractal_dimension(data[pair_key], keltner_prev[pair_key]['upper'], keltner_prev[pair_key]['lower'])
            else:
                stats[pair_key]['fd_val'] = 0.0

            collection = "collection_" + pair_key + "_stats"

            if i <= 20:
                record_forex_statistics(collection, fx_timestamp, stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'], stats[pair_key]['fd_val'], 0, 0)
            else:
                record_forex_statistics(collection, fx_timestamp, stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'], stats[pair_key]['fd_val'], correlations[pair_key]["corr_EURUSD"], correlations[pair_key]["corr_USDJPY"])
            keltner_prev[pair_key]['upper'], keltner_prev[pair_key]['lower'] = generate_keltner_channels(stats[pair_key]['mean'], stats[pair_key]['vol'])

        # Execute the L/S strategy at specified cycles
        if i in [5, 6, 7, 8]:
            pl = execute_ls_strategy(fetch_last_20_entries('collection_GBPUSD', 20), fetch_last_20_entries('collection_USDJPY', 20), i)
            if pl is not None:
                print(f"Total Profit/Loss for cycle {i}: ${pl:.2f}")

        # Clear MongoDB collection for next 6 minutes
        db = client_mongo[database_name]
        collections = [collection_EURUSD, collection_GBPCHF, collection_USDCAD, collection_EURCHF, collection_EURCAD, collection_GBPEUR, collection_GBPUSD, collection_GBPCAD, collection_USDCHF, collection_USDJPY]
        for collection_name in collections:
            collection = db[collection_name]
            collection.delete_many({})

    if i in [10, 20]:
        print(f"Running PyCaret experiments for cycle {i}")

        create_synth_pair(i)
        conduct_pycaret_analysis()

        # Example usage
        model_name = 'best_model'  # Adjust based on your actual saved model's name
        mae_results = assess_model_accuracy_per_currency(model_name)
        print(mae_results)

        transformed_results = transform_mae_results(mae_results)

        # Sort results by MAE to determine forecastability
        sorted_results = sorted(transformed_results.items(), key=lambda x: x[1])
        
        sorted_dict = {pair: value for pair, value in sorted_results}

        forecastability = {
            sorted_results[0][0]: 'FORECASTABLE',
            sorted_results[1][0]: 'FORECASTABLE',
            sorted_results[2][0]: 'FORECASTABLE',
            sorted_results[3][0]: 'UNDEFINED',
            sorted_results[4][0]: 'UNDEFINED',
            sorted_results[5][0]: 'UNDEFINED',
            sorted_results[6][0]: 'UNDEFINED',
            sorted_results[7][0]: 'NON FORECASTABLE',
            sorted_results[8][0]: 'NON FORECASTABLE',
            sorted_results[9][0]: 'NON FORECASTABLE'
        }

        # Append results to DataFrame including MAE values
        for pair, classification in forecastability.items():
            new_row = pd.DataFrame({
                'Cycle': [i],
                'Pair': [pair],
                'MAE': [sorted_dict[pair]],  # Include the MAE value from the results
                'Classification': [classification]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Check if the file exists to determine if headers should be written
        file_exists = os.path.isfile('classification_results.csv')

        # Save the DataFrame to CSV, append if file exists, write header only if file does not exist
        results_df.to_csv('classification_results.csv', mode='a', header=not file_exists, index=False)
        data = {}
        stats = {}
        
        # Initialize the structures for each pair
        for baseCurrency, quoteCurrency in forexList:
            pair_key = f"{baseCurrency}{quoteCurrency}"
            data[pair_key] = []
            stats[pair_key] = {'max': None, 'min': None, 'mean': None, 'vol': None, 'fd_val': 0.0}

        for m in range(360):  # Fetch data every second for 6 minutes
            start = time.time()
            for baseCurrency, quoteCurrency in forexList:
                first_fx_timestamps = {}
                pair, fx_timestamp, rate = get_currency_exchange_rate((baseCurrency, quoteCurrency))

                pair_key = f"{baseCurrency}{quoteCurrency}"
                collection = "collection_" + pair_key
                # If the pair key is not in the dictionary, add the timestamp
                if pair_key not in first_fx_timestamps:
                    first_fx_timestamps[pair_key] = fx_timestamp

                print(f"pair: {pair}, fx_timestamp: {fx_timestamp}, rate = {rate}")
                stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'] = store_and_analyze_forex_data(collection, pair, fx_timestamp, rate)
                data[pair_key].append(rate)
                
            end = time.time()
            elapsed = end - start 
            if elapsed < 1:
                time.sleep(1 - elapsed)
            else:
                print("Warning: iteration took more than 1 sec")
        
        if i > 20:
            # Fetch data and prepare DataFrame
            stats_data = {}
            for pair, coll_name in collections_stats.items():
                stats_data[pair] = fetch_last_20_mean_values(coll_name)

            df = pd.DataFrame(stats_data)

            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            # Organize correlations in a nested dictionary
            correlations = {}
            for pair in df.columns:
                correlations[pair] = {
                    "corr_EURUSD": correlation_matrix.at[pair, "EURUSD"],
                    "corr_USDJPY": correlation_matrix.at[pair, "USDJPY"]
                }

        # After 6 minutes, calculate Keltner Bands and FD, then store all data in SQL database
        for pair_key in data.keys():
            if i > 1:
                stats[pair_key]['fd_val'] = compute_fractal_dimension(data[pair_key], keltner_prev[pair_key]['upper'], keltner_prev[pair_key]['lower'])
            else:
                stats[pair_key]['fd_val'] = 0.0

            collection = "collection_" + pair_key + "_stats"

            if i <= 20:
                record_forex_statistics(collection, fx_timestamp, stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'], stats[pair_key]['fd_val'], 0, 0)
            else:
                record_forex_statistics(collection, fx_timestamp, stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'], stats[pair_key]['fd_val'], correlations[pair_key]["corr_EURUSD"], correlations[pair_key]["corr_USDJPY"])
            keltner_prev[pair_key]['upper'], keltner_prev[pair_key]['lower'] = generate_keltner_channels(stats[pair_key]['mean'], stats[pair_key]['vol'])

        # Clear MongoDB collection for next 6 minutes
        db = client_mongo[database_name]
        collections = [collection_EURUSD, collection_GBPCHF, collection_USDCAD, collection_EURCHF, collection_EURCAD, collection_GBPEUR, collection_GBPUSD, collection_GBPCAD, collection_USDCHF, collection_USDJPY]
        for collection_name in collections:
            collection = db[collection_name]
            collection.delete_many({})
    

if __name__ == "__main__":
    main()