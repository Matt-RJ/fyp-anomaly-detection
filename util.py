import pandas as pd
import json
import matplotlib.pyplot as plt

# Graphs multiple CloudWatch metric data frames as subplots
def graph_metrics(dfs, xlim=None, ylims=[], ylabels=[], titles=[], figtext=None, suptitle=None):
    fig, axs = plt.subplots(len(dfs), figsize = (30,10), sharex=True)
    fig.set_figheight(15)
    if (suptitle is not None):
        fig.suptitle(suptitle, fontsize=30)
    if (xlim):
        plt.xlim(xlim)
    for i, df in enumerate(dfs):
        axs[i].plot(df["Timestamps"], df["Values"])
        axs[i].tick_params(axis='both', labelsize=20, length=10, width=2)
        if (ylims is not None):
            if (ylims[i] is not None):
                axs[i].set_ylim(ylims[i])
            
        if (ylabels[i] is not None):
            if (ylabels[i] is not None):
                axs[i].set_ylabel(ylabels[i], fontsize=30)
            
        if (titles[i] is not None):
            if (titles[i] is not None):
                axs[i].set_title(titles[i], fontsize=30)
    plt.show()

# Loads config JSON from a given file path
def get_config(config_file):
  with open(config_file, 'r') as file:
    return json.load(file)

# Loads a data frame from CloudWatch metric JSON
def json_to_pandas(filepath):
    with open(filepath) as f:
        data = json.load(f)
        dfs = {}
        
        for i, df in enumerate(data):
            # Timestamp conversion
            for j, timestamp in enumerate(data[i]["Timestamps"]):
                data[i]["Timestamps"][j] = pd.to_datetime(timestamp, utc=True)

            df = pd.DataFrame({
                "Timestamps": data[i]["Timestamps"],
                "Values": data[i]["Values"]
            })
            dfs[data[i]["Label"]] = df
    return dfs

# Loads service release info from a json file
def load_releases(filepath):
    with open(filepath) as f:
        df = pd.DataFrame(json.load(f))

        # Timestamp conversion
        df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s', utc=True))
        
        # Cleanup
        df = df.rename(columns={'timestamp': 'Timestamps', 'serviceName': 'ServiceNames'})
    return df

# Adds a new feature 'PostRelease' to df, based on the nearest release timestamps in df_releases
def calculate_postrelease_feature(df, df_releases):
    df['PostRelease'] = 0

    # Removing any releases that fall outside of the df metric time range
    earliest_timestamp = df.Timestamps.iloc[1]
    latest_timestamp = df.Timestamps.iloc[-1]
    print('Earliest in-scope release date: ', earliest_timestamp)
    print('Latest in-scope release date: ', latest_timestamp)

    out_of_scope_release_indices = [] # Release indices to be dropped
    for i, row in df_releases.iterrows():
        if (row.Timestamps < earliest_timestamp or row.Timestamps > latest_timestamp):
            out_of_scope_release_indices.append(i)

    if (len(out_of_scope_release_indices) > 0):
        df_releases = df_releases.drop(index=out_of_scope_release_indices)
        print(f'Dropped {len(out_of_scope_release_indices)} out of scope release date(s)')

    # Adding PostRelease feature by nearest timestamp
    for i, df_row in df_releases.iterrows():
        closest = pd.Timedelta.max
        for j, release_row in df.iterrows():
            timedelta = abs(df_row.Timestamps - release_row.Timestamps)
            if (timedelta < closest):
                closest = timedelta
            else:
                df.loc[j, 'PostRelease'] = 1
                break
    return df

# Removes anomalies that aren't part of a consecutive group at least min_consec long
# df - dataframe to remove anomalies from
# feature - feature name of whether the data point is anomalous
# anomalous_value - the value an anomalous value has
# replace_value - the value that a non-anomalous value has
# min_consec - minimum size of an anomaly cluster - smaller clusters are set with replace_value
def limit_anomalies(df, feature, anomalous_value, replace_value, min_consec):
    potential_anomalies = df.loc[df[feature] == anomalous_value].index.tolist()
    anomalies = []
    non_anomalies = []
    group = [] # A single cluster of neighboring anomalies
    for i, value in enumerate(potential_anomalies):
        group.append(value)
        if ((value+1) not in potential_anomalies): # Non-neighbouring anomaly found - group end found
            if (len(group) >= min_consec): # Cluster of anomalies big enough to stay
                anomalies.extend(group) 
            else: # Anomalies in group removed
                print(f'Dropping {len(group)} potential {"anomaly" if len(group) == 1 else "anomalies"}')
                non_anomalies.extend(group)
            potential_anomalies = [j for j in potential_anomalies if j not in group]
            group = []

    df.loc[non_anomalies, feature] = replace_value
    print(f'Anomalies dropped for feature "{feature}": {len(non_anomalies)}')
    return df