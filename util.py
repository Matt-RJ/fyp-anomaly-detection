import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

### Isolation Forest

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

# Adds a new feature 'Release_Point' to df, based on the nearest release timestamps in df_releases
def calculate_release_point_feature(df, df_releases):
    df['Release_Point'] = 0
    if (type(df_releases) == None):
        print('No releases - All release points set to 0')
        return df

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

    # Adding Release_Point feature by nearest timestamp
    for i, df_row in df_releases.iterrows():
        closest = pd.Timedelta.max
        for j, release_row in df.iterrows():
            timedelta = abs(df_row.Timestamps - release_row.Timestamps)
            if (timedelta < closest):
                closest = timedelta
            else:
                df.loc[j, 'Release_Point'] = 1
                break
    return df

def calculate_post_release_feature(df, post_release_timedelta):
    if 'Release_Point' not in df:
        print('Missing \'Release_Point\' feature. Run util.calculate_release_point_feature() first (df not modified).')
        return df
    df['Post_Release'] = 0
    release_points = df.loc[df['Release_Point'] == 1]
    post_releases = []
    for i, row in release_points.iterrows():
        j = -1
        while (i+j < len(df)-1):
            j += 1
            if (j != 0 and df.loc[i+j, 'Release_Point'] == 1): # Next release point found
                break
            elif ((df.loc[i+j, 'Timestamps'] - row.Timestamps) <= post_release_timedelta): # Within post release treshold
                post_releases.append(i+j)
    df.loc[post_releases, 'Post_Release'] = 1
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
                # print(f'Dropping {len(group)} potential {"anomaly" if len(group) == 1 else "anomalies"}')
                non_anomalies.extend(group)
            potential_anomalies = [j for j in potential_anomalies if j not in group]
            group = []

    df.loc[non_anomalies, feature] = replace_value
    print(f'Anomalies dropped for feature "{feature}": {len(non_anomalies)}')
    return df


### K-Means
# Based on http://amid.fish/anomaly-detection-with-k-means-clustering, modified for this project.

# Splits up a dataframe into overlapping segments
def window_df(df, segment_len=32, slide_len=2):
    # Removing n oldest rows so segments divide evenly into df rows
    to_remove = len(df) % segment_len
    if (to_remove > 0):
        df = df.iloc[to_remove:]
        print(f'Dropped {to_remove} row(s) from the beginning')

    segments = []
    for start_pos in range(0, len(df), slide_len):
        end_pos = start_pos + segment_len
        segment = df[start_pos:end_pos].copy()
        if len(segment) != segment_len: #
            continue
        segments.append(segment)
    return (df, segments)

# Segments are normalized to ensure they can be stitched back together later.
def normalize_segments(segments, feature, segment_len):
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads) ** 2
    windowed_segments = []
    for i, segment in enumerate(segments):
        windowed_segment = segment.copy()[feature] * window
        # windowed_segment[feature] = windowed_segment[feature] * window
        windowed_segments.append(windowed_segment)
    return windowed_segments

# Src: https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py (modified for data frames)
# Splits a data frame into a list, where each element is a slice of the data frame window_len long, sliding by slide_len each time.
def sliding_chunker(df, window_len, slide_len):
    chunks = []
    for pos in range(0, len(df), slide_len):
        chunk = df.iloc[pos:pos+window_len].copy()
        if (len(chunk) != window_len):
            continue
        chunks.append(chunk)
    return chunks
