import pandas as pd
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

### General

def graph_metrics(dfs, xlim=None, ylims=[], ylabels=[], titles=[], figtext=None, suptitle=None):
    """Graphs multiple CloudWatch metric data frames as subplots."""
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

def get_config(config_file):
    """Loads config JSON from a given file path."""
    with open(config_file, 'r') as file:
        return json.load(file)

def json_to_pandas(filepath):
    """Loads a data frame from CloudWatch metric JSON file."""
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

def load_releases(filepath):
    """Loads service release info from a json file."""
    with open(filepath) as f:
        df = pd.DataFrame(json.load(f))

        # Timestamp conversion
        df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s', utc=True))
        
        # Cleanup
        df = df.rename(columns={'timestamp': 'Timestamps', 'serviceName': 'ServiceNames'})
    return df

def calculate_release_point_feature(df, df_releases):
    """Adds a new feature 'Release_Point' to df, based on the nearest release timestamps in df_releases."""
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
    """Adds a new feature 'Post_Releases' to df."""
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

def limit_anomalies(df, feature, anomalous_value, replace_value, min_consec):
    """
    Removes anomalies that aren't part of a consecutive group at least min_consec long\n
    df: dataframe to remove anomalies from\n
    feature: feature name of whether the data point is anomalous\n
    anomalous_value: the value an anomalous value has\n
    replace_value: the value that a non-anomalous value has\n
    min_consec: minimum size of an anomaly cluster - smaller clusters are set with replace_value
    """
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

def window_df(df, segment_len=32, slide_len=2):
    """Splits up a dataframe into overlapping segments."""
    # Removing n oldest rows so segments divide evenly into df rows
    to_remove = len(df) % slide_len
    if (to_remove > 0):
        df = df.iloc[to_remove:]
        print(f'Dropped {to_remove} row(s) from the beginning')

    segments = []
    for start_pos in range(0, len(df), slide_len):
        end_pos = start_pos + segment_len
        segment = df[start_pos:end_pos].copy()
        if len(segment) != segment_len:
            continue
        segments.append(segment)
    return (df.reset_index(), segments)

# Segments are normalized to ensure they can be stitched back together later.
def normalize_segments(segments, feature, segment_len):
    """Normalizes segments so that each segment's values taper off to 0 nearing the start/end of the segment."""
    window_rads = np.linspace(0, np.pi, segment_len)
    bell_curve = np.sin(window_rads) ** 2
    normalized_segments = []
    for i, segment in enumerate(segments):
        windowed_segment = segment.copy()
        windowed_segment[feature] *= bell_curve
        normalized_segments.append(windowed_segment)
    return normalized_segments

def sliding_chunker(df, window_len, slide_len):
    """
    Splits a data frame into a list, where each element is a slice of the data frame window_len long, sliding by slide_len each time.\n
    Based on: https://github.com/mrahtz/sanger-machine-learning-workshop/blob/master/learn_utils.py (modified for data frames)
    """
    chunks = []
    for pos in range(0, len(df), slide_len):
        chunk = df.iloc[pos:pos+window_len].copy()
        if (len(chunk) != window_len):
            continue
        chunks.append(chunk)
    return chunks

def segment_plot(segments, suptitle='Figure', rows=3, cols=3):
    """Plots subplots for windowed segments"""
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rc('font', **{ 'size': 20 })
    fig = plt.figure(figsize=(30,15))
    fig.suptitle(suptitle)
    n = 1
    for row in range(rows):
        for col in range(cols):
            axs = plt.subplot(rows, cols, n)
            axs.set_title(f'segments[{n-1}]')
            axs.tick_params(length=15, width=2)
            plt.plot(segments[n-1].Timestamps, segments[n-1].Values)
            n += 1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/k-means/segments.pdf', bbox_inches='tight')
    plt.show()
    matplotlib.rc('font', **{ 'size': old_font_size })

def normalization_plot(segments, segment_n, suptitle='Figure'):
    """Plots a bell curve, windowed segment, and normalized segment in 3 subplots."""
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rc('font', **{ 'size': 20 })
    fig, axs = plt.subplots(1,3, figsize=(30,10))
    fig.suptitle(suptitle)
    axs[0].set_title('')

    segment = segments[segment_n].reset_index()

    bell_curve = np.sin(np.linspace(0, np.pi, len(segments[segment_n]))) ** 2
    axs[0].plot(bell_curve)
    axs[0].set_title('Bell curve')

    axs[1].plot(segment.Values)
    axs[1].set_title('Segment')

    axs[2].plot(bell_curve * segment.Values)
    axs[2].set_title('normalized segment')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/k-means/normalization.pdf', bbox_inches='tight')
    plt.show()
    matplotlib.rc('font', **{ 'size': old_font_size })

def single_segment_reconstruction_plot(segments, segment_n, clusterer, segment_len):
    """Plots the original segment segments[segment_n], its normalized version, and its nearest centroid."""
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rc('font', **{ 'size': 16 })

    segment = segments[segment_n].copy()
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads) ** 2
    normalized_segment = segment.copy()
    normalized_segment.Values *= window

    nearest_centroid_idx = clusterer.predict([normalized_segment.Values])[0]
    centroids = clusterer.cluster_centers_
    nearest_centroid = np.copy(centroids[nearest_centroid_idx])

    fig = plt.figure(figsize=(15,10))
    plt.tick_params(length=15, width=2)
    plt.title('K-Means: Single Segment Reconstruction')
    plt.plot(segment.Timestamps, segment.Values, label='Original Segment')

    plt.plot(normalized_segment.Timestamps, normalized_segment.Values, label='normalized Segment')

    plt.plot(segment.Timestamps, nearest_centroid, label='Nearest Centroid')
    plt.savefig('output/k-means/single-segment-reconstruction.pdf', bbox_tight='inches')

    plt.legend()
    plt.show()
    matplotlib.rc('font', **{ 'size': old_font_size })


def reconstruct(df, feature, clusterer, segment_len, reconstruction_quantile):
    """
    Reconstructs the original graph by selecting the predicted cluster for each segment.\n
    Creates three new features - Reconstructed_Values, Reconstruction_Error, and Anomalies.
    """
    window_rads = np.linspace(0, np.pi, segment_len)
    bell_curve = np.sin(window_rads) ** 2

    slide_len = int(segment_len/2)
    test_segments = sliding_chunker(df, segment_len, slide_len)
    reconstruction = np.zeros(len(df))

    for i, segment in enumerate(test_segments):
        segment = segment.copy()
        segment[feature] *= bell_curve
        nearest_centroid_idx = clusterer.predict([segment[feature]])[0]
        centroids = clusterer.cluster_centers_
        nearest_centroid = np.copy(centroids[nearest_centroid_idx])
        pos = int(i * slide_len)
        reconstruction[pos:pos+segment_len] += nearest_centroid[0:len(reconstruction[pos:pos+segment_len])]

    # TODO: Fix reconstruction values of first and final segments dropping off rapidly.

    df['Reconstructed_Values'] = reconstruction
    df['Reconstruction_Error'] = abs(df['Reconstructed_Values'] - df.Values)
    
    # Anomalies defined by highest reconstruction errors
    df['Anomalies'] = (
        df['Reconstruction_Error'] > df['Reconstruction_Error'].quantile(reconstruction_quantile)
    ).astype(int)

    return df

def reconstruction_plot(df, start=None, end=None, suptitle='Reconstruction'):
    x = None
    if (start != None and end != None):
        df = df[start:end]

    anomalies = df.loc[df.Anomalies == 1]

    fig, axs = plt.subplots(4,1, figsize=(30,15))
    fig.suptitle(suptitle)
    axs[0].plot(df.Timestamps, df.Values, label='Original Values')
    axs[0].plot(anomalies.Timestamps, anomalies.Values, 'o', label='Anomalies')
    axs[0].legend(loc=2)

    axs[1].plot(df.Timestamps, df.Reconstructed_Values, color='yellow', label='Reconstructed Values')
    axs[1].legend(loc=2)

    axs[2].plot(df.Timestamps, df.Reconstruction_Error, color='red', label='Reconstruction Error')
    axs[2].legend(loc=2)

    axs[3].plot(df.Timestamps, df.Values, label='Original Values')
    axs[3].plot(df.Timestamps, df.Reconstructed_Values, color='yellow', label='Reconstructed Values')
    axs[3].legend(loc=2)

    plt.savefig('output/k-means/full-reconstruction.pdf')
    plt.show()

def anomaly_plot(df, anomaly_feature, title='Figure'):
    anomalies = df.loc[(df[anomaly_feature] == 1)]
    plt.title(title)
    plt.plot(df.Timestamps, df.Values)
    plt.plot(anomalies.Timestamps, anomalies.Values, 'o', color='red')
    plt.show()

def anomaly_plot_with_releases(df, metric_name, post_release_threshold, anomaly_feature='Anomalies', title='Figure'):
    import numpy.ma as ma

    values_inside_release_threshold = df.loc[df.Post_Release == 1]
    values_outside_release_threshold = df.loc[df.Post_Release == 0]
    release_points = df.loc[df.Release_Point == 1]

    # Anomalies not soon after a release
    anomalies = df.loc[(df[anomaly_feature] == 1) & (df.Post_Release == 0)]

    # Anomalies soon after a release
    post_release_anomalies = df.loc[(df[anomaly_feature] == 1) & (df.Post_Release == 1)]

    from matplotlib.pyplot import figure
    plt.figure(figsize=(30,15))
    plt.xlabel("Timestamps")
    plt.ylabel(metric_name)


    # Not close to release
    mc = ma.array(df.Values.values)

    mc[(df.Post_Release == 1) | (df.Gap)] = ma.masked
    plt.plot(df.Timestamps.values, mc, color='silver')

    # Close to release
    mc = ma.array(df.Values.values)
    mc[(df.Post_Release == 0) | (df.Gap)] = ma.masked
    plt.plot(df.Timestamps.values, mc, color='dimgrey')

    plt.plot(anomalies.Timestamps, anomalies.Values, 'o', color='blue')
    plt.plot(post_release_anomalies.Timestamps, post_release_anomalies.Values, 's', color='red')

    # Release date lines
    for release in release_points.Timestamps:
        plt.axvline(release, color='purple')

    plt.title(title)
    plt.legend([
        f'Values within release threshold ({post_release_threshold})',
        f'Values outside release threshold',
        'Anomalies outside the release threshold',
        'Anomalies within the release threshold',
        'Release points'
    ], loc=2)

    # plt.ylim(0, 2000)

    # # Output .png & .pdf files
    # import os

    # savedir = f'{os.getcwd()}\\output\\k-means\\{SERVICE}\\{LAMBDA}'
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # plt.savefig(f'{savedir}\\{METRIC}.pdf', transparent=True, bbox_tight="inches")
    # plt.savefig(f'{savedir}\\{METRIC}.png')

    plt.show()

    print(f'Post-release anomalies: {len(post_release_anomalies)} of {len(anomalies)} total values anomalies')