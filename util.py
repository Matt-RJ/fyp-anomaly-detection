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
    for i, release_timestamp in enumerate(df_releases.Timestamps):
        if (release_timestamp < earliest_timestamp or release_timestamp > latest_timestamp):
            out_of_scope_release_indices.append(i)

    df_releases = df_releases.drop(out_of_scope_release_indices)
    print(f'Dropped {len(out_of_scope_release_indices)} out of scope release date(s)')

    # Adding PostRelease feature, where the 
    for release_date in df_releases.Timestamps:
        closest = pd.Timedelta.max
        for i, row in enumerate(df.Timestamps):
            timedelta = abs(release_date - row)
            if (timedelta < closest):
                closest = timedelta
            else:
                df.loc[i, 'PostRelease'] = 1
                break
    return df