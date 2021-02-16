import pandas as pd
import json

# Loads a data frame from CloudWatch metric JSON
def json_to_pandas(filepath):
    with open(filepath) as f:
        data = json.load(f)
        dfs = []
        
        for i, df in enumerate(data):
            # Timestamp conversion
            for j, timestamp in enumerate(data[i]["Timestamps"]):
                data[i]["Timestamps"][j] = pd.to_datetime(timestamp)

            df = pd.DataFrame({
                "Timestamps": data[i]["Timestamps"],
                "Values": data[i]["Values"]
            })
            dfs.append(df)
    return dfs