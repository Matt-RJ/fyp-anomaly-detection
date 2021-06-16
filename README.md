# FYP Anomaly Detection

This repo contains a part of my final year project for Applied Computing.

Related repo: [fyp-use-case](https://github.com/Matt-RJ/fyp-use-case).

## Demo Video

The demo video that was submitted as part of the project can be found [here](https://www.youtube.com/watch?v=BNU7gm1Dd1Q).

## Description

This program is an anomaly detection system for time series data, particularly, time series data from AWS Lambda functions. The anomaly detector has two types of models implemented: isolation forest and k-means. The detector has functionality for:

* Analysing a time series data set with Timestamps and Values features.
* Analysing exported CloudWatch metrics for anomalies retrospectively.
* Generating graphs of anomalous results.
* Live anomaly detection against a running Lambda function.

## Prerequisites

The following are required to run the anomaly detectors:

* A Windows installation (This repo was developed on Windows, functionality on MacOS and Linux may break)
* aws-cli
* Python 3 with:
  * pandas
  * numpy
  * scikit-learn
  * boto3
  * matplotlib

## Getting Started

Getting the anomaly detector to run requires the following:

1. Install all the required packages and programs as outlined in the `Prerequisites section above`.
2. Clone the repo.
3. For retrospective anomaly detection:
    * A folder is expected in the parent directory of the repo called `ExportedMetrics`. A sample file has been provided with one sample Lambda metric, courtesy of ServisBOT in the SampleExportedMetrics folder.
    * Move the SampleExportedMetrics folder to the parent directory of the repo and rename it to ExportedMetrics.
4. For live AWS Lambda anomaly detection:
    1. Run `aws config` and set your access key and secret access key. Set the default region your Lambda function is set in, and set the default output format to json.
    2. In `start.py`, edit the function parameter for either `test_aws_k_means` or `test_aws_isolation_forest` to use your Lambda function name.
5. Check `start.py` - demo code is prewritten in this file. Uncomment the desired function to be called at the bottom of the file.
6. Run `python start.py` from the repo directory to start the anomaly detector.

## Components

* **\*.ipynb**: Jupyter notebooks for anomaly detection development. Superseded by the anomaly_detector files:
  * **anomaly_detector.py**: Abstract base class for anomaly detectors.
  * **k_means_anomaly_detector.py**: Anomaly detector using k-means.
  * **isolation_forest_anomaly_detector.py**: Anomaly detector using isolation forest.
* **util.py**: Contains various helper and graphing functions.
* **download_metrics.py**: Contains functionality for exporting Lambda metrics from CloudWatch as a .json file.
* **use-case-runner**: Contains code for running the [use case](https://github.com/Matt-RJ/fyp-use-case).
* **start.py**: Contains anomaly detector demo code.
