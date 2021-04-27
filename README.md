# FYP Anomaly Detection

This repo contains a part of my final year project for Applied Computing.

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

## Components

* **\*.ipynb**: Jupyter notebooks for anomaly detection development. Superseded by the anomaly_detector files:
  * **anomaly_detector.py**: Abstract base class for anomaly detectors.
  * **k_means_anomaly_detector.py**: Anomaly detector using k-means.
  * **isolation_forest_anomaly_detector.py**: Anomaly detector using isolation forest.
* **util.py**: Contains various helper and graphing functions.
* **download-metrics.py**: Contains functionality for exporting Lambda metrics from CloudWatch as a .json file.
* **use-case-runner**: Contains code for running the [use case](https://github.com/Matt-RJ/fyp-use-case).
* **start.py**: Contains anomaly detector demo code.
