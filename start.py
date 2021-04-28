from k_means_anomaly_detector import KMeansAnomalyDetector
from isolation_forest_anomaly_detector import IsolationForestAnomalyDetector
from datetime import datetime, timedelta

def test_k_means(metric_file=None):
  """Demonstrates k-means anomaly detection."""
  print('Testing K-Means Anomaly Detection')
  detector = KMeansAnomalyDetector(
    n_clusters=150,
    segment_len=32,
    slide_len=32
  )
  detector.release_train_test(
    df_train_filepath    = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_test_filepath     = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_releases_filepath = '../ExportedMetrics/releases.json',
    metric_name          = 'Duration',
    df_test_service_name = 'ServiceA'
  )
  detector.reconstruction_plot()

def test_aws_k_means(lambda_name):
  """Demonstrates AWS monitoring with K-Means."""
  detector = KMeansAnomalyDetector(
    n_clusters=150,
    segment_len=32,
    slide_len=32
  )
  detector.reconstruction_quantile = 0.99
  detector.start_monitoring_lambda(
    lambda_function_name=lambda_name,
    start_datetime=datetime.now(tz=detector.timezone)-timedelta(hours=6),
  )

def test_isolation_forest(metric_file=None):
  """Demonstrates isolation forest anomaly detection."""
  print('Testing Isolation Forest Anomaly Detection')
  detector = IsolationForestAnomalyDetector()
  detector.contamination = 0.005
  detector.release_train_test(
    df_train_filepath    = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_test_filepath     = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_releases_filepath = '../ExportedMetrics/releases.json',
    metric_name          = 'Duration',
    df_test_service_name = 'ServiceA'
  )

def test_aws_isolation_forest(lambda_name):
  """Demonstrates AWS monitoring with isolation forest."""
  detector = IsolationForestAnomalyDetector()
  detector.contamination = 0.02
  detector.start_monitoring_lambda(
    lambda_function_name=lambda_name,
    start_datetime=datetime.now(tz=detector.timezone)-timedelta(hours=6)
  )

test_k_means()
# print('\n\n')
test_isolation_forest()

# test_aws_k_means('fyp-image-processor')
# test_aws_isolation_forest('fyp-image-processor')