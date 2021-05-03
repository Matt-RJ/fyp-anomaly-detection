from k_means_anomaly_detector import KMeansAnomalyDetector
from isolation_forest_anomaly_detector import IsolationForestAnomalyDetector
from datetime import datetime, timedelta

def test_k_means(metric_file=None):
  """Demonstrates k-means anomaly detection."""
  print('Testing K-Means Anomaly Detection')
  detector = KMeansAnomalyDetector(
    n_clusters=350,
    segment_len=32,
    slide_len=2
  )
  detector.reconstruction_quantile = 0.995
  detector.release_train_test(
    df_train_filepath    = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_test_filepath     = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_releases_filepath = '../ExportedMetrics/releases.json',
    metric_name          = 'ConcurrentExecutions',
    df_test_service_name = 'ServiceA',
    df_test_lambda_name  = 'LambdaB'
  )
  detector.reconstruction_plot()

def test_isolation_forest(metric_file=None):
  """Demonstrates isolation forest anomaly detection."""
  print('Testing Isolation Forest Anomaly Detection')
  detector = IsolationForestAnomalyDetector()
  detector.stl_decomp_period = 288
  detector.release_train_test(
    df_train_filepath    = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_test_filepath     = metric_file or '../ExportedMetrics/ServiceA/LambdaB.json',
    df_releases_filepath = '../ExportedMetrics/releases.json',
    metric_name          = 'Duration',
    df_test_service_name = 'ServiceA',
    df_test_lambda_name  = 'LambdaB'
  )
  detector.decomposition_plot('STL Decomposition (Service A Lambda B - Duration)')

def test_aws_k_means(lambda_name):
  """Demonstrates AWS monitoring with K-Means."""
  detector = KMeansAnomalyDetector(
    n_clusters=150,
    segment_len=32,
    slide_len=2
  )
  detector.reconstruction_quantile = 0.99
  detector.start_monitoring_lambda(
    lambda_function_name=lambda_name,
    start_datetime=datetime.now(tz=detector.timezone)-timedelta(hours=6),
  )

def test_aws_isolation_forest(lambda_name):
  """Demonstrates AWS monitoring with isolation forest."""
  detector = IsolationForestAnomalyDetector()
  detector.contamination = 0.02
  detector.start_monitoring_lambda(
    lambda_function_name=lambda_name,
    start_datetime=datetime.now(tz=detector.timezone)-timedelta(hours=6)
  )

def test_both(service_name, lambda_name):
  """ Tests both k-means and isolation forest, saving the resulting graphs."""
  k_means_detector = KMeansAnomalyDetector(
    n_clusters=500,
    segment_len=32,
    slide_len=2
  )
  k_means_detector.reconstruction_quantile = 0.995
  isolation_forest_detector = IsolationForestAnomalyDetector()
  isolation_forest_detector.stl_decomp_period = 288
  isolation_forest_detector.contamination = 0.005

  for metric_name in ['Duration', 'ConcurrentExecutions']:
    for detector in [isolation_forest_detector]:
      try:
        detector.release_train_test(
          df_train_filepath    = f'../ExportedMetrics/{service_name}/{lambda_name}.json',
          df_test_filepath     = f'../ExportedMetrics/{service_name}/{lambda_name}.json',
          df_releases_filepath = f'../ExportedMetrics/releases.json',
          metric_name          = metric_name,
          df_test_service_name = service_name,
          df_test_lambda_name = lambda_name,
          show_plot=False,
          save_fig=True
        )
      except Exception as e:
        print(e)
        

test_k_means()
# test_isolation_forest()
# test_aws_k_means('fyp-image-processor')
# test_aws_isolation_forest('fyp-image-processor')
# test_both('ServiceA', 'LambdaA')