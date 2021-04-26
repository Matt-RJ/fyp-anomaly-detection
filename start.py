from k_means_anomaly_detector import KMeansAnomalyDetector
from isolation_forest_anomaly_detector import IsolationForestAnomalyDetector

def test_k_means():
  print('Testing K-Means Anomaly Detection')
  detector = KMeansAnomalyDetector(
    n_clusters=150,
    segment_len=32,
    slide_len=32
  )
  detector.release_train_test(
    df_train_filepath    = '../ExportedMetrics/ServiceA/LambdaA.json',
    df_test_filepath     = '../ExportedMetrics/ServiceA/LambdaA.json',
    df_releases_filepath = '../ExportedMetrics/releases.json',
    metric_name          = 'Duration',
    df_test_service_name = 'ServiceA'
  )
  detector.reconstruction_plot()

def test_isolation_forest():
  print('Testing Isolation Forest Anomaly Detection')
  detector = IsolationForestAnomalyDetector()
  detector.contamination = 0.01
  detector.release_train_test(
    df_train_filepath    = '../ExportedMetrics/ServiceA/LambdaA.json',
    df_test_filepath     = '../ExportedMetrics/ServiceA/LambdaA.json',
    df_releases_filepath = '../ExportedMetrics/releases.json',
    metric_name          = 'Duration',
    df_test_service_name = 'ServiceA'
  )

test_k_means()
print('\n\n')
test_isolation_forest()