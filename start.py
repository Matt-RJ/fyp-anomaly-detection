from k_means_anomaly_detector import KMeansAnomalyDetector

detector = KMeansAnomalyDetector(
  n_clusters=150,
  segment_len=32,
  slide_len=32
)
# detector.load_df('../ExportedMetrics/ServiceA/LambdaB.json', 'Duration')
# # detector.clean_df()
# detector.train('Values', (0, 2000))
# # detector.load_df('../ExportedMetrics/ServiceA/LambdaB.json', 'Duration')
# detector.test()
# detector.anomaly_plot()
# # print(detector.df)

detector.release_train_test(
  df_train_filepath    = '../ExportedMetrics/ServiceA/LambdaA.json',
  df_test_filepath     = '../ExportedMetrics/ServiceA/LambdaA.json',
  df_releases_filepath = '../ExportedMetrics/releases.json',
  metric_name          = 'Duration',
  df_test_service_name = 'ServiceA'
)

detector.reconstruction_plot()