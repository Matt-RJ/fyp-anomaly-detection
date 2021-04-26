from anomaly_detector import AnomalyDetector
from sklearn.cluster import KMeans
import util

class KMeansAnomalyDetector(AnomalyDetector):
  def __init__(self, n_clusters=150, segment_len=32, slide_len=2):
    super().__init__()
    self._n_clusters = n_clusters
    self._segment_len = segment_len
    self._slide_len = slide_len
    self._train_end_datetime = None
    self._random_state = None
    self._df = None
    self._df_releases = None
    self._clusterer = None
    self._reconstruction_quantile = 0.995

  @property
  def n_clusters(self):
    return self._n_clusters

  @n_clusters.setter
  def n_clusters(self, n_clusters):
    self._n_clusters = n_clusters
  
  @property
  def segment_len(self):
    return self._segment_len

  @segment_len.setter
  def segment_len(self, segment_len):
    self._segment_len = segment_len
  
  @property
  def slide_len(self):
    return self._slide_len

  @slide_len.setter
  def slide_len(self, slide_len):
    self._slide_len = slide_len

  @property
  def clusterer(self):
    return self._clusterer

  @clusterer.setter
  def clusterer(self, clusterer):
    self._clusterer = clusterer

  @property
  def reconstruction_quantile(self):
    return self._reconstruction_quantile

  @reconstruction_quantile.setter
  def reconstruction_quantile(self, reconstruction_quantile):
    self._reconstruction_quantile = reconstruction_quantile
  
  def release_train_test(self, df_train_filepath, df_test_filepath, df_releases_filepath, metric_name, df_test_service_name):
    """Performs training on a given metric, then performs anomaly detection on another metric. Displays and counts post-release anomalies."""
    
    self.load_df(df_train_filepath, metric_name)
    self.train('Values')
    self.load_df(df_test_filepath, metric_name)
    self.load_df_releases(df_releases_filepath)
    self.clean_df()
    self.create_release_features(df_test_service_name)
    self.reconstruct()
    util.anomaly_plot_with_releases(self.df, metric_name, self.post_release_threshold)

  def train(self, feature, df_slice=None):
    """Trains the model with the currently-loaded data frame."""

    df = self.df
    if (df_slice): # Taking a part of the data frame for training instead of the full thing (optional)
      print(f'Training with df[{df_slice[0]}:{df_slice[1]}]')
      df = df[df_slice[0]:df_slice[1]]
    print('Starting training on currently-loaded data frame...')
    df, segments = util.window_df(df, segment_len=self.segment_len, slide_len=self.slide_len)
    windowed_segments = util.normalize_segments(segments, feature=feature, segment_len=self.segment_len)

    X = list(map(lambda x: x[feature], windowed_segments))
    n_clusters = self.n_clusters
    if (n_clusters > len(X)):
      print(f'n_clusters ({n_clusters}) exceeds the total number of samples for training ({len(X)}). Using n_clusters={len(X)}')
      n_clusters = len(X)

    self.clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
    self.clusterer.fit(X)
    self.train_end_datetime = df.tail(1).Timestamps.values[0]
    print('Training complete.')
    
  def reconstruct(self):
    """Performs data frame reconstruction with clustering."""
    self.df = util.reconstruct(self.df, 'Values', self.clusterer, self.segment_len, self.reconstruction_quantile)
    self.df = util.limit_anomalies(self.df, 'Anomalies', -1, 1, self.anomaly_neighbor_limit)

  def test(self):
    """Performs anomaly detection with the previously-trained data."""
    print('Starting test')
    print('Reconstructing...')
    self.reconstruct()
    print('Testing complete.')

  def reconstruction_plot(self):
    """Displays a reconstruction plot."""
    util.reconstruction_plot(self.df)

  def anomaly_plot(self):
    """Displays a graph with anomalies."""
    util.anomaly_plot(self.df, 'Anomalies')
