from datetime import timedelta
from sklearn.cluster import KMeans
import util

class KMeansAnomalyDetector:
  def __init__(self, n_clusters=150, segment_len=32, slide_len=2, graph_gap_threshold=timedelta(hours=2),
  anomaly_neighbor_limit=1, post_release_threshold=timedelta(days=1)):
    self.n_clusters = n_clusters
    self.segment_len = segment_len
    self.slide_len = slide_len
    self.graph_gap_threshold = graph_gap_threshold
    self.anomaly_neighbor_limit = anomaly_neighbor_limit
    self.post_release_threshold = post_release_threshold
    self.train_end_datetime = None # Datetime of the last data point of training. In practice, this is the last point before a release.
    self.random_state = None
    self.df = None
    self.df_releases = None
    self.clusterer = None
    self.reconstruction_quantile = 0.995

  def set_n_clusters(self, n_clusters):
    self.n_clusters = n_clusters
  
  def set_segment_len(self, segment_len):
    self.segment_len = segment_len
  
  def set_slide_len(self, slide_len):
    self.slide_len = slide_len

  def set_graph_gap_threshold(self, graph_gap_threshold):
    self.graph_gap_threshold = graph_gap_threshold

  def set_anomaly_neighbor_limit(self, anomaly_neighbor_limit):
    if (not isinstance(anomaly_neighbor_limit, int)):
      raise TypeError(f'anomaly_neighbor_limit must be an integer. Received: {type(anomaly_neighbor_limit)}.')
    elif (anomaly_neighbor_limit < 1):
      raise ValueError(f'anomaly_neighbor_limit must be a positive integer. Received: {anomaly_neighbor_limit}.')
    else:
      self.anomaly_neighbor_limit = anomaly_neighbor_limit

  def set_post_release_threshold(self, post_release_threshold):
    if (not isinstance(post_release_threshold, timedelta)):
      raise TypeError(f'post_release_threshold must be a timedelta object. Received: {type(post_release_threshold)}')
    else:
      self.post_release_threshold = post_release_threshold

  def load_df(self, filepath, metric_name):
    """Loads a data frame from exported CloudWatch JSON."""
    print(f'Loading df from {filepath}...')
    self.df = util.json_to_pandas(filepath)[metric_name]
    print('Loaded.')

  def load_df_releases(self, filepath):
    self.df_releases = util.load_releases(filepath)

  def clean_df(self):
    """Cleans the currently-loaded data frame."""
    if (self.df is None):
      raise TypeError('No data frame loaded. Use load_df first.')

    # Gap feature for not drawing lines between two distant data points when plotting
    self.df['Gap'] = (self.df.Timestamps.diff() >= self.graph_gap_threshold).astype(int)

  def create_release_features(self, service_name):
    """Creates two features in the data frame: Release_Points and Post_Releases."""
    if (self.df is None):
      raise TypeError('No df loaded. Use load_df() first.')
    if (self.df_releases is None):
      raise TypeError('No df_releases loaded. Use load_df_releases() first.')

    # Some microservices are grouped together during release
    release_service_map = {
        'ServiceA': 'ServiceA',
        'ServiceB': 'ServiceB',
        'ServiceC': 'ServiceCD',
        'ServiceD': 'ServiceCD',
        'ServiceE': 'ServiceEF',
        'ServiceF': 'ServiceEF',
        'ServiceG': None,
        'ServiceH': 'ServiceHIK',
        'ServiceI': 'ServiceHIK',
        'ServiceJ': 'ServiceJ',
        'ServiceK': 'ServiceHIK'
    }

    releases = self.df_releases.loc[self.df_releases['ServiceNames'] == release_service_map[service_name]] if release_service_map[service_name] != None else None
    self.df = util.calculate_release_point_feature(self.df, releases)
    self.df = util.calculate_post_release_feature(self.df, self.post_release_threshold)
  
  def release_train_test(self, df_train_filepath, df_test_filepath, df_releases_filepath, metric_name, df_test_service_name):
    """Performs training on a given metric, then performs anomaly detection on another metric. Displays and counts post-release anomalies."""
    
    self.load_df(df_train_filepath, metric_name)
    self.train('Values')
    self.load_df(df_test_filepath, metric_name)
    self.load_df_releases(df_releases_filepath)
    self.clean_df()
    self.create_release_features(df_test_service_name)
    self.reconstruct()
    # Anomaly detection
    util.anomaly_plot_with_releases(self.df, metric_name, self.post_release_threshold)

  def train(self, feature, df_slice=None):
    """Trains the model with the currently-loaded data frame."""

    df = self.df
    if (df_slice): # Taking a part of the data frame for training instead of the full thing (optional)
      print(f'Training with df[{df_slice[0]}:{df_slice[1]}]')
      df = df[df_slice[0]:df_slice[1]]
    print('Starting training on currently-loaded data frame...')
    df, segments = util.window_df(df, segment_len=self.segment_len, slide_len=self.slide_len)
    windowed_segments = util.normalise_segments(segments, feature=feature, segment_len=self.segment_len)

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
