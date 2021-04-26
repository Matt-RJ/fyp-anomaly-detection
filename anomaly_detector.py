from datetime import timedelta
import util
from abc import ABC, abstractmethod

class AnomalyDetector(ABC):
  """Abstract base class for anomaly detectors."""

  @abstractmethod
  def __init__(self):
    self._graph_gap_threshold = timedelta(hours=1)
    self._anomaly_neighbor_limit = 1
    self._post_release_threshold = timedelta(days=1)
    self._random_state = None
    self._df = None
    self._df_releases = None
    self._train_end_datetime = None

  @property
  def graph_gap_threshold(self):
    return self._graph_gap_threshold

  @graph_gap_threshold.setter
  def graph_gap_threshold(self, graph_gap_threshold):
    self._graph_gap_threshold = graph_gap_threshold

  @property
  def anomaly_neighbor_limit(self):
    return self._anomaly_neighbor_limit

  @anomaly_neighbor_limit.setter
  def anomaly_neighbor_limit(self, anomaly_neighbor_limit):
    if (not isinstance(anomaly_neighbor_limit, int)):
      raise TypeError(f'anomaly_neighbor_limit must be an integer. Received: {type(anomaly_neighbor_limit)}.')
    elif (anomaly_neighbor_limit < 1):
      raise ValueError(f'anomaly_neighbor_limit must be a positive integer. Received: {anomaly_neighbor_limit}.')
    else:
      self._anomaly_neighbor_limit = anomaly_neighbor_limit

  @property
  def post_release_threshold(self):
    return self._post_release_threshold

  @post_release_threshold.setter
  def post_release_threshold(self, post_release_threshold):
    if (not isinstance(post_release_threshold, timedelta)):
      raise TypeError(f'post_release_threshold must be a timedelta object. Received: {type(post_release_threshold)}')
    else:
      self.post_release_threshold = post_release_threshold

  @property
  def random_state(self):
    return self._random_state

  @random_state.setter
  def random_state(self, random_state):
    self._random_state = random_state

  @property
  def df(self):
    return self._df

  @df.setter
  def df(self, df):
    self._df = df

  def load_df(self, filepath, metric_name):
    """Loads an exported CloudWatch .json file and converts it to a data frame."""
    print(f'Loading df from {filepath}...')
    self._df = util.json_to_pandas(filepath)[metric_name]
    print('Loaded.')

  @property
  def df_releases(self):
    return self._df_releases

  @df_releases.setter
  def df_releases(self, df_releases):
    self._df_releases = df_releases

  def load_df_releases(self, filepath):
    self._df_releases = util.load_releases(filepath)

  @property
  def train_end_datetime(self):
    return self._train_end_datetime

  @train_end_datetime.setter
  def train_end_datetime(self, train_end_datetime):
    self._train_end_datetime = train_end_datetime

  def clean_df(self):
    """Cleans the currently-loaded data frame."""
    if (self.df is None):
      raise TypeError('No data frame loaded. Use .df(filepath) first.')

    # Gap feature for not drawing lines between two distant data points when plotting
    self.df['Gap'] = (self.df.Timestamps.diff() >= self.graph_gap_threshold).astype(int)

  def create_release_features(self, service_name):
    """Creates two features in the data frame: Release_Points and Post_Releases."""
    if (self._df is None):
      raise TypeError('No df loaded. Use .df(filepath) first.')
    if (self._df_releases is None):
      raise TypeError('No df_releases loaded. Use .df_releases(filepath) first.')

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

  @abstractmethod
  def train(self, feature, df_slice=None):
    """Trains the model with the currently-loaded data frame."""
    pass

  @abstractmethod
  def test(self):
    """Performs anomaly detection with the previously-trained data."""
    pass

  @abstractmethod
  def release_train_test(self, df_train_filepath, df_test_filepath, df_releases_filepath, metric_name, df_test_service_name):
    """Performs training on a given train metric, then performs anomaly detection on a given test metric. Displays and counts post-release anomalies."""
    pass
