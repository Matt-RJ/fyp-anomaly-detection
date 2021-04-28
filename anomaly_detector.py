from datetime import datetime, timedelta
from pytz import timezone
import util
from download_metrics import download_lambda_metric
from abc import ABC, abstractmethod

class AnomalyDetector(ABC, object):
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
    self._timezone = timezone('Europe/Dublin')

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

  @property
  def timezone(self):
    return self._timezone

  @timezone.setter
  def timezone(self, timezone):
    self._timezone = timezone

  def load_df(self, filepath, metric_name):
    """Loads an exported CloudWatch .json file and converts it to a data frame."""
    print(f'Loading df from {filepath}...')
    self.df = util.json_to_pandas(filepath)[metric_name]
    print('Loaded.')

  @property
  def df_releases(self):
    return self._df_releases

  @df_releases.setter
  def df_releases(self, df_releases):
    self._df_releases = df_releases

  def load_df_releases(self, filepath):
    self.df_releases = util.load_releases(filepath)

  @property
  def train_end_datetime(self):
    return self._train_end_datetime

  @train_end_datetime.setter
  def train_end_datetime(self, train_end_datetime):
    self._train_end_datetime = train_end_datetime

  def clean_df(self):
    """Cleans the currently-loaded data frame."""
    if (self.df is None):
      raise TypeError('No data frame loaded. Use .load_df(filepath) first.')

    # Gap feature for not drawing lines between two distant data points when plotting
    self.df['Gap'] = (self.df.Timestamps.diff() >= self.graph_gap_threshold).astype(int)

  def create_release_features(self, service_name):
    """Creates two features in the data frame: Release_Points and Post_Releases."""
    if (self.df is None):
      raise TypeError('No df loaded. Use .load_df(filepath) first.')
    if (self.df_releases is None):
      raise TypeError('No df_releases loaded. Use .load_df_releases(filepath) first.')

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

  def download_lambda_metrics(self, function_name, start_time=datetime.now()-timedelta(weeks=1), end_time=datetime.now(), out_file='./lambda-metrics.json', load=False, metric_name='Duration'):
    """Downloads a Lambda function's Duration and ConcurrentExecutions metrics to a .json file. If load=True, then load the metric into df."""
    download_lambda_metric(function_name, start_time, end_time, out_file)
    if (load):
      self.load_df(filepath=out_file, metric_name=metric_name)

  def get_anomaly_count(self, after=None, anomaly_feature='Anomalies', anomalous_value=1):
    """Gets the total number of anomalies in df, or after a given datetime."""
    if (self.df is None):
      raise TypeError('No data frame loaded. Use .load_df(filepath) first.')
    if (anomaly_feature not in self.df):
      raise ValueError(f'Anomaly feature \'{anomaly_feature}\' missing.')

    if (after):
      return len(self.df[(self.df.Timestamps >= after) & (self.df[anomaly_feature] == anomalous_value)])
    else:
      return len(self.df[(self.df[anomaly_feature] == anomalous_value)])

  @abstractmethod
  def train(self, feature='Values', df_slice=None):
    """Trains the model with the currently-loaded data frame."""
    pass

  @abstractmethod
  def test(self, feature='Values'):
    """Performs anomaly detection with the previously-trained data."""
    pass

  @abstractmethod
  def release_train_test(self, df_train_filepath, df_test_filepath, df_releases_filepath, metric_name, df_test_service_name):
    """Performs training on a given metric, then performs anomaly detection on another (or same) metric. Displays and counts post-release anomalies."""
    pass

  @abstractmethod
  def anomaly_plot(self, title='Figure'):
    """Displays a graph with anomalies."""
    pass

  @abstractmethod
  def start_monitoring_lambda(self, lambda_function_name, start_datetime=None,
                              refresh_frequency=timedelta(minutes=1), monitor_duration=timedelta(minutes=30)):
    """ Begins monitoring an AWS Lambda function for anomalies."""
    pass

  # @abstractmethod
  # def report(self, anomaly_feature):
  #   """Returns a summary report."""
  #   if (self.df is None):
  #     raise TypeError('No data frame loaded. Use .load_df(filepath) first.')
  #   report = {}
  #   report['Total_Anomalies'] = None
  #   pass # TODO: Implement