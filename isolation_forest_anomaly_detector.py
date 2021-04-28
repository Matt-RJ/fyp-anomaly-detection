from anomaly_detector import AnomalyDetector
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from time import sleep
from datetime import datetime, timedelta

import util

class IsolationForestAnomalyDetector(AnomalyDetector):
  """Performs time series anomaly detection with time series decomposition and Isolation Forest."""
  def __init__(self):
    super().__init__()
    self._stl_decomp_period = 50
    self._contamination = 0.01
    self._isolation_forest = None
    self._max_features = 1.0
    self._n_estimators = 50
    self._max_samples = 'auto'

  @property
  def stl_decomp_period(self):
    return self._stl_decomp_period

  @stl_decomp_period.setter
  def stl_decomp_period(self, stl_decomp_period):
    if (stl_decomp_period is not None and stl_decomp_period <= 1):
      raise ValueError(f'stl_decomp_period must be greater than 1. Received: {stl_decomp_period}.')
    else:
      self._stl_decomp_period = stl_decomp_period

  @property
  def contamination(self):
    return self._contamination

  @contamination.setter
  def contamination(self, contamination):
    if (not (isinstance(contamination, str) or isinstance(contamination, float))):
      raise TypeError(f'Contamination must be an instance of float or str. Received: {contamination} ({type(contamination)}')
    else:
      self._contamination = contamination

  @property
  def isolation_forest(self):
    return self._isolation_forest

  @isolation_forest.setter
  def isolation_forest(self, isolation_forest):
    self._isolation_forest = isolation_forest

  @property
  def max_features(self):
    return self._max_features

  @max_features.setter
  def max_features(self, max_features):
    self._max_samples = max_features

  @property
  def n_estimators(self):
    return self._n_estimators

  @n_estimators.setter
  def n_estimators(self, n_estimators):
    self._n_estimators = n_estimators

  @property
  def max_samples(self):
    return self._max_samples

  @max_samples.setter
  def max_samples(self, max_samples):
    self._max_samples = max_samples
  
  def release_train_test(self, df_train_filepath, df_test_filepath, df_releases_filepath, metric_name, df_test_service_name):
    """
    Performs training on a given metric, then performs anomaly detection on another (or same) metric. Displays and counts post-release anomalies.\n
    Training and testing are performed twice - once for the original values and once for the decomposed residual values.
    """

    self.load_df(df_train_filepath, metric_name)
    self.clean_df()
    self.decompose_df()
    self.train('Residual_Values')
    self.test()
    self.load_df(df_test_filepath, metric_name)
    self.clean_df()
    self.decompose_df()
    self.load_df_releases(df_releases_filepath)
    self.create_release_features(df_test_service_name)
    self.test('Residual_Values')
    util.anomaly_plot_with_releases(
      self.df,
      metric_name,
      self.post_release_threshold,
      anomaly_feature='Residual_Values_Anomalies',
      title='Isolation Forest Anomaly Detection'
    )

  def train(self, feature='Values', df_slice=None):
    """
    Trains the model with the currently-loaded data frame.\n
    df_slice: Optional tuple (slice_start,slice_end) to train with part of df.
    """

    df = self.df

    if (df_slice): # Taking a part of the data frame for training instead of the full thing (optional)
      print(f'Training with df[{df_slice[0] or None}:{df_slice[1] or None}]')
      df = df[df_slice[0]:df_slice[1]]

    print('Starting training with currently-loaded data frame...')
    self.isolation_forest = IsolationForest(
        max_features = self.max_features,
        n_estimators = self.n_estimators,
        max_samples = self.max_samples,
        contamination = self.contamination,
        random_state = self.random_state or None
    )
    self.isolation_forest.fit(df[[feature]])
    self.train_end_datetime = df.tail(1).Timestamps.values[0]
    self.df = df
    print('Training complete.')
  
  def decompose_df(self):
    """
    Performs STL decomposition on the currently-loaded data frame.\n
    Creates three new features: Trend_Values, Seasonal_Values, and Residual_Values.
    """
    if (self.df is None):
      raise TypeError('No df loaded. Use .load_df(filepath) first.')
    df = self.df
    decompose_result = seasonal_decompose(
        df.Values,
        period=self.stl_decomp_period,
        extrapolate_trend='freq',
        model='multiplicative'
    )
    df['Trend_Values'] = decompose_result.trend
    df['Seasonal_Values'] = decompose_result.seasonal
    df['Residual_Values'] = decompose_result.resid

    self.df = df

  def test(self, feature='Values'):
    """Performs anomaly detection with the previously-trained data."""
    print('Starting test')
    df = self.df
    iso_forest = self.isolation_forest
    score_feature = f'{feature}_Scores'
    anomaly_feature = f'{feature}_Anomalies'
    df[score_feature] = iso_forest.decision_function(df[[feature]])
    df[anomaly_feature] = iso_forest.predict(df[[feature]])
    # iso_forest.predict creates an inlier feature (1=inlier, -1=anomaly). Remapping 
    df[anomaly_feature] = df[anomaly_feature].map(lambda x: 0 if x == 1 else 1)
    print('Testing complete.')

  def start_monitoring_lambda(self, lambda_function_name, start_datetime=None,
                              refresh_frequency=timedelta(minutes=1), monitor_duration=timedelta(minutes=30)):
    """ Begins monitoring an AWS Lambda function for anomalies."""
    start_time = datetime.now(tz=self.timezone)
    while (datetime.now(tz=self.timezone) < start_time + monitor_duration):
      # Loading metrics
      self.download_lambda_metrics(
        function_name=lambda_function_name,
        load=True
      )
      self.clean_df()
      self.decompose_df()

      # Optional df truncation
      if (start_datetime):
        self._df = self.df.loc[self.df.Timestamps >= start_datetime]

      self.train('Residual_Values')
      self.test('Residual_Values')
      new_anomalies = self.get_anomaly_count(after=start_time, anomaly_feature='Residual_Values_Anomalies')
      print('NEW ANOMALIES:', new_anomalies)
      if (new_anomalies > 0):
        self.anomaly_plot(feature='Residual_Values_Anomalies', title=f'Isolation Forest Anomaly Detection ({lambda_function_name})')
      print(f'Next test in {refresh_frequency.total_seconds()/60} minute(s)...')
      sleep(refresh_frequency.total_seconds())
    print('Monitoring complete.')

  def anomaly_plot(self, feature, title='Isolation Forest Anomaly Detection'):
    """Displays a graph with anomalies."""
    util.anomaly_plot(self.df, feature, title)
