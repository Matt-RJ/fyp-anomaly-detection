from datetime import datetime, timedelta
import boto3
import json

cw_client = boto3.client('cloudwatch')

def download_lambda_metric(function_name, start_time=datetime.now()-timedelta(days=1), end_time=datetime.now(), out_file='./metric.json'):
  """Downloads a given Lambda metric's Duration and ConcurrentExecutions metrics."""
  
  queries = [
    {
      'Id': 'm1',
      'MetricStat': {
        'Metric': {
          'Namespace': 'AWS/Lambda',
          'MetricName': 'Duration',
          'Dimensions': [
            {
              'Name': 'FunctionName',
              'Value': function_name
            },
          ],
        },
        'Period': 60,
        'Stat': 'Average',
        'Unit': 'Milliseconds'
      }
    },
    {
      'Id': 'm2',
      'MetricStat': {
        'Metric': {
          'Namespace': 'AWS/Lambda',
          'MetricName': 'ConcurrentExecutions',
          'Dimensions': [
            {
              'Name': 'FunctionName',
              'Value': function_name
            },
          ],
        },
        'Period': 60,
        'Stat': 'Average',
        'Unit': 'Count'
      }
    }
  ]
  res = cw_client.get_metric_data(
    MetricDataQueries=queries,
    StartTime=start_time,
    EndTime=end_time,
  )

  # JSON can't parse datetime objects that boto3 returns; converting to timestamp strings.
  for i, metric in enumerate(res['MetricDataResults']):
    res['MetricDataResults'][i]['Timestamps'] = list(
      map(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'), metric['Timestamps'])
    )

  res = res['MetricDataResults']

  with open(out_file, 'w+') as file:
    file.write(json.dumps(res))
  file.close()

  return res