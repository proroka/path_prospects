import argparse
import matplotlib.pylab as plt
import msgpack
import msgpack_numpy
import pandas as pd
import os.path
import seaborn as sns

import run


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.

  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--input', action='store', default=None, help='Where to the results are stored.')
  args = parser.parse_args()

  # Aggregate results.
  data = []
  data_columns = ['problem_set', 'problem_name', 'scheme', 'priority_scheme', 'tiebreak_scheme', 'communication_radius'] + list(run.AggregatedStatistics._fields)
  all_results = run.read_results(args.input)
  for raw_args, results in all_results.items():
    args = run.Arguments(*raw_args)
    for raw_stats in results:
      stats = run.compute_aggregated_statistics(
          run.Statistics(*raw_stats))
      data.append((os.path.dirname(args.problem), os.path.basename(args.problem),
                   args.scheme[0] + '+' + args.scheme[1],
                   args.scheme[0], args.scheme[1],
                   args.communication_radius) +
                  tuple(getattr(stats, f) for f in run.AggregatedStatistics._fields))
  df = pd.DataFrame(data, columns=data_columns)

  # Plot makespan.
  sns.catplot(x='communication_radius', y='flowtime_ratio', hue='scheme', col='problem_set', data=df,
              kind='bar')
  plt.show()
