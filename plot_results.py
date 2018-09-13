import argparse
import collections
import matplotlib.pylab as plt
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import os.path
import seaborn as sns

import run


_RENAME = {
    'random_priority': 'Random',
    'surroundings_50_priority': 'Surroundings',
    'longest_priority': 'Longest first',
    'forward_looking_priority': 'Forwards looking',
    # Naive is moved to the tiebreak row.
    'naive/tiebreak_random': 'N-R',
    'naive/tiebreak_longest_first': 'N-LF',
    'tiebreak_random': 'R',
    'tiebreak_longest_first': 'LF',
    # Problem sets.
    'problems/berlin': 'Berlin',
    'problems/corridor': 'Corridor',
    'problems/cross': 'Crossing',
    'problems/multi_corridor': 'Multi-corridor',
    'problems/successive_clutter': 'Clutter',
}

_SCHEMES = [
    'random_priority+tiebreak_random',
    'naive_surroundings_50_priority+tiebreak_longest_first',
    'surroundings_50_priority+tiebreak_longest_first',
    'longest_priority+tiebreak_random',
    'naive_forward_looking_priority+tiebreak_longest_first',
    'forward_looking_priority+tiebreak_random',
    'forward_looking_priority+tiebreak_longest_first',
]

_PROBLEMS = [
    'problems/berlin',
    'problems/cross',
    'problems/corridor',
    'problems/multi_corridor',
    'problems/successive_clutter',
]


def compute_stats(df, y):
  # col_order = list(sorted(df.problem_set.drop_duplicates().values))
  col_order = _PROBLEMS
  # hue_order = list(sorted(df.scheme.drop_duplicates().values))
  hue_order = _SCHEMES
  order = list(sorted(df.communication_radius.drop_duplicates().values))
  g = sns.catplot(x='communication_radius', y=y, hue='scheme', col='problem_set', data=df, kind='bar', col_order=col_order, hue_order=hue_order, order=order)

  means = collections.defaultdict(lambda: collections.defaultdict(dict))
  stds = collections.defaultdict(lambda: collections.defaultdict(dict))
  for ax in g.axes.flat:
    problem_set = ax.title.get_text().split(' = ', 1)[1]
    assert len(ax.patches) == len(ax.lines)
    for i, (b, err) in enumerate(zip(ax.patches, ax.lines)):
      scheme = hue_order[i // len(order)]
      communication_radius = order[i % len(order)]
      means[problem_set][communication_radius][scheme] = b.get_height()
      stds[problem_set][communication_radius][scheme] = err.get_xydata()[:, 1]

  latex = ['\section{%s}' % y.replace('_', ' ')]
  schemes = collections.defaultdict(list)
  for scheme in hue_order:
    priority_scheme, tiebreak_scheme = scheme.split('+', 1)
    if priority_scheme.startswith('naive_'):
      priority_scheme = priority_scheme[len('naive_'):]
      tiebreak_scheme = 'naive/' + tiebreak_scheme
    if priority_scheme in _RENAME:
      priority_scheme = _RENAME[priority_scheme]
    if tiebreak_scheme in _RENAME:
      tiebreak_scheme = _RENAME[tiebreak_scheme]
    schemes[priority_scheme].append(tiebreak_scheme)

  # Header.
  columns = ['c|c||']  # Problem set, Communication radius.
  for k, v in schemes.items():
    columns.append('c' * len(v) + '|')
  columns = ''.join(columns)
  latex.append('\\begin{tabular}{%s}' % columns)
  headers = ['\multirow{2}{6em}{Problem set}', '\multirow{2}{1em}{$c$}']
  for k, v in schemes.items():
    if len(v) > 1:
      headers.append('\multicolumn{%d}{|c|}{%s}' % (len(v), k))
    else:
      headers.append(k)
  latex.append(' & '.join(headers) + ' \\\\')
  headers = ['', '']
  for k, v in schemes.items():
    for tb in v:
      headers.append(tb)
  latex.append(' & '.join(headers) + ' \\\\')
  latex.append('\hline')

  # Values.
  for problem_set in col_order:
    pb = problem_set
    if pb in _RENAME:
      pb = _RENAME[pb]
    for i, communication_radius in enumerate(order):
      row = []
      if i == 0:
        row.append('\multirow{2}{6em}{%s}' % pb)
      else:
        row.append('')
      row.append(str(int(communication_radius)))
      values = [means[problem_set][communication_radius][scheme] for scheme in hue_order]
      min_value = round(min(values), 4)
      for v in values:
        if round(v, 4) == min_value:
          row.append('\\textbf{%.2f\\%%}' % (v * 100.))
        else:
          row.append('%.2f\\%%' % (v * 100.))
      latex.append(' & '.join(row) + ' \\\\')
    latex.append('\hline')
  latex.append('\end{tabular}')

  latex = '\n'.join(latex)
  print(latex)


def show_ranks(df, y):
  expected_number = len(_SCHEMES)
  ranks = collections.defaultdict(lambda: collections.defaultdict(list))
  for k, g in df.groupby(['problem_set', 'problem_name', 'communication_radius']):
    successes = g.success.values
    if not np.all(successes) or len(successes) != expected_number:
      continue
    scheme = g.scheme.values
    rank = g[y].rank(ascending=True, method='min').values
    for s, r in zip(scheme, rank):
      ranks[k[0]][s].append(r)
  data_ranks = []
  data_ranks_columns = ['problem_set', 'scheme', 'rank']
  for problem_set, v in ranks.items():
    for scheme, rank in v.items():
      for r in rank:
        data_ranks.append((problem_set, scheme, r))
  df_ranks = pd.DataFrame(data_ranks, columns=data_ranks_columns)
  g = sns.catplot(x='problem_set', y='rank', hue='scheme', data=df_ranks, kind='bar', hue_order=_SCHEMES, order=_PROBLEMS)
  plt.grid(axis='y')
  plt.title(y)
  plt.yticks([1, 2, 3, 4])


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
      problem_set = os.path.dirname(args.problem)
      if problem_set.endswith('basic'):
        continue
      data.append((problem_set, os.path.basename(args.problem),
                   args.scheme[0] + '+' + args.scheme[1],
                   args.scheme[0], args.scheme[1],
                   args.communication_radius) +
                  tuple(getattr(stats, f) for f in run.AggregatedStatistics._fields))
  df = pd.DataFrame(data, columns=data_columns)

  # Plots.
  # for y in ('makespan_ratio', 'flowtime_ratio', 'success'):
  #   compute_stats(df, y)
  #   print('')

  for y in ('makespan_ratio', 'flowtime_ratio'):
    show_ranks(df, y)
  plt.show()
