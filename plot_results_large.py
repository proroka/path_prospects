import argparse
import collections
import matplotlib
import matplotlib.pylab as plt
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import os.path
import seaborn as sns
import scipy.stats

import run
import elopy

_SCHEMES = [
    'random_priority+tiebreak_random',  # Must be first.
    'naive_surroundings_30_priority+tiebreak_longest_first',
    'naive_forward_looking_priority+tiebreak_longest_first',
    'longest_priority+tiebreak_random',
    'surroundings_30_priority+tiebreak_longest_first',
    'forward_looking_priority+tiebreak_random',
    'forward_looking_priority+tiebreak_longest_first',
]

_PROBLEMS = [
    'large_maze',
]

_RENAME = {
    'random_priority': 'Random',
    'surroundings_50_priority': 'Surroundings',
    'surroundings_10_priority': 'Surroundings (10)',
    'surroundings_20_priority': 'Surroundings (20)',
    'surroundings_30_priority': 'Surroundings',
    'longest_priority': 'Longest first',
    'forward_looking_priority': 'Path Prospects',
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
    'problems/large_maze': 'Large Maze',
}


def rename_scheme(scheme):
  priority_scheme, tiebreak_scheme = scheme.split('+', 1)
  if priority_scheme.startswith('naive_'):
    priority_scheme = priority_scheme[len('naive_'):]
    tiebreak_scheme = 'naive/' + tiebreak_scheme
  if priority_scheme in _RENAME:
    priority_scheme = _RENAME[priority_scheme]
  if tiebreak_scheme in _RENAME:
    tiebreak_scheme = _RENAME[tiebreak_scheme]
  return priority_scheme + ' (' + tiebreak_scheme + ')'


def plot_barhist(df, y):
  plt.rc('font', family='serif')
  # matplotlib.rcParams['ps.useafm'] = True
  # matplotlib.rcParams['pdf.use14corefonts'] = True
  # matplotlib.rcParams['text.usetex'] = True
  matplotlib.rcParams['pdf.fonttype'] = 42

  col_order = _PROBLEMS
  order = [rename_scheme(s) for s in _SCHEMES]
  g = sns.catplot(x='scheme', y=y, col=None, data=df, kind='bar', col_order=col_order, order=order,
                  palette=sns.color_palette(['orangered'] * 4 + ['dodgerblue'] * 3))
  g.set_xticklabels(rotation=30, ha='right')
  plt.ylim(bottom=.75, top=1.)
  locs, _ = plt.yticks()
  labels = []
  for l in locs:
    labels.append('{:.0f}%'.format(l * 100.))
  plt.yticks(locs, labels)
  plt.tight_layout()


def plot_pareto(df, x, y):
  plt.rc('font', family='serif')
  # matplotlib.rcParams['ps.useafm'] = True
  # matplotlib.rcParams['pdf.use14corefonts'] = True
  # matplotlib.rcParams['text.usetex'] = True
  matplotlib.rcParams['pdf.fonttype'] = 42


  symbols = {
      'random_priority+tiebreak_random': '<',
      'naive_surroundings_30_priority+tiebreak_longest_first': '^',
      'naive_forward_looking_priority+tiebreak_longest_first': 'v',
      'longest_priority+tiebreak_random': '>',
      'surroundings_30_priority+tiebreak_longest_first': 'o',
      'forward_looking_priority+tiebreak_random': 's',
      'forward_looking_priority+tiebreak_longest_first': 'D',
  }
  colors = {
      'random_priority+tiebreak_random': 'orangered',
      'naive_surroundings_30_priority+tiebreak_longest_first': 'orangered',
      'naive_forward_looking_priority+tiebreak_longest_first': 'orangered',
      'longest_priority+tiebreak_random': 'orangered',
      'surroundings_30_priority+tiebreak_longest_first': 'orangered',
      'forward_looking_priority+tiebreak_random': 'dodgerblue',
      'forward_looking_priority+tiebreak_longest_first': 'dodgerblue',
  }
  limits = {
      'clutter': [0.015, 0.029, 0., 0.017],
      'corridor': [0.021, 0.032, 0.0, 0.04],
      'crossing': [0.03, 0.05, 0.01, 0.085],
      'maze': [0.035, 0.065, 0.01, 0.05],
      'tunnel': [0.059, 0.069, 0.02, 0.115],
      'warehouse': [0.025, 0.042, 0., 0.03],
      'large_maze': [0.08, 0.12, 0., 0.18],
  }

  for problem in _PROBLEMS:
    plt.figure()
    for scheme in _SCHEMES:
      s = rename_scheme(scheme)
      d = df[np.logical_and(np.logical_and(df.success, df.problem_set == problem), df.scheme == s)]
      vx = d[x].values
      vy = d[y].values
      mx = np.mean(vx)
      my = np.mean(vy)
      n = len(vx)
      confidence = 0.95
      sx = scipy.stats.sem(vx) * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
      sy = scipy.stats.sem(vy) * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
      plt.plot([mx, mx], [my - sy, my + sy], color=colors[scheme], lw=2, alpha=.5)
      plt.plot([mx - sx, mx + sx], [my, my], color=colors[scheme], lw=2, alpha=.5)
      plt.plot(mx, my, symbols[scheme], color=colors[scheme], label=s, markersize=10, markeredgecolor='k')
    plt.axis(limits[problem])
    locs, _ = plt.xticks()
    labels = []
    for l in locs:
      labels.append('{:.2f}%'.format(l * 100.))
    plt.xticks(locs, labels)
    locs, _ = plt.yticks()
    labels = []
    for l in locs:
      labels.append('{:.2f}%'.format(l * 100.))
    plt.yticks(locs, labels)
    plt.ylabel('Makespan increase')
    plt.xlabel('Flowtime increase')
    plt.tight_layout()
  plt.legend()


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.

  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--input', action='store', default=None, help='Where to the results are stored.')
  args = parser.parse_args()

  # Problem results.
  statistics = collections.defaultdict(
      lambda: collections.defaultdict(
          lambda: collections.defaultdict(dict)))

  all_results = run.read_results(args.input)
  for raw_args, results in all_results.items():
    args = run.Arguments(*raw_args)

    for raw_stats in results:
      stats = run.compute_aggregated_statistics(run.Statistics(*raw_stats))
      problem_set = os.path.basename(os.path.dirname(args.problem))
      problem_idx = os.path.basename(args.problem)
      scheme = args.scheme[0] + '+' + args.scheme[1]
      if scheme in _SCHEMES and problem_set in _PROBLEMS:
        statistics[problem_set][problem_idx][args.communication_radius][scheme] = stats

  # Global statistics.
  baseline = 'forward_looking_priority+tiebreak_longest_first'
  win_counts_flowtime = collections.defaultdict(lambda: 0)
  draw_counts_flowtime = collections.defaultdict(lambda: 0)
  lose_counts_flowtime = collections.defaultdict(lambda: 0)
  win_counts_makespan = collections.defaultdict(lambda: 0)
  draw_counts_makespan = collections.defaultdict(lambda: 0)
  lose_counts_makespan = collections.defaultdict(lambda: 0)

  elo_flowtime = elopy.Implementation()
  for s in _SCHEMES:
    elo_flowtime.addPlayer(s)
  elo_makespan = elopy.Implementation()
  for s in _SCHEMES:
    elo_makespan.addPlayer(s)

  data_columns = ['problem_set', 'problem_name', 'scheme', 'flowtime', 'makespan', 'success']
  data = []
  for problem_set in _PROBLEMS:
    for idx in statistics[problem_set]:
      print(problem_set)
      for radius, schemes in statistics[problem_set][idx].items():
        if baseline not in schemes:
          print('skip')
          continue
        base_flowtime = schemes[baseline].flowtime
        base_makespan = schemes[baseline].makespan
        if base_flowtime is None:
          base_flowtime = float('inf')
        if base_makespan is None:
          base_makespan = float('inf')

        for a, scheme_a in enumerate(_SCHEMES):
          if scheme_a not in schemes:
            flowtime_a = float('inf')
            makespan_a = float('inf')
          else:
            flowtime_a = schemes[scheme_a].flowtime
            makespan_a = schemes[scheme_a].makespan
            if flowtime_a is None:
              flowtime_a = float('inf')
            if makespan_a is None:
              makespan_a = float('inf')
          for scheme_b in _SCHEMES[a + 1:]:
            if scheme_b not in schemes:
              flowtime_b = float('inf')
              makespan_b = float('inf')
            else:
              flowtime_b = schemes[scheme_b].flowtime
              makespan_b = schemes[scheme_b].makespan
              if flowtime_b is None:
                flowtime_b = float('inf')
              if makespan_b is None:
                makespan_b = float('inf')
            if flowtime_a < flowtime_b:
              elo_flowtime.recordMatch(scheme_a, scheme_b, winner=scheme_a)
            elif flowtime_b < flowtime_a:
              elo_flowtime.recordMatch(scheme_a, scheme_b, winner=scheme_b)
            else:
              elo_flowtime.recordMatch(scheme_a, scheme_b, draw=True)
            if makespan_a < makespan_b:
              elo_makespan.recordMatch(scheme_a, scheme_b, winner=scheme_a)
            elif makespan_b < makespan_a:
              elo_makespan.recordMatch(scheme_a, scheme_b, winner=scheme_b)
            else:
              elo_makespan.recordMatch(scheme_a, scheme_b, draw=True)

        for scheme, stats in schemes.items():
          flowtime = stats.flowtime
          makespan = stats.makespan
          if flowtime is None:
            flowtime = float('inf')
          if makespan is None:
            makespan = float('inf')
          if base_flowtime < flowtime:
            win_counts_flowtime[scheme] += 1
          elif base_flowtime == flowtime:
            draw_counts_flowtime[scheme] += 1
          else:
            lose_counts_flowtime[scheme] += 1
          if base_makespan < makespan:
            win_counts_makespan[scheme] += 1
          elif base_makespan == makespan:
            draw_counts_makespan[scheme] += 1
          else:
            lose_counts_makespan[scheme] += 1

          flowtime_performance = stats.flowtime_ratio
          makespan_performance = stats.makespan_ratio
          success = stats.success
          data.append([problem_set, idx, rename_scheme(scheme), flowtime_performance, makespan_performance, success])

  df = pd.DataFrame(data, columns=data_columns)
  # plot_barhist(df, 'flowtime')
  # plot_barhist(df, 'makespan')
  plot_barhist(df, 'success')
  plot_pareto(df, 'flowtime', 'makespan')

  # print('\section{Flowtime}')
  # print('\\begin{tabular}{l|ccc}')
  # print('{\\bf %s opponent} & {\\bf Win rate} & {\\bf Lose rate} & {\\bf Draw rate}\\\\' % rename_scheme(baseline))
  # print('\hline')
  # for scheme in _SCHEMES:
  #   w = win_counts_flowtime[scheme]
  #   l = lose_counts_flowtime[scheme]
  #   d = draw_counts_flowtime[scheme]
  #   t = w + l + d
  #   print('%s & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\' % (rename_scheme(scheme), w / t * 100., l / t * 100., d / t * 100.))
  # print('\\end{tabular}')

  # print('\section{Makespan}')
  # print('\\begin{tabular}{l|ccc}')
  # print('{\\bf %s opponent} & {\\bf Win rate} & {\\bf Lose rate} & {\\bf Draw rate}\\\\' % rename_scheme(baseline))
  # print('\hline')
  # for scheme in _SCHEMES:
  #   w = win_counts_makespan[scheme]
  #   l = lose_counts_makespan[scheme]
  #   d = draw_counts_makespan[scheme]
  #   t = w + l + d
  #   print('%s & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\' % (rename_scheme(scheme), w / t * 100., l / t * 100., d / t * 100.))
  # print('\\end{tabular}')
  # print()

  # print('Flowtime:')
  # for scheme in _SCHEMES:
  #   w = win_counts_flowtime[scheme]
  #   l = lose_counts_flowtime[scheme]
  #   d = draw_counts_flowtime[scheme]
  #   t = w + l + d
  #   print('  {} vs. {} => W: {:.2f}%, L: {:.2f}%, D: {:.2f}%'.format(
  #         rename_scheme(baseline), rename_scheme(scheme), w / t * 100., l / t * 100., d / t * 100.))
  # print('Makespan:')
  # for scheme in _SCHEMES:
  #   w = win_counts_makespan[scheme]
  #   l = lose_counts_makespan[scheme]
  #   d = draw_counts_makespan[scheme]
  #   t = w + l + d
  #   print('  {} vs. {} => W: {:.2f}%, L: {:.2f}%, D: {:.2f}%'.format(
  #         rename_scheme(baseline), rename_scheme(scheme), w / t * 100., l / t * 100., d / t * 100.))
  # print('ELO ratings (flowtime):')
  # for k, v in elo_flowtime.getRatingList():
  #   print('  {}: {:.0f}'.format(rename_scheme(k), v))
  # print('ELO ratings (makespan):')
  # for k, v in elo_makespan.getRatingList():
  #   print('  {}: {:.0f}'.format(rename_scheme(k), v))

  plt.tight_layout()
  plt.show()
