import argparse
import msgpack
import msgpack_numpy

import run


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.

  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--input', action='store', default=None, help='Where to the results are stored.')
  args = parser.parse_args()

  all_results = run.read_results(args.input)
  for args, results in all_results.items():
    print('Results for', args)
    for stats in results:
      run.print_stats(run.Statistics(*stats))
