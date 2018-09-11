"""Converts an image to an environment that can be used within generate_problems_from_environment.py.

Usage:
python3 convert_image.py --image=image.png
"""

import argparse
import matplotlib.pylab as plt
import numpy as np
import os
from PIL import Image


def load_image(filename) :
  img = Image.open(filename)
  img.load()
  return np.asarray(img, dtype=np.uint8)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--image', action='store', default=None, help='Path to image.')
  parser.add_argument('--output', action='store', default=None, help='Path to image.')
  args = parser.parse_args()

  image = load_image(args.image)
  assert len(image.shape) == 3, 'Wrong format.'
  if image.shape[-1] == 4:
    image = image[:, :, :3]
  assert image.shape[0] != 3, 'Wrong format. Needs RGB.'

  name = os.path.splitext(os.path.basename(args.image))[0]
  size = max(image.shape[0], image.shape[1])
  obstacles = np.where(np.logical_and(image[:, :, 0] < 20, np.logical_and(image[:, :, 1] < 20, image[:, :, 2] < 20)))  # Black.
  start_positions = np.where(np.logical_and(image[:, :, 0] < 150, np.logical_and(image[:, :, 1] > 150, image[:, :, 2] < 50)))  # Green.
  goal_positions = np.where(np.logical_and(image[:, :, 0] > 150, np.logical_and(image[:, :, 1] < 150, image[:, :, 2] < 150)))  # Red.

  if args.output:
    with open(args.output, 'w') as fp:
      fp.write('{}\n'.format(size))
      for x, y in zip(*obstacles):
        fp.write('({},{})\n'.format(x, y))
  else:
    plt.imshow(image)
    plt.show()

  print('To generate 50 problems, run the following command:')
  print('pypy3 scripts/generation/generate_problems_from_environment.py \\\n'
        '  problems/{name} \\\n'
        '  --e {path} \\\n'
        '  --n 50 --r 10 --sizes \'[1,1,2,2,3,3,4,4,5,5]\' --speeds 1. \\\n'
        '  --startranged \'{startx}\' \'{starty}\' \\\n'
        '  --goalranged \'{goalx}\' \'{goaly}\''.format(
            name=name, path=args.output,
            startx=start_positions[0].tolist(), starty=start_positions[1].tolist(),
            goalx=goal_positions[0].tolist(), goaly=goal_positions[1].tolist()))
