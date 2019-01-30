#!/bin/bash

python3 run.py \
  --problem=problems/** \
  --communication_radius=30,50 \
  --mode="['dynamic']" \
  --scheme="[('longest_priority', 'tiebreak_random'), \
             ('forward_looking_priority', 'tiebreak_longest_first'), \
             ('forward_looking_priority', 'tiebreak_random'), \
             ('naive_forward_looking_priority', 'tiebreak_longest_first'), \
             ('surroundings_10_priority', 'tiebreak_longest_first'), \
             ('naive_surroundings_10_priority', 'tiebreak_longest_first'), \
             ('surroundings_20_priority', 'tiebreak_longest_first'), \
             ('naive_surroundings_20_priority', 'tiebreak_longest_first'), \
             ('surroundings_30_priority', 'tiebreak_longest_first'), \
             ('naive_surroundings_30_priority', 'tiebreak_longest_first'), \
             ('random_priority', 'tiebreak_random')]" \
  --output_results=results.bin
sudo poweroff
