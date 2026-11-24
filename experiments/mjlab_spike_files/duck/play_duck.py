#!/usr/bin/env python3
"""`play` with the duck task registered. Example (dummy agent, no checkpoint needed):

  cd ~/src/mjlab_spike/duck && ../mjlab/.venv/bin/python play_duck.py \
      Mjlab-Tracking-Flat-Open-Duck-Mini --agent zero --num-envs 4 \
      --motion-file /path/to/motion.npz --viewer viser
"""

import sys

import duck_tasks  # noqa: F401  (registers the task)
from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
