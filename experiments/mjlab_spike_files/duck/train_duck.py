#!/usr/bin/env python3
"""`train` with the duck task registered. Example:

  cd ~/src/mjlab_spike/duck && ../mjlab/.venv/bin/python train_duck.py \
      Mjlab-Tracking-Flat-Open-Duck-Mini \
      --env.commands.motion.motion-file /path/to/motion.npz \
      --env.scene.num-envs 4096 --agent.max-iterations 3000
"""

import sys

import duck_tasks  # noqa: F401  (registers the task)
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
