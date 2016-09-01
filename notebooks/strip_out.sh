#!/usr/bin/env python

import sys
import os
import subprocess

path = '.'
if len(sys.argv) >= 2:
    path = sys.argv[1]

for dirpath, dirnames, filenames in os.walk(path):
    if os.path.basename(dirpath) != '.ipynb_checkpoints':
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext == '.ipynb':
                filepath = os.path.join(dirpath, filename)
                cmd = ('jupyter-nbconvert --inplace --to notebook '
                       '--ClearOutputPreprocessor.enabled=True {} --output {}')
                subprocess.check_call(cmd.format(filepath, filepath).split())
