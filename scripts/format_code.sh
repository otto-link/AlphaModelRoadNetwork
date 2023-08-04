#!/bin/bash

# python
DIRS="tools"

yapf -vv --in-place --recursive main.py

for i in ${DIRS}; do
    yapf -vv --in-place --recursive "$i/."
done

# clean-up
find . -type f -name 'yapf*.py' -exec rm -f {} +
