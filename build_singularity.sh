#!/bin/bash
#
# Build a Singularity container, with a log file.
#
# Usage:
# ./build_singularity.sh
#

sudo singularity build singularity.sif singularity/build.def 2>&1 | tee singularity/build.log
