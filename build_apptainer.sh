#!/bin/bash
#
# Build a Singularity / Apptainer container, with a log file.
#
# Usage:
# ./build_apptainer.sh
#

sudo apptainer build apptainer.sif apptainer/build.def 2>&1 | tee apptainer/build.log
