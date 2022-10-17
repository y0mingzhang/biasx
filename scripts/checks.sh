#/bin/bash
set -Eeuo pipefail

black src
flake8 src