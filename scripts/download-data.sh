#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(dirname $(realpath $0))
REPO_DIR=$(dirname $SCRIPT_DIR)
DATA_DIR=$REPO_DIR/data

mkdir -p $DATA_DIR/SBIC.v2
cd $DATA_DIR

wget -nc -q http://maartensap.com/social-bias-frames/SBIC.v2.tgz
tar -xzf SBIC.v2.tgz -C SBIC.v2
cd SBIC.v2
mv SBIC.v2.trn.csv train.csv
mv SBIC.v2.dev.csv dev.csv
mv SBIC.v2.tst.csv test.csv
rm *agg*

echo "Done!"