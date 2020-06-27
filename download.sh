#!/usr/bin/env bash

TagModel=https://www.dropbox.com/s/cwqbnojlnadjb41/Tagbert.bin?dl=1

PredictModel=https://www.dropbox.com/s/0nomgboldn46z7a/PredictBert.bin?dl=1

wget "${TagModel}" -O TagBert.bin
wget "${PredictModel}" -O PredictBert.bin
