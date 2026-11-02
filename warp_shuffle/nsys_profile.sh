#!/bin/sh
mkdir -p profile
nsys profile --force-overwrite=true -o ./profile/warp_shuffle_report ./debug/warp_shuffle
