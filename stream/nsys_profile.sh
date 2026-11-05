#!/bin/sh
mkdir -p profile
nsys profile --force-overwrite=true -o ./profile/stream_report ./debug/stream
