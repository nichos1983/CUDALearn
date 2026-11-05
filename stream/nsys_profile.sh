#!/bin/sh
mkdir -p profile
nsys profile --force-overwrite=true -o ./profile/stream_report_serial ./debug/stream
nsys profile --force-overwrite=true -o ./profile/stream_report_concurrent ./debug/stream multi
