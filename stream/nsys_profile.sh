#!/bin/sh
mkdir -p profile
# for tests except testMultiAddStream
# nsys profile -t cuda --force-overwrite=true -o ./profile/stream_report_serial ./debug/stream
# nsys profile -t cuda --force-overwrite=true -o ./profile/stream_report_concurrent ./debug/stream multi

#for test testMultiAddStream
nsys profile -t cuda --force-overwrite=true -o ./profile/stream_default ./debug/stream
nsys profile -t cuda --force-overwrite=true -o ./profile/stream_nonblocking ./debug/stream nonblocking
