#!/bin/sh
mkdir -p profile
nsys profile --force-overwrite=true -o ./profile/shared_mem_report ./debug/shared_mem
