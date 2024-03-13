#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

# Check if output file is provided
if [ -z "$2" ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

# Sort lines alphanumerically and store in new file
sort "$1" > "$2"
