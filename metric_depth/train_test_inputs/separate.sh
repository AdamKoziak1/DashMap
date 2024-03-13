#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 input_file none_output_file non_none_output_file"
    exit 1
fi

# Check if none output file is provided
if [ -z "$2" ]; then
    echo "Usage: $0 input_file none_output_file non_none_output_file"
    exit 1
fi

# Check if non-none output file is provided
if [ -z "$3" ]; then
    echo "Usage: $0 input_file none_output_file non_none_output_file"
    exit 1
fi

# Filter lines containing "None" and store in one file, and store the rest in another
grep "None" "$1" > "$2"
grep -v "None" "$1" > "$3"
