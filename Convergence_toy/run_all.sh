#!/bin/bash

# Change to the directory where the folders are located

# Loop through each item in the current directory
for dir in */ ; do
    # Check if it is a directory and not the __pycache__ directory
    if [ -d "$dir" ] && [ "$dir" != "__pycache__/" ]; then
        echo "Entering directory $dir"
        cd "$dir"
        
        # Check if run.py exists in the directory
        if [ -f run.py ]; then
            # Run the python job
            python -u run.py -n 100000
        else
            echo "run.py not found in $dir"
        fi
        
        # Return to the parent directory
        cd ..
    fi
done
