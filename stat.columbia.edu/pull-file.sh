#!/bin/sh

# use wget to automatically get files from the course website


wget -r -np -nd -l 1 -A ipynb https://stat.columbia.edu/~cunningham/teaching/

echo "Getting files"

rm robots.txt.tmp
