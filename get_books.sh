#!/bin/bash
set -euo pipefail

BUCKET="$1"
BOOK_DIR="$2"

mkdir -p "$BOOK_DIR"

aws s3 cp "s3://$BUCKET" "$BOOK_DIR" --recursive --exclude "*" --include "*.txt"

echo "File path" > book_list.txt

find "$(realpath "$BOOK_DIR")" -name "*.txt" | LC_ALL=C sort >> book_list.txt

NUM_BOOKS=$(tail -n +2 book_list.txt | wc -l)
echo "$NUM_BOOKS books listed in book_list.txt"

