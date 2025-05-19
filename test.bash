#!/bin/bash

# Print the directory structure from the current directory
# Works in Git Bash or WSL on Windows

print_tree() {
	local indent="$2"
	local dir="$1"
	echo "${indent}$(basename "$dir")/"
	local item
	for item in "$dir"/*; do
		if [ -d "$item" ]; then
			print_tree "$item" "  $indent"
		elif [ -f "$item" ]; then
			echo "${indent}  $(basename "$item")"
		fi
	done
}

print_tree "$(pwd)" ""