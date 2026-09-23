#!/usr/bin/env bash
# Create the two convenience symlinks, AFTER their targets exist (Decision 7).
#
# The physical root is /weka/projects/bvandur1/zhuicon1/rewrite-3b (117 TB free).  The scratch
# paths the project was originally specified with are kept working as symlinks.
set -euo pipefail

DATA=/weka/projects/bvandur1/zhuicon1/rewrite-3b
S=/weka/scratch/jhu/bvandur1/zhuicon1/datasets

link() {  # link <target> <linkname>
  local target="$1" name="$2"
  [[ -d "$target" ]] || { echo "SKIP $name -> $target (target does not exist yet)"; return 0; }
  if [[ -L "$name" ]]; then
    echo "exists  $name -> $(readlink "$name")"; return 0
  fi
  if [[ -e "$name" ]]; then
    if [[ -d "$name" && -z "$(ls -A "$name")" ]]; then
      rmdir "$name"
    else
      echo "REFUSING to replace non-empty $name"; return 1
    fi
  fi
  ln -s "$target" "$name"
  echo "linked  $name -> $target"
}

mkdir -p "$S"
link "$DATA/00_pool"  "$S/dclm-refinedweb-100m-sample"
link "$DATA/dataset"  "$S/rewrite-3b-llama-60b-tokens"
