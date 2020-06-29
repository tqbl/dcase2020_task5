#!/bin/bash

set -e

command -v curl >/dev/null 2>&1 || { echo 'curl is missing' >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo 'tar is missing' >&2; exit 1; }

mkdir -p _dataset && cd _dataset

curl -fOL 'https://zenodo.org/record/3873076/files/README.md'
curl -fOL 'https://zenodo.org/record/3873076/files/annotations.csv'
curl -fOL 'https://zenodo.org/record/3873076/files/dcase-ust-taxonomy.yaml'
curl -fOL 'https://zenodo.org/record/3873076/files/unpack_audio.sh'
curl -fOL 'https://zenodo.org/record/3873076/files/audio-eval-0.tar.gz'
curl -fOL 'https://zenodo.org/record/3873076/files/audio-eval-1.tar.gz'
curl -fOL 'https://zenodo.org/record/3873076/files/audio-eval-2.tar.gz'
curl -fOL 'https://zenodo.org/record/3873076/files/audio.tar.gz'

bash unpack_audio.sh
