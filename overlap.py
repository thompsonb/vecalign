#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import argparse

from dp_utils import yield_overlaps


def go(output_file, input_files, num_overlaps):
    output = set()
    for fin in input_files:
        lines = open(fin, 'rt', encoding="utf-8").readlines()
        for out_line in yield_overlaps(lines, num_overlaps):
            output.add(out_line)

    # for reproducibility
    output = list(output)
    output.sort()

    with open(output_file, 'wt', encoding="utf-8") as fout:
        for line in output:
            fout.write(line + '\n')


def _main():
    parser = argparse.ArgumentParser('Create text file containing overlapping sentences.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--inputs', type=str, nargs='+',
                        help='input text file(s).')

    parser.add_argument('-o', '--output', type=str,
                        help='output text file containing overlapping sentneces')

    parser.add_argument('-n', '--num_overlaps', type=int, default=4,
                        help='Maximum number of allowed overlaps.')

    args = parser.parse_args()
    go(output_file=args.output,
       num_overlaps=args.num_overlaps,
       input_files=args.inputs)


if __name__ == '__main__':
    _main()
