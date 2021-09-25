#!/usr/bin/env python3
'''
Test if there are any duplicates in the data sets.
author: Daniel Nichols
date: September 2021
'''
# std imports
from argparse import ArgumentParser
import glob
from os.path import join as path_join
from json import load as json_load

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
    parser.add_argument('-f', '--full-check', action='store_true', help='check all fields')
    parser.add_argument('--dataset', type=str, default='./raw', help='dataset path')
    return parser.parse_args()


def get_data(root, build_full=False, verbose=False):
    dataset = []
    for path in glob.glob(path_join(root, '*.json')):
        with open(path, 'r') as fp:
            obj = json_load(fp)
            obj['path'] = path

            if build_full:
                obj['full'] = '\n'.join(obj['ingredients'])
                obj['full'] += '\n'
                obj['full'] += '\n'.join(obj['steps'])
            
            dataset.append(obj)

    return dataset


def check_datapoint(i, dataset, full_check=False, verbose=False):
    for j in range(i+1, len(dataset)):
        x = dataset[i]
        point = dataset[j]

        if x['url'] == point['url']:
            return x

        if full_check:
            if x['full'] == point['full']:
                return x
    
    return None


def main():
    args = get_args()

    dataset = get_data(args.dataset, build_full=args.full_check, verbose=args.verbose)

    for i in range(len(dataset)):
        dup = check_datapoint(i, dataset, full_check=args.full_check, verbose=args.verbose)
        if dup:
            print('{} and {} match'.format(dataset[i]['path'], dup['path']))
            exit(1)
    


if __name__ == '__main__':
    main()
