#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import numpy as np
import pandas as pd
import argparse
import copy
import pprint as pp

# ============================================================================
# parse command line options, using argparseq
# ============================================================================

def auto_int(x):
    return int(x,0)

def to_upper(s):
    return s.upper()

def auto_split(x):
    return x.split()

version = "%(prog)s 1.0"

usage = '''
%(prog)s [opts] <latency csv> .....
'''

description = '''
Gather latency stats from qaic-runner/api-test per-inference latency output (--aic-profiling-format=latency).
If the network was compiled with opStats, device side stats will also be included.
'''

epilog = '''
Examples:
    %(prog)s aic-profiling-program-0-latency.txt
    %(prog)s aic-profiling-program-0-latency.txt aic-profiling-program-1-latency.txt
'''

def parse_options():

    parser = argparse.ArgumentParser(usage = usage, description = description,
                epilog = epilog, formatter_class=argparse.RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group('positional parameters')
    group1.add_argument('latFileNames', nargs="+",
                        help='''One or more latency profiling file produced by qaic-runner/api-test with --aic-profiling-format=latency.
                        This can be from a compile either with or without opStats.
                        Device-side latencies are only available with opStats.
                        '''
                        )

    group2 = parser.add_argument_group('optional parameters')
    group2.add_argument('-k', '--keepOnlyMiddlePct', type=int, default=None,
                        help='''Retain only the middle keep_pct of the inferences.
                        Intended to eliminate any leading and trailing edge effects.
                        ''')

    group3 = parser.add_argument_group('options/flags')
    group3.add_argument('-v', '--verbosity', type=int, default=2,
                        help='''Verbosity range from 0-10. Intended for debug.
                        (default: %(default)s)
                        ''')
    parser.add_argument('--version', action='version', version=version)

    args = parser.parse_args()

    return args

class NoStatError(Exception):
    '''Raise if the requested statName is not found'''

class InferenceLatencies:

    def __init__(self, latFile, cvt2ms=True):
        # skip the header
        for l in range(0,4):
            dummy = latFile.readline()
        self.df = pd.read_csv(latFile, skipinitialspace=True)
        # remove whitespace from colpun names
        self.df.columns = self.df.columns.str.strip()
        if cvt2ms:
            self.cvt_us_to_ms()

    def append(self, other):
        '''Merge in inferences from another InferenceLatencies instance.'''
        self.df = pd.concat([self.df, other.raw_latencies()], ignore_index=True)

    def raw_latencies(self):
        '''Return dataframe with all latencies for each inference.'''
        return self.df

    def num_inferences(self):
        '''Return dataframe with all latencies for each inference.'''
        return len(self.df.index)

    def cvt_us_to_ms(self):
        '''Convert all columns in uS to mS & rename headings.'''
        columns = self.df.columns
        for column in columns:
            if column.endswith('Us'):
                try:
                    self.df.loc[:, column] = self.df[column].apply(lambda x: x / 1000.0)
                except TypeError:
                    # skip N/A's
                    pass
                self.df.rename( columns = {column : column[:-2]}, inplace=True )

    def keep_only_middle(self, keep_pct=50):
        '''Retain only the middle keep_pct of the inferences.
        Intended to eliminate any leading and trailing edge effects.
        '''
        prune_pct = 1.0 - (keep_pct / 100.0)
        prune_size = int(round( (len(self.df) * prune_pct) / 2.0 ))
        self.df = self.df[prune_size:-prune_size]

    def stats(self, df=None, percentiles = [.5, .75, .90, .95, .99]):
        '''Return dataframe containing min/max/etc for each individual latency stat.'''
        if df is None:
            df = self.df

        df = df.drop(['Sample', 'queueDepth', 'numInfInQueue'], axis=1)
        stats = df.describe(percentiles=percentiles)
        stats = stats.drop(['count', 'std'], axis=0)
        stats = stats.transpose()

        return(stats)

    def stat(self, statName, stats_df=None):
        '''Return dataframe containing min/max/etc for the named latency stat.'''
        if stats_df is None:
            stats_df = self.stats()
        try:
            stat = stats_df.loc[statName]
        except KeyError as e:
            raise NoStatError(e)
        return stat

    def print_df(self, df=None):
        if df is None:
            df = self.df
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print(df)


def main():
    args = parse_options()

    allLat = None

    for latFileName in args.latFileNames:

        with open(latFileName, 'r') as f:
            lat = InferenceLatencies(f)

        if args.verbosity >= 5:
            print(latFileName)

        if args.verbosity >= 9:
            lat.print_df(lat.raw_latencies())

        if args.keepOnlyMiddlePct is not None:
            lat.keep_only_middle(keep_pct=args.keepOnlyMiddlePct)
            if args.verbosity >= 9:
                lat.print_df(lat.raw_latencies())

        if args.verbosity >= 5:
            lat.print_df(lat.stats())

        if allLat is None:
            allLat = copy.deepcopy(lat)
        else:
            allLat.append(lat)

    print("All activations combined:")
    stats_df = allLat.stats()
    allLat.print_df(stats_df)

    statName = 'totalRoundtripTime'
    stat = allLat.stat(statName, stats_df=stats_df)
    if args.verbosity >= 5:
        allLat.print_df(stat)

    percentile = 'mean'
    ms = stat[percentile]
    print("{}: percentile({}) = {:.3f} ms".format(statName, percentile, ms))



if __name__ == "__main__":


    main()
