import os
import sys
import time
from datetime import datetime # For printing current datetime
import subprocess # For executing c++ executable
import numpy as np
import argparse
import pdb
import pandas as pd
from os.path import exists
from collections import defaultdict

'''
0. parse bd (fix eecbs by having it make another txt output, parse it here)
    note that each agent has its own bd for a given map
1. write the map, timestep arrays to np instead of csv
2. make it a batch that accepts directories instead of files
3. dictionary that maps map_name to map
4. dictionary that maps bd_name to bd
'''

# Read map
def parse_map(mapfile):
    # mapfile = "../random-32-32-20.map"
    with open(mapfile) as f:
        line = f.readline()  # "type octile"

        line = f.readline()  # "height 32"
        height = int(line.split(' ')[1])

        line = f.readline()  # width 32
        width = int(line.split(' ')[1])

        line = f.readline()  # "map\n"
        assert(line == "map\n")

        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width,height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    print(type(mapdata))
    return mapdata

def parse_paths(file):
    # maps timesteps to a list of agent coordinates
    timestepsToMaps = defaultdict(list)
    # get max number of timesteps by counting number of commas
    maxTimesteps = 0
    with open(file, 'r') as fd:
        for line in fd.readlines():
            timesteps = 0
            for c in line:
                if c == ',': timesteps += 1
            maxTimesteps = max(maxTimesteps, timesteps)
    # get path for each agent and update dictionary of maps accordingly
    with open(file, 'r') as fd:
        for line in fd.readlines():
            i = 0
            # omit up to the first left paren
            while line[i] != '(': i += 1
            # omit the ending space and final arrow to nothing
            line = line[i:-3]
            # get list of coordinates, as raw (x, y) strings
            rawCoords = line.split("->")
            # add the coordinates to the dictionary of maps
            for i, coord in enumerate(rawCoords):
                temp = coord.split(',')
                x = temp[0][1:]
                y = temp[1][:-1]
                # if you're at the last coordinate then append it to the rest of the maps
                if i == len(rawCoords) - 1:
                    while i != maxTimesteps:
                        timestepsToMaps[i].append([x, y])
                        i += 1
                else: timestepsToMaps[i].append([x, y])
    
    # make each map a np array
    for key in timestepsToMaps:
        timestepsToMaps[key] = np.asarray(timestepsToMaps[key])

    return timestepsToMaps

# going to load some number of instances together, consisting of a path and map
def batch_path(path):
    pass

def main():
    # cmdline argument parsing: take in a txt file of paths for each agent, map file, and where you want the outputted formatted files
    parser = argparse.ArgumentParser()
    parser.add_argument("pathsIn", help="txt file of agents and paths taken", type=str) # Note: Positional is required, make not position by changing to --yKey
    parser.add_argument("mapIn", help="map file with obstacles", type=str) # Note: Positional is required, make not position by changing to --yKey
    parser.add_argument("pathsOut", help="map file to output map of agents for each timestep", type=str) # Note: Positional is required, make not position by changing to --yKey
    parser.add_argument("mapOut", help="map file to output obstacles", type=str) # Note: Positional is required, make not position by changing to --yKey

    args = parser.parse_args()

    # get a dictionary mapping timesteps to positions of each agent
    parsed_paths = parse_paths(args.pathsIn)
    parsed_map = parse_map(args.mapIn)

    # send the parsed map and file to npz

if __name__ == "__main__":
    main()