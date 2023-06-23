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
1. make it a batch that accepts directories instead of files
2. dictionary that maps map_name to map
3. dictionary that maps bd_name to bd
4. write tuples of (map name, bd name, paths np) to npz
'''

# TODO make this inherit torch.utils.data.Dataset
# will probably be helpful for future pipeline purposes when we want to use pytorch
class PipelineDataset():
    # instantiate class variables
    def __init__(self):
        # TODO implement
        pass
   
    # get number of instances in total (length of training data)
    def __len__(self):
        return len(self.data)

    # return the data for a particular instance
    def __getitem__(self, idx):
        # TODO implement
        pass

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
    return mapdata

# reads a txt file of paths for each agent, returning a dictionary mapping timestep->position of each agent
def parse_path(pathfile):
    # maps timesteps to a list of agent coordinates
    timestepsToMaps = defaultdict(list)
    # get max number of timesteps by counting number of commas
    maxTimesteps = 0
    with open(pathfile, 'r') as fd:
        for line in fd.readlines():
            timesteps = 0
            for c in line:
                if c == ',': timesteps += 1
            maxTimesteps = max(maxTimesteps, timesteps)
    # get path for each agent and update dictionary of maps accordingly
    with open(pathfile, 'r') as fd:
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

# parses a txt file of bd info for each agent
def parse_bd(bdfile):
    # TODO implement
    res = defaultdict(list)
    with open(bdfile, 'r') as fd:
        agent = 0
        for line in fd.readlines():
            line = line[:-2]
            heuristics = line.split(",")
            res[agent] = heuristics
            agent += 1
    for key in res:
        res[key] = np.asarray(res[key])
    return res


# goes through a directory of maps, parsing each one and saving to a dictionary
def batch_map(dir):
    res = {} # string->np
    # iterate over files in directory, parsing each map
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        print(f)
        # checking if it is a file
        if os.path.isfile(f):
            if ".DS_Store" in f: continue # deal with invisible ds_store file
            # parse the map file and add to a global dictionary (or some class variable dictionary)
            val = parse_map(f)
            res[filename] = val # TODO make sure that filename doesn't have weird chars you don't want in the npz
        else:
            print("bad map dir")
            return False
    return res

# goes through a directory of bd outputs, parsing each one and saving to a dictionary
def batch_bd(dir):
    res = {} # string->np
    # iterate over files in directory, parsing each map
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the bd file and add to a global dictionary (or some class variable dictionary)
            val = parse_bd(f)
            res[filename] = val # TODO make sure that filename doesn't have weird chars you don't want in the npz
            print(f)
        else:
            print("bad bd dir")
            return False
    return res

# TODO goes through a directory of outputted EECBS paths, 
# returning a dictionary of tuples of the map name, bd name, and paths dictionary
# NOTE we assume that the file of each path is formatted as 'raw_data/paths/mapnameandbdname.txt'
# NOTE and also that bdname has agent number grandfathered into it
def batch_path(dir):
    res = [] # list of (mapname, bdname, int->np.darray dictionary)
    # iterate over files in directory, making a tuple for each
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # TODO parse the path file, regex its map name and bd that it was formed from based on file name, 
            # and add the resulting triplet to a global dictionary (or some class variable dictionary)
            raw = filename.split("and") # isolate map name, bd name
            mapname, bdname = raw[0], raw[1]
            val = parse_path(f) # get the path dict
            res.append((mapname, bdname, val))
            print(f)
        else:
            print("bad path dir")
            return False
    return res

def main():
    # cmdline argument parsing: take in dirs for paths, maps, and bds, and where you want the outputted npz
    parser = argparse.ArgumentParser()
    parser.add_argument("pathsIn", help="directory containing txt files of agents and paths taken", type=str)
    parser.add_argument("bdIn", help="directory containing txt files with backward djikstra output", type=str) 
    parser.add_argument("mapIn", help="directory containing txt files with obstacles", type=str) 
    npzMsg = "output file with maps, bds as name->array dicts, along with (mapname, bdname, path) triplets for each EECBS run"
    parser.add_argument("npzOut", help=npzMsg, type=str) 

    args = parser.parse_args()
    pathsIn = args.pathsIn
    bdIn = args.bdIn
    mapIn = args.mapIn
    npzOut = args.npzOut

    # instantiate global variables that will keep track of each map and bd that you've encountered
    maps = {} # maps mapname->np array containing the obstacles in map
    bds = {} # maps bdname->np array containing bd for each agent in the instance (NOTE: keep track of number agents in bdname)
    data = [] # contains all run instances, in the form of (map name, bd name)

    # TODO parse each map, add to global dict
    maps = batch_map(mapIn)

    # error handling
    if not maps:
        print("error parsing maps")
        return

    # TODO parse each bd, add to global dict
    bds = batch_bd(bdIn)
    print(bds)

    # error handling
    if not bds:
        print("error parsing bds")
        return

    # TODO parse each path, add to global list of data
    data = batch_path(pathsIn)

    # error handling
    if not data:
        print("error parsing data")
        return

    # send each map, each bd, and each tuple representing a path + instance to npz
    print(data)

if __name__ == "__main__":
    main()