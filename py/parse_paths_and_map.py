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
import pdb

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

def parse_map(mapfile):
    '''
    takes in a mapfile and returns a parsed np array
    '''
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

def parse_path(pathfile):
    '''
    reads a txt file of paths for each agent, returning a dictionary mapping timestep->position of each agent
    inputs: pathfile (string)
    outputs: (T,N,2) np.darray: where is each agent at time T?
    '''
    # save dimensions for later array saving
    w = h = 0
    # maps timesteps to a list of agent coordinates
    timestepsToMaps = defaultdict(list)
    # get max number of timesteps by counting number of commas
    maxTimesteps = 0
    with open(pathfile, 'r') as fd:
        linenum = 0
        for line in fd.readlines():
            if linenum == 0: 
                linenum += 1
                continue # ignore dimension line
            timesteps = 0
            for c in line:
                if c == ',': timesteps += 1
            maxTimesteps = max(maxTimesteps, timesteps)
            linenum += 1
    print(linenum, maxTimesteps)
    # get path for each agent and update dictionary of maps accordingly
    with open(pathfile, 'r') as fd:
        linenum = 0
        for line in fd.readlines():
            if linenum == 0: # parse dimensions of map and keep going
                line = line[:-1]
                line = line.split(",")
                w = int(line[0])
                h = int(line[1])
                linenum += 1
                continue
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
                x = int(temp[0][1:])
                y = int(temp[1][:-1])
                # if you're at the last coordinate then append it to the rest of the maps
                if i == len(rawCoords) - 1:
                    while i != maxTimesteps:
                        timestepsToMaps[i].append([x, y])
                        i += 1
                else: timestepsToMaps[i].append([x, y])
            linenum += 1
    
    # make each map a np array
    for key in timestepsToMaps:
        timestepsToMaps[key] = np.asarray(timestepsToMaps[key])
    
    # make this t x n x 2
    res = []
    for i in range(len(timestepsToMaps)):
        res.append(timestepsToMaps[i])

    t, n = len(res), len(res[0])

    # TODO and then make a t x w x h and return that too
    res2 = np.zeros((t, w, h))
    inc = 0
    for time in range(t):
        arr = res[time]
        for agent in range(n):
            width, height = arr[agent]
            res2[time][width][height] = 1

    res = np.asarray(res)
    print(t, w, h, n)

    # res2 = [[[1 if [width, height] in res[time] else 0 for width in range(w)] for height in range(h)] for time in range(t)]
    return res, res2

def parse_bd(bdfile):
    '''
    parses a txt file of bd info for each agent
    input: bdfile (string)
    output: (N,W,H)
    '''
    timetobd = defaultdict(list)
    w, h = None, None
    with open(bdfile, 'r') as fd:
        agent = 0
        linenum = 0
        for line in fd.readlines():
            if linenum == 0: # parse dimensions and keep going
                line = line[:-1]
                line = line.split(",")
                w = int(line[0])
                h = int(line[1])
                linenum += 1
                continue
            line = line[:-2]
            heuristics = line.split(",")
            timetobd[agent] = heuristics
            agent += 1
    for key in timetobd:
        timetobd[key] = np.asarray(timetobd[key])
    
    # make this n x w x h (right now is n x wh)
    res = []
    for i in range(len(timetobd)):
        res.append(timetobd[i])
    res = np.asarray(res)
    
    return res

def batch_map(dir):
    '''
    goes through a directory of maps, parsing each one and saving to a dictionary
    input: directory of maps (string)
    output: dictionary mapping filenames to parsed maps
    '''

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
            raise RuntimeError("bad map dir")
    return res

def batch_bd(dir):
    '''
    goes through a directory of bd outputs, parsing each one and saving to a dictionary
    input: directory of backward djikstras (string)
    output: dictionary mapping filenames to backward djikstras
    '''
    res = {} # string->np
    # iterate over files in directory, parsing each map
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the bd file and add to a global dictionary (or some class variable dictionary)
            val = parse_bd(f)
            bdname, agents = (filename.split(".txt")[0]).split(".scen") # e.g. "Paris_1_256-random-110, where 1 is instance, 10 is agents"
            res[bdname + agents] = val # TODO make sure that filename doesn't have weird chars you don't want in the npz
            print(f)
        else:
            raise RuntimeError("bad bd dir")
    return res

def batch_path(dir):
    '''
        goes through a directory of outputted EECBS paths, 
        returning a dictionary of tuples of the map name, bd name, and paths dictionary
        NOTE we assume that the file of each path is formatted as 'raw_data/paths/mapnameandbdname.txt'
        NOTE and also that bdname has agent number grandfathered into it
    '''
    res = [] # list of (mapname, bdname, int->np.darray dictionary)
    # iterate over files in directory, making a tuple for each
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the path file, index out its map name, seed, agent number, and bd that it was formed from based on file name, 
            # and add the resulting triplet to a global dictionary (or some class variable dictionary)
            raw = filename.split(".txt")[0] # remove .txt
            seed = raw[-1]
            raw = raw[:-1]
            mapname = raw.split("-")[0] + ".map"
            bdname = raw
            val = parse_path(f) # get the path dict
            # print(mapname, bdname, seed, val)
            res.append((mapname, bdname, seed, val))
            print(f)
        else:
            raise RuntimeError("bad path dir")
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
    print(maps)

    # # TODO parse each bd, add to global dict
    bds = batch_bd(bdIn)
    print(bds)

    # TODO parse each path, add to global list of data
    data = batch_path(pathsIn)

    # send each map, each bd, and each tuple representing a path + instance to npz
    print([(d[0], d[1], d[2]) for d in data])

if __name__ == "__main__":
    main()