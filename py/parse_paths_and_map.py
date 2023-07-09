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
import torchvision
from torch.utils.data import Dataset
from collections import defaultdict
import pdb
import math

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
class PipelineDataset(Dataset):        

    # instantiate class variables
    def __init__(self, numpy_data_path, k):
        '''
        INPUT: 
            numpy_data_path: the path to the npz file storing all map, backward dijkstra, and path information across all EECBS runs. (string)
            k: window size. (int)
        contains 3 class variables: self.maps, self.bds, and self.tn2.
        maps: a dictionary mapping mapname (name.map) to the np (w,h) of all obstacle locations. (1s for obstacles, 0s for free space)
            e.g. {"Paris_1_256.map": (256,256)}
        bds: a dictionary mapping backward dijkstra names to the np (n,w,h) of all backward dijkstra calculations for all agents.
            e.g. {"Paris_1_256-random-1{max_agents}": ({max_agents},256,256)}
            naming convention: scen name + # agents
        tn2: a dictionary mapping eecbs instance names to the np (t,n,2) of all paths (x,y) for all agents.
            e.g. {"Paris_1_256.map,Paris_1_256-random-110,2": (t,n,2)}
            naming convention: mapname + "," + bdname + "," + seed        
        '''

        # read in the dataset, saving map, bd, and path info to class variables
        loaded = np.load(numpy_data_path)
        self.parse_npz(loaded)
        self.k = k
   
    # get number of instances in total (length of training data)
    def __len__(self):
        return self.length # go through the tn2 dict with # data and np arrays saved, and sum all the ints

    # return the data for a particular instance: the location, bd, and map
    def __getitem__(self, idx):
        '''
        INPUT: index (must be smaller than len(self))
        OUTPUT: map, bd, and direction
            map: (w,h) TODO: (2k+1, 2k+1)
            bd: (w,h)
            direction: (2)
        TODO: centered version. when passing in the map and bd, return a (2k+1,2k+1) window centered at current location of agent. 
        '''
        if idx >= self.__len__():
            print("Index too large for {}-sample dataset".format(self.__len__()))
            return

        def translate_to_bd(bd):
            bd = bd.split("-random-")
            bd = bd[0] + "-random-" + bd[1][0] + "200"
            return bd
        
        items = list(self.tn2.items())
        tn2ind = 0
        tracker = 0
        while tracker + items[tn2ind][1][0] <= idx:
            tracker += items[tn2ind][1][0] # add number of data in the (t,n,2) matrix
            tn2ind += 1
        # so now tn2ind holds the index to the (t,n,2) matrix containing the data we want
        mapname, bdname, seed = items[tn2ind][0].split(",")
        bdname = translate_to_bd(bdname)
        bd = self.bds[bdname]
        grid = self.maps[mapname]
        # get the location, dir to next location
        newidx = idx - tracker # index within the matrix to get
        paths = items[tn2ind][1][1] # (t,n,2) paths matrix
        t, n, _ = np.shape(paths)
        timestep, agent = newidx // n, newidx % n
        curloc = paths[timestep, agent]
        nextloc = paths[timestep+1, agent] if timestep < t-1 else curloc

        # TODO have ids of 4 closest agents within the window (will be passing in the bds of those 4 agents)
        # 1. get ids of 4 closest agents
        # 2. for all those agents that are in the window, return their bd centered at current location
        # e.g. bd[agent 2, curlocx-k:curlocx+k+1,curlocy-k:curlocy+k+1]
        # return map, bd, and direction TODO get the window in postprocess

        # get the agents within the window (that are not the agent we are looking at),
        windowAgents = list(filter(lambda loc: abs(loc[1][0]-curloc[0]) <= self.k and abs(loc[1][1]-curloc[1]) <= self.k and loc[0] != agent, enumerate(paths[timestep])))
        # and create list of tuples of (euclidean distance from agent,agentnumber)
        windowAgents = list(map(lambda tup: (math.sqrt((tup[1][0]-curloc[0])**2 + (tup[1][1]-curloc[1])**2), tup[0]), windowAgents))
        # get the 4 closest agents, if possible
        windowAgents.sort()
        windowAgents = windowAgents[:4]
        print("WINDOW AGENTS: ", windowAgents)

        # return 2k+1 by 2k+1 grid centered at agent's current location for both the map, bd
        print(len(bd))
        grid = grid[curloc[0]-self.k:curloc[0]+self.k+1, curloc[1]-self.k:curloc[1]+self.k+1]
        dijk = bd[agent][curloc[0]-self.k:curloc[0]+self.k+1, curloc[1]-self.k:curloc[1]+self.k+1]
        helper_bds = [bd[inwindow] for inwindow in windowAgents]

        return grid, dijk, helper_bds, nextloc - curloc
    
    def find_instance(self, idx):
        '''
        returns the backward dijkstra, map, and path arrays, and indices to get into the path array
        '''
        pass
        # TODO implement; essentially lines 68-83 (85? 86?)

    def parse_npz(self, loaded):
        loaded = {k:v for k, v in loaded.items()}
        items = list(loaded.items())
        # print(loaded["Paris_1_256.map,Paris_1_256-random-110,2"]) # testing
        # index -> tuple mapping, finding maps, then bds, then paths
        i = 0
        while "-random-" not in items[i][0]:
            i += 1
        self.maps = dict(items[:i]) # get all the maps
        j = i
        while "," not in items[j][0]:
            j += 1
        self.bds = dict(items[i:j]) # get all the bds
        k = j
        while k < len(items) and "twh" not in items[k][0]:
            k += 1
        self.tn2 = dict(items[j:k]) # get all the paths in (t,n,2) form
        # since the # of data is simply number of agent locations, this is t*n, which we append to the dictionary for each path
        for k, v in self.tn2.items():
            t, n, _ = np.shape(v)
            self.tn2[k] = (t*n, v)
        self.length = sum([value[0] for value in self.tn2.values()])
        # self.twh = dict(items[k:]) # get all the paths in (t,w,h) form



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
    res2 -= 1 # if no agent, -1
    for time in range(t):
        arr = res[time]
        for agent in range(n):
            width, height = arr[agent]
            res2[time][width][height] = agent

    res = np.asarray(res)
    print(t, w, h, n)

    # res2 = [[[1 if [width, height] in res[time] else 0 for width in range(w)] for height in range(h)] for time in range(t)]
    return res

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
        nwh = timetobd[key]
        new = []
        assert(not len(nwh) % w and not len(nwh) % h)
        # transform to n x w x h here, assuming row-major order
        while len(nwh):
            takeaway = nwh[:w]
            new.append(takeaway)
            nwh = nwh[w:]
        timetobd[key] = new
    
    # make this n x w x h from dictionary of n w x h arrays
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
            res[filename] = val 
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
    res1 = {} # dict of (mapname, bdname, int->np.darray dictionary), and is (n, t, 2)
    res2 = {} # dict of (mapname, bdname, int->np.darray dictionary), and is (t, w, h)
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
            val = parse_path(f) # get the 2 types of paths: the first being a list of agent locations for each timestep, the 2nd being a map for each timestep with -1 if no agent, agent number otherwise
            # print(mapname, bdname, seed, np.count_nonzero(val2 != -1)) # debug statement
            print("___________________________\n")
            res1[mapname + "," + bdname + "," + seed] = val
            # res2[mapname + "," + bdname + "," + seed + ",twh"] = val2
            print(f)
        else:
            raise RuntimeError("bad path dir")
    return res1

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
    # print(maps)

    # # # TODO parse each bd, add to global dict
    bds = batch_bd(bdIn)
    # print(bds)

    # TODO parse each path, add to global list of data
    data1 = batch_path(pathsIn)

    # send each map, each bd, and each tuple representing a path + instance to npz
    np.savez_compressed(npzOut, **maps, **bds, **data1) # Note automatically stacks to numpy vectors

    # DEBUGGING: test out the dataloader
    loader = PipelineDataset(npzOut + ".npz", 400)
    print(loader.bds.keys())
    # tn2s = {k:v for k, v in loader.tn2.items()}
    # print(tn2s.keys())
    print(len(loader))
    print("2nd TO LAST ITEM")
    print(loader[24297199])
    print("LAST ITEM")
    print(loader[24297200])


if __name__ == "__main__":
    main()