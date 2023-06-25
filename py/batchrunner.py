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

mapsToNumAgents = {
    "Paris_1_256": (10, 200), # Verified
    "random-32-32-20": (50, 409), # Verified
    "random-32-32-10": (50, 461), # Verified
    "den520d": (50, 1000), # Verified
    "den312d": (10, 200), # Verified
    "empty-32-32": (50, 511), # Verified
    "empty-48-48": (50, 1000), # Verified
    "ht_chantry": (50, 1000), # Verified
}

class BatchRunner:
    # TODO ADD THE DEFAULTS FOR OPTIMIZATION (make the defaults the offs)
    def __init__(self, mapfile, scenfile, timeLimit, suboptimality, batchFolderName):
        self.mapfile = mapfile
        self.scenfile = scenfile
        self.timeLimit = timeLimit
        self.suboptimality = suboptimality
        self.batchFolderName = batchFolderName

    def runSingleSettingsOnMap(self, numAgents):
        mods_str = "with_mods" if self.all_optimizations else "no_mods"
        for aSeed in self.seeds:
            command = "./build_release/eecbs -m {} -a {}".format(self.mapfile, self.scenfile)
            command += " --seed={} -k {} -t {}".format(aSeed, numAgents, self.timeLimit)
            command += " --r_val={} --w_h={}".format(self.r, self.w_h)
            command += " --suboptimality={}".format(self.suboptimality)
            command += " --batchFolder={} -o allresults{}{}{}.csv".format(self.batchFolderName, self.r, self.w_h, mods_str)
            command += " --reuse_toggle={}".format(self.reuse_toggle)

            subprocess.run(command.split(" "), check=False) # True if want failure error
    
    def detectAllFailed(self, numAgents):
        """
        select all runs with the same hyperparameters (r, w_h, number agents, toggle, map, so)
        if a set of hyperparams fails for all seeds, stop for all future # agents
        """ 

        curOutputFile = "logs/{}/allresults{}{}{}.csv".format(self.batchFolderName, self.r, self.w_h, "with_mods" if self.all_optimizations else "no_mods")
        if exists(curOutputFile):
            df = pd.read_csv(curOutputFile)
            df = df[(df["r val"] == self.r) & (df["num agents"] == numAgents) & (df["toggle set"] == self.reuse_toggle) \
                    & (df["suboptimality"] == self.suboptimality) & (df["instance name"] == self.scenfile)]
            numFailed = len(df[(df["solution cost"] <= 0) | (df["solution cost"] >= 1073741823)])
            return len(df) == numFailed
        return False
    
    def runBatchExps(self, agentNumbers, seeds):

        self.seeds = seeds
        for aNum in agentNumbers:
            print("    Number of agents: {}".format(aNum))
            self.runSingleSettingsOnMap(aNum)
            if self.detectAllFailed(aNum): # Terminate early
                print("Terminating early because all failed with {} number of agents".format(aNum))
                break

ExpSettings = dict()

def main():
    # TODO: 1) run with random-32-32, 20, 30, 40 agents, 2) check parse, 3) add multiple maps (den312d) with same agent #s, 4) check parse file names again, 5) try pytorch
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="map to run", type=str) # Note: Positional is required, make not position by changing to --yKey

    args = parser.parse_args()

    mapfolder = "data/mapf-map"
    scenfolder = "data/scen-random"
    curMap = args.map # E.g. "Paris_1_256"
    if curMap not in mapsToNumAgents.keys():
        raise KeyError("Unknown map: {}".format(curMap))
    startOffset = 0
    rangeOfAgents = np.arange(mapsToNumAgents[curMap][0] + startOffset, mapsToNumAgents[curMap][1]+1, 10, dtype=int)

    myExps = []

    for aSo in [1]:
        aName = "base_so{}_toggle".format(aSo)
        ExpSettings[aName] = dict(
            timeLimit=60,
            suboptimality=aSo,
            reuse_toggle = 1
        )
        myExps.append(aName)
        bName = "base_so{}_no_toggle".format(aSo)
        ExpSettings[bName] = dict(
            timeLimit=60,
            suboptimality=aSo,
            reuse_toggle = 0
        )
        myExps.append(bName)

    dateString = ""
    batchFolderName = "GridSubOptFast/GridSOFast{}_{}".format(dateString, curMap)
    
    # seeds = [1, 2, 3, 4, 5]
    seeds = [1, 2, 3, 4, 5]
    scens = [1, 2, 3, 4, 5][:1] # Just run one 1st scene

    for expName in myExps:
        print("Starting ExpSettings: {}".format(expName))
        exp = ExpSettings[expName]
        for aScenNum in scens:
            print("  Starting scen number: {}".format(aScenNum))
            
            myBR = BatchRunner("{}/{}.map".format(mapfolder, curMap), "{}/{}-random-{}.scen".format(scenfolder, curMap, aScenNum),
                exp["timeLimit"], exp["suboptimality"], 5, 4, batchFolderName, exp["reuse_toggle"])
            myBR.runBatchExps(rangeOfAgents, seeds)

if __name__ == "__main__":
    main()