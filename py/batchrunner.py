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
import shutil

mapsToNumAgents = {
    "Paris_1_256": (10, 50), # Verified
    "random-32-32-20": (50, 409), # Verified
    "random-32-32-10": (50, 461), # Verified
    "den520d": (50, 1000), # Verified
    "den312d": (10, 200), # Verified
    "empty-32-32": (50, 511), # Verified
    "empty-48-48": (50, 1000), # Verified
    "ht_chantry": (50, 1000), # Verified
}

class BatchRunner:
    def __init__(self, mapfile, scenfile, timeLimit, suboptimality, batchFolderName):
        self.mapfile = mapfile
        self.scenfile = scenfile
        self.timeLimit = timeLimit
        self.suboptimality = suboptimality
        self.batchFolderName = batchFolderName

    def runSingleSettingsOnMap(self, numAgents):
        for aSeed in self.seeds:
            command = "./build_release/eecbs -m {} -a {}".format(self.mapfile, self.scenfile)
            scensplit = self.scenfile.split(".scen")[0]
            scensplit = scensplit.split("/")[-1]
            print(scensplit)
            # command += " --outputPaths={}".format("./raw_data/paths/" + "scen" + scensplit + "numAg" + str(numAgents) + "sD" + str(aSeed) + ".txt")
            # command += " --outputPaths={}".format("./raw_data/paths/" + scensplit + ".txt")
            command += " --outputPaths={}".format("./raw_data/paths/" + scensplit + str(numAgents) + str(aSeed) + ".txt") # TODO format based on instance + agents + seed
            command += " --seed={} -k {} -t {}".format(aSeed, numAgents, self.timeLimit)
            command += " --suboptimality={}".format(self.suboptimality)
            command += " --batchFolder={} -o allresults.csv".format(self.batchFolderName)
            subprocess.run(command.split(" "), check=False) # True if want failure error
            # delete file if not max agents on the map setting
            mapname = self.mapfile.split('.')[0].split("/")[-1]
            print(mapname)
            maxagents = mapsToNumAgents[mapname][1]
            if numAgents != maxagents:
                # DELETE THE FILE
                os.remove("./raw_data/bd/"+ self.scenfile.split("/")[-1] + str(numAgents) + ".txt")

    def detectAllFailed(self, numAgents):
        """
        select all runs with the same hyperparameters (number agents, map, so)
        if a set of hyperparams fails for all seeds, stop for all future # agents
        """

        curOutputFile = "logs/{}/allresults.csv".format(self.batchFolderName) # TODO change allresults to more descriptive name
        if exists(curOutputFile):
            df = pd.read_csv(curOutputFile)
            df = df[(df["agent num"] == numAgents)]
            df = df[(df["suboptimality"] == self.suboptimality)]
            df = df[(df["instance name"] == self.scenfile)]
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

    for aSo in [1.5]: # TODO fix; how do we treat suboptimality
        aName = "base_so{}".format(aSo)
        ExpSettings[aName] = dict(
            timeLimit=60,
            suboptimality=aSo,
        )
        myExps.append(aName)

    dateString = ""
    batchFolderName = "GridSubOptFast/GridSOFast{}_{}".format(dateString, curMap)

    seeds = [1, 2, 3]
    scens = [1, 2, 3] # TODO should we run all these scenes?
    for expName in myExps:
        print("Starting ExpSettings: {}".format(expName))
        exp = ExpSettings[expName]
        for aScenNum in scens:
            print("  Starting scen number: {}".format(aScenNum))

            myBR = BatchRunner("{}/{}.map".format(mapfolder, curMap), "{}/{}-random-{}.scen".format(scenfolder, curMap, aScenNum),
                exp["timeLimit"], exp["suboptimality"], batchFolderName)
            myBR.runBatchExps(rangeOfAgents, seeds)

    # make a folder with all the current map info
    os.mkdir("./raw_data/" + curMap)
    shutil.move("./raw_data/bd/", "./raw_data/" + curMap)
    shutil.move("./raw_data/paths/", "./raw_data/" + curMap)

    # replace old bd and paths folder
    os.mkdir("./raw_data/bd")
    os.mkdir("./raw_data/paths")

if __name__ == "__main__":
    main()
