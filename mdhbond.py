"""mdhbonds: performs hbonds analysis between specific groups
   Copyright (C) 2023  Yevhen Kustovskiy
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. 
   
   A copy of GNU license as well as NumPy and Pandas license is available at:
   
   https://github.com/YevhenKustovskiy/md-scripts/edit/main/LICENSE   
   
   Contact email: ykustovskiy@gmail.com
   
   Requirements: script was succesfully implemented with 
   Python 3.10, NumPy 1.23.1, Pandas 1.5.3, MDAnalysis 2.5.0
                                                                          """
   
# Standard
import os
import warnings
import argparse

# Third party
import numpy as np
import pandas as pd

from MDAnalysis import Universe
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

parser = argparse.ArgumentParser(description="Performs hbonds analysis between specific groups")
# example of input (Windows): mdhbond.py -s md.tpr -f sys_md.trr -angl 150.0 -dist 3.5 -groups "protein" "resname IVM" -bytime -bytype -v -nowarn
# example of input (Linux/Ubuntu): python3 mdhbond.py -s md.tpr -f sys_md.trr -angl 150.0 -dist 3.5 -groups "protein" "resname IVM" -bytime -bytype -v -nowarn

# Define topology and trajectory files
parser.add_argument("-s", help="Define topology file name (e.g., tpr, pdb, gro)")
parser.add_argument("-f", help="Define trajectory file name (e.g., xtc, trr)")

parser.add_argument("-groups", help="Specify groups to use in analysis", nargs=2)
parser.add_argument("-dist", help="Specify distance cutoff", type=float, default=3.0)
parser.add_argument("-angl", help="Specify angle cutoff", type=float, default=150.0)
parser.add_argument("-upsel", help="Whether to update selections each frame", type=bool, default=True)

# Trajectory handling
parser.add_argument("-start", help="Specify the first frame of trajectory analysis", type=int)
parser.add_argument("-stop", help="Specify the last frame of trajectory analysis", type=int)
parser.add_argument("-step", help="Specify the number of frames to skip between each analysed frame", type=int)

# Postprocessing
parser.add_argument("-bytime", help="Provide total number of hbonds per timestep", action="store_true")
parser.add_argument("-bytype", help="Provide total number of hbonds per type", action="store_true")
parser.add_argument("-byids", help="Provide total number of hbonds per id", action="store_true")

parser.add_argument("-v", help="Provide the information about progress", action="store_true")
parser.add_argument("-post", help="Use for the postprocessing of existing hbonds.csv file", action="store_true")
parser.add_argument("-nowarn", help="Mute warnings which may occure during evaluation", action="store_true")


args = parser.parse_args()

# Global variables
CONSTRUCTOR = {}
ARGUMENTS = {}

DISTANCE = 4
ANGLE = 5

# Subclassing HydrogenBondAnalysis
class HBAnalysis(HydrogenBondAnalysis):
    def __init__(self,
                 universe,
                 donors_sel=None,
                 hydrogens_sel=None,
                 acceptors_sel=None,
                 between=None,
                 d_h_cutoff=1.2,
                 d_a_cutoff=3.0,
                 d_h_a_angle_cutoff=150,
                 update_selections=True,
                 ):

        HydrogenBondAnalysis.__init__(self,
                                      universe,
                                      donors_sel,
                                      hydrogens_sel,
                                      acceptors_sel,
                                      between,
                                      d_h_cutoff,
                                      d_a_cutoff,
                                      d_h_a_angle_cutoff,
                                      update_selections)

    def _single_frame(self) -> None:
        super()._single_frame()
        self._conclude()

    # reimplementing _conclude() method to handle large trajectories
    def _conclude(self) -> None:

        # Call original method to convert list of results to numpy array
        super()._conclude()

        # Append data to dataframe
        df = pd.DataFrame(self.results.hbonds[:, :DISTANCE].astype(int),
                          columns=["Frame",
                                   "Donor_ix",
                                   "Hydrogen_ix",
                                   "Acceptor_ix", ])

        df["Distances"] = self.results.hbonds[:, DISTANCE]
        df["Angles"] = self.results.hbonds[:, ANGLE]

        df["Donor resname"] = self.u.atoms[df.Donor_ix].resnames
        df["Acceptor resname"] = self.u.atoms[df.Acceptor_ix].resnames
        df["Donor resid"] = self.u.atoms[df.Donor_ix].resids
        df["Acceptor resid"] = self.u.atoms[df.Acceptor_ix].resids
        df["Donor name"] = self.u.atoms[df.Donor_ix].names
        df["Acceptor name"] = self.u.atoms[df.Acceptor_ix].names

        # Append dataframe to csv file
        kwargs = {"sep": ";", "index": False, "mode": "a"}

        # Check if file exists; if exists - write no headers
        if os.path.isfile("hbonds.csv"):
            kwargs["header"] = None

        df.to_csv("hbonds.csv", **kwargs)

        # Clear data on hydrogen bonds found at this frame
        self.results.hbonds = [[], [], [], [], [], []]

    # Reads specified columns of hbonds.csv into pandas dataframe
    # and converts it to numpy array
    @staticmethod
    def _read_results(cols: list[str], sep: str) -> np.ndarray:
        return pd.read_csv("hbonds.csv", usecols=cols, sep=sep).to_numpy()

    # Reimplementing count_by_time() method to work with "hbonds.csv"
    def count_by_time(self) -> None:

        # Read frames from hbonds.csv
        array = self._read_results(cols=["Frame"],
                                   sep=";")

        indices, tmp_counts = np.unique(array, return_counts=True)
        indices = indices.astype("float64")

        # If postprocessing without calling run method
        # take start and step from trajectory
        if not hasattr(self,"start") or not hasattr(self,"step"):
            self._setup_frames(self._trajectory)

        indices -= self.start
        indices /= self.step

        counts = np.zeros_like(self.frames)
        counts[indices.astype(np.intp)] = tmp_counts

        # Instead of returning hbonds number per each timestep
        # create dataframe and save to csv
        df = pd.DataFrame()
        df.index.name = "Frames"
        df["Number"] = counts

        df.to_csv("hbonds_by_time.csv", sep=";")

    # Reimplementing count_by_type() method to work with "hbonds.csv"
    def count_by_type(self) -> None:

        # Read Donor_ix and Acceptor_ix columns from hbonds.csv
        array = self._read_results(cols=["Donor_ix",
                                         "Acceptor_ix"],
                                   sep=";")


        d = self.u.atoms[array[:, 0].astype(np.intp)]
        a = self.u.atoms[array[:, 1].astype(np.intp)]

        # Original code
        if hasattr(d, "resnames"):
            d_res = d.resnames
            a_res = a.resnames
        else:
            d_res = len(d.types) * ["None"]
            a_res = len(a.types) * ["None"]

        tmp_hbonds = np.array([d_res, d.types, a_res, a.types], dtype=str).T
        hbond_type, type_counts = np.unique(
            tmp_hbonds, axis=0, return_counts=True)
        hbond_type_list = []
        for hb_type, hb_count in zip(hbond_type, type_counts):
            hbond_type_list.append([":".join(hb_type[:2]),
                                    ":".join(hb_type[2:4]), hb_count])

        hbond_type_list = np.asarray(hbond_type_list)
        # Instead of returning create dataframe and save to csv
        df = pd.DataFrame(hbond_type_list)

        df.index.name = "No."
        df["Donor(Residue:Atom)"] = hbond_type_list[:, 0]
        df["Acceptor(Residue:Atom)"] = hbond_type_list[:, 1]
        df["Number"] = hbond_type_list[:, 2]

        df.to_csv("hbonds_by_type.csv", index=False, sep=";")

    # Reimplementing count_by_ids() method to work with "hbonds.csv"
    def count_by_ids(self) -> None:
        # Read Donor_ix, Hydrogen_ix, and Acceptor_ix columns from hbonds.csv
        array = self._read_results(cols=["Donor_ix",
                                         "Hydrogen_ix",
                                         "Acceptor_ix"],
                                   sep=";")

        d = self.u.atoms[array[:, 0].astype(np.intp)]
        h = self.u.atoms[array[:, 1].astype(np.intp)]
        a = self.u.atoms[array[:, 2].astype(np.intp)]

        atom_names = np.asarray([d.resnames, d.resids, d.names, d.ids,
                                 h.ids,
                                 a.resnames, a.resids, a.names, a.ids],
                                dtype=str).T

        hbond_type_list = []
        # Joining names of residues and atoms with ix
        for hb_type in atom_names:
            hbond_type_list.append([":".join(hb_type[0:4]),
                                    hb_type[4],
                                    ":".join(hb_type[5:9])])

        hbonds_type_array = np.asarray(hbond_type_list)

        # Original code
        tmp_hbonds = np.array([hbonds_type_array[:,0],
                               hbonds_type_array[:,1],
                               hbonds_type_array[:,2]]).T
        hbond_ids, ids_counts = np.unique(tmp_hbonds, axis=0,
                                          return_counts=True)

        # Find unique hbonds and sort rows so that most frequent observed bonds are at the top of the array
        unique_hbonds = np.concatenate((hbond_ids, ids_counts[:,None]),
                                       axis=1)

        unique_hbonds = unique_hbonds[unique_hbonds[:, 3].astype("int32").argsort()[::-1]]

        # Instead of returning create dataframe and save to csv
        df = pd.DataFrame()

        df.index.name = "No."
        df["Donor(resname:resid:atomname:ix)"] = unique_hbonds[:, 0]
        df["Hydrogen_ix"] = unique_hbonds[:, 1]
        df["Acceptor(resname:resid:atomname:ix)"] = unique_hbonds[:, 2]
        df["Number"] = unique_hbonds[:, 3]

        df.to_csv("hbonds_by_ids.csv", sep=";")

def unpack():
    """
    unpack kwargs for class constructor
    """

    CONSTRUCTOR["between"] = [args.groups]
    CONSTRUCTOR["d_a_cutoff"] = args.dist
    CONSTRUCTOR["d_h_a_angle_cutoff"] = args.angl
    CONSTRUCTOR["update_selections"] = args.upsel

    """
    unpack kwargs for run function 
    """

    if args.start:
        ARGUMENTS["start"] = int(args.start)

    if args.stop:
        ARGUMENTS["stop"] = int(args.stop)

    if args.step:
        ARGUMENTS["step"] = int(args.step)

    if args.v:
        ARGUMENTS["verbose"] = args.v

if __name__ == "__main__":

    print("Creating universe... Processing large trajectory files may take a while.")
    try:
        system = Universe(args.s, args.f, refresh_offsets=True)
    except Exception as e:
        print(e)
    else:
        print("Universe created!")

    unpack()

    if args.nowarn:
        warnings.filterwarnings("ignore")
        print("Warnings are muted!")

    hbonds = HBAnalysis(system, **CONSTRUCTOR)

    if not args.post:
        hbonds.run(**ARGUMENTS)

    if args.bytime:
        print("Postprocessing... Counting hbonds by timestep...")
        hbonds.count_by_time()

    if args.bytype:
        print("Postprocessing... Counting hbonds by type...")
        hbonds.count_by_type()

    if args.byids:
        print("Postprocessing... Counting hbonds by ids...")
        hbonds.count_by_ids()



