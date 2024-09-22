import os
import mat4py
import fnmatch
import mat73
import numpy as np
import pickle
from scipy import io


class DataLoader:
    def __init__(self, root_data_path, mouse_name, session_id):
        """class to easily transform matlab .mat data files into Python dictionaries

        TODO: allow dataset to be a list. also modify loadData for this case where you can specify if you want all of the same things from each dataset
              or specific things from each

        Args:
            root_data_path (str): users path to folder where ALL data is stored, which contains subfolders per mouse eg 'oh28' (this does not point to an animal folder)
            mouse_name (str): name of mouse, eg 'oh28'
            session_id (int): day number 0 - 7
        """
        self.mouse = mouse_name
        self.data_path = os.path.join(root_data_path, mouse_name)
        self.dataset = np.sort(
            [file for file in os.listdir(self.data_path) if (file.startswith(
                "oh") or file.startswith("ph"))]
        )[session_id]
        self.path = os.path.join(self.data_path, self.dataset)
        self.file_patterns = {
            "resampled": "resampled_*.mat",
            "dff": "rec_*.mat",
            "cascade": "cascade_*.mat",
            "cell_classification": "class_*.mat",
            "ctxstr": "ctxstr.mat",
            "dlc": "dlc.mat",
            "skeleton": "skeleton.mat",
            "match": "new_all_matches.mat",
        }
        self.valid_data_names = [
            "dff_ctx",
            "dff_str",
            "cascade_ctx",
            "cascade_str",
            "behavior",
            "dlc",
            "skeleton",
            "resampled",
            "match",
        ]

    def loadData(self, which="all"):
        """loads data from files

        Args:
            which (str, optional): which files to load. options can be found in class variable valid_data_names. defaults to 'all'.

        Raises:
            ValueError: raised if input is not a valid data name.

        Returns:
            list: List containing all data files requested in dictionary format.
        """
        if which == "all":
            which = self.valid_data_names
        elif isinstance(
            which, str
        ):  # if only one of the data types is reqested. this avoids making input a list with only one element
            which = [which]
        returns = []
        for w in which:
            if w not in self.valid_data_names:
                raise ValueError(
                    f"{w} is not a valid data name. See DataLoader.valid_data_names"
                )
            elif "dff" in w:
                region = w[-3:] # ctx or str
                data = self.loadNeuralData(region, kind="dff")
            elif "cascade" in w:
                region = w[-3:]
                data = self.loadNeuralData(region, kind="cascade")
            elif w == "behavior":
                data = self.loadBehavioralData()
            elif w == "resampled":
                data = self.loadResampledData()
            elif w == "match":
                data = self.loadMatches()

            else:  # for files that are simple structs with arrays
                d = self.loadmat(self.getFilePath(w))
                if "info" in d.keys():
                    del d["info"]
                data = {k: np.array(d[k]).squeeze() for k in d.keys()}
            returns.append(data)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    def loadResampledData(self):
        """generates a python dictionary with the contents of resampled_data.mat files

        Returns:
            dict: data
        """
        # load the resampled_data.mat file using the keyword associated with "resampled"
        # TEMPORARY TWEEK: return None in order to be able to skip sessions that currently have no "resmapled"
        picklefile = os.path.join(self.path, "resampled.pkl")
        if os.path.exists(picklefile):
            data = pickle.load(open(picklefile, "rb"))
            return data

        files = self.getFilePath("resampled")
        if files is None:
            return None
        data = self.loadmat(files)
        for k, v in data.items():
            # format the "list of lists" values in the dictionary (make a list of arrays, one for each trial)
            if "_by_trial" in k:
                data[k] = [np.array(_).squeeze() for _ in v]
            # convert lists into arrays (ctx_traces, str_traces, t)
            elif k in ["ctx_traces", "str_traces", "t"]:
                data[k] = np.array(v).squeeze()
            # change indices to begin at 0
            elif "_info" in k:
                v["ind2rec"] = [ind[0] - 1 for ind in v["ind2rec"]]
                v["rec2ind"] = [ind[0] - 1 for ind in v["rec2ind"]]

                # striatum files also have tdTomato pos/neg index lists
                if isinstance(v['tdt'], dict):
                    for key, val in v["tdt"].items():
                        v["tdt"][key] = [ind - 1 for ind in val]
            elif k == "st_trial_inds":
                data[k] = [ind - 1 for ind in v]
            else:
                continue
        # format behavioral data ('session' key)
        data["session"]["behavior"] = self.formatBehavioralData(
            data["session"]["behavior"]
        )
        # format the trial separated data
        data["trials"] = self.formatTrialData(data["trials"])

        pickle.dump(data, open(picklefile, "wb"))
        return data

    def loadBehavioralData(self):
        """formats the behavioral portion of ctxstr.mat files. all matlab structs become dictionaries and matlab arrays become numpy arrays

        Returns:
            dict: contains behavioral data
        """
        if self.getFilePath("ctxstr") is None:
            resampled = self.loadResampledData()
            behavior = (
                resampled["session"]["behavior"] if resampled is not None else None
            )
            return behavior

        ctxstr = self.loadmat(self.getFilePath("ctxstr"))
        behavior = ctxstr["behavior"]
        position = behavior.pop("position")
        position_dict = {
            "position_by_trial": [
                np.array(position["by_trial"][n]).squeeze()
                for n in range(len(position["by_trial"]))
            ],
            "position_cont": np.array(position["cont"]),
            "us_threshold": position["us_threshold"],
        }
        for k in behavior.keys():
            behavior[k] = np.array(behavior[k]).squeeze()
        return {**behavior, **position_dict}

    def loadNeuralData(self, region, kind="dff", returns="all"):
        """loads neural traces from matlab file

        Args:
            region (str): ctx or str
            kind (str, optional): option for convolved traces (uses 'cascade'). Defaults to 'dff'.
            returns (str, optional): what data to return— if 'traces' only the neural data will be returned. Defaults to 'all'.

        Returns:
            dict: contains requested files in a dictionary. If returns = 'all' dictionary includes neural traces and all of the files under region folder in ctxstr.mat
        """
        trace_data = self.loadmat(self.getFilePath(kind, region))
        if kind == "dff":
            trace_data["traces"] = np.array(trace_data["traces"]).squeeze()
            trace_data["filters"] = np.array(trace_data["filters"]).squeeze()
            traces = trace_data["traces"]
        elif kind == "cascade":
            trace_data["spike_probs"] = np.array(
                trace_data["spike_probs"]).squeeze()
            traces = trace_data["spike_probs"]

        if returns == "traces":
            return {"traces": traces}
        elif returns == "all":
            exp_data = self.loadmat(self.getFilePath("ctxstr"))[region]
            exp_data = {k: np.array(exp_data[k]).squeeze()
                        for k in exp_data.keys()}

            return {**trace_data, **exp_data}

    def loadMatches(self):
        """ """
        match = mat73.loadmat(self.data_path + "/new_all_matches_v2.mat")

        ctx = match["new_ctx_matches"]
        ctx_match = [ctx[di][0] for di in range(8)]
        str = match["new_str_matches"]
        str_match = [str[di][0] for di in range(8)]
        data = {}
        data["ctx_matches"] = ctx_match
        data["str_matches"] = str_match
        return data

    def formatBehavioralData(self, d):
        position = d.pop("position")
        position_dict = {
            "position_by_trial": [
                np.array(position["by_trial"][n]).squeeze()
                for n in range(len(position["by_trial"]))
            ],
            "position_cont": np.array(position["cont"]),
            "us_threshold": position["us_threshold"],
        }
        for k in d.keys():
            d[k] = np.array(d[k]).squeeze()
        return {**d, **position_dict}

    def formatTrialData(self, d):
        for k, v in d.items():
            # reindex to start at 0
            if k == "ind":
                d[k] = [ind - 1 for ind in v]
            # convert lists of lists into lists of arrays
            elif k in ["lick_times", "position", "velocity", "times"]:
                d[k] = [np.array(arr).squeeze() for arr in v]
            # convert dlc for each trial
            elif k == "dlc":
                # each item in this list is a dlc matlab struct for each trial
                for dlc in d[k]:
                    for key, val in dlc.items():
                        dlc[key] = np.array(val).squeeze()
            # so far this motion struct only has the motion onset times ---> just convert to a list of motion onsets
            elif k == "motion":
                d[k] = [motion["onsets"] for motion in v]

        return d

    def getFilePath(self, file, folder=None):
        """searches in your directory for a certain file and return path

        Args:
            file ('str'): one of the keys in self.file_patterns (TODO this may be redundant. could just have a list of file patterns and bypass the file names)
            folder (str, optional): 'ctx' or 'str'. If looking for neural data file, need to specify looking in the ctx or str folder. Defaults to None.

        Raises:
            Exception: No file was found
            Exception: Multiple files found with the given type

        Returns:
            'str': path to the specified file
        """
        if folder:
            path = os.path.join(self.path, folder)
        else:
            path = self.path
        files = self.findFile(self.file_patterns[file], path)
        if not files:
            # TEMPORARY TWEEK: return None in order to be able to skip sessions that currently have no "resmapled"

            # raise Exception(f"No file found for keyword {self.file_patterns[file]}")
            return None
        if len(files) > 1:
            raise Exception(
                f"More than one file found. File keyword {self.file_patterns[file]} not specific enough. Returned first file match"
            )
        return files[0]

    def findFile(self, pattern, path):
        """finds a file beginning with a certain pattern by walking through all subdirectories in specified path

        Args:
            pattern (str): identifiable pattern in the file name. e.g. 'rec_...'
            path (str): current path

        Returns:
            list: list of files matching the pattern
        """
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def loadmat(self, path):
        """loads a .mat file into a python dictionary

        Args:
            path (str): path of .mat file

        Returns:
            dict: contains contets of matlab struct. Typically, matlab arrays are returned as python lists and structs are returned as dictionaries.
        """
        try:
            dict = mat4py.loadmat(path)
        except Exception as error:
            print(error)
            print("opened using mat73")
            dict = mat73.loadmat(path)

        return dict

    @staticmethod
    def getSessions(root_data_path, mouse_name):
        data_path = os.path.join(root_data_path, mouse_name)
        datasets = np.sort(
            [file[-4:] for file in os.listdir(data_path) if (file.startswith(
                "oh") or file.startswith("ph"))]
        )
        return datasets

    def get_trial_traces(data):
        """
        return the traces_by_trial for both ctx and str AFTER removing either empty trials or trials with nan in them

        """
        ctx_trials = data["ctx_traces_by_trial"]
        str_trials = data["str_traces_by_trial"]
        t_trials = data["time_by_trial"]
        ntrials = len(t_trials)

        ctx_has_nans = [
            np.isnan(ctx_trials[ni]).any() or len(
                ctx_trials[ni].flatten()) == 0
            for ni in range(ntrials)
        ]
        str_has_nans = [
            np.isnan(str_trials[ni]).any() or len(
                str_trials[ni].flatten()) == 0
            for ni in range(ntrials)
        ]
        good_trials = [
            not ctx_has_nans[ni] and not str_has_nans[ni] for ni in range(ntrials)
        ]

        ctx_trials = [ctx_trials[ti]
                      for ti in range(ntrials) if good_trials[ti]]
        str_trials = [str_trials[ti]
                      for ti in range(ntrials) if good_trials[ti]]
        t_trials = [t_trials[ti] for ti in range(ntrials) if good_trials[ti]]
        ntrials = len(t_trials)
        return ctx_trials, str_trials, t_trials, ntrials


# ########## DEBUGGING ###########
# data_path = "Data/oh28"
# data_set = "oh28-0209"

# dl = DataLoader(data_path, data_set)
# which = "match"
# match = dl.loadData(which)


# print(match["session_names"])
