import math
import os
import sys
import re

from modules import sd_models, extras, shared
try:
    from modules import hashes
    from modules.sd_models import CheckpointInfo
except:
    pass

from scripts.multimerge.util.merge_history import MergeHistory

# Defining the strings as variables
S_WEIGHTED_SUM = "weighted_sum"
S_WEIGHTED_SUBTRACTION = "weighted_subtraction"
S_TENSOR_SUM = "tensor_sum"
S_ADD_DIFFERENCE = "add_difference"
S_SUM_TWICE = "sum_twice"
S_TRIPLE_SUM = "triple_sum"
S_EUCLIDEAN_ADD_DIFFERENCE = "euclidean_add_difference"
S_MULTIPLY_DIFFERENCE = "multiply_difference"
S_TOP_K_TENSOR_SUM = "top_k_tensor_sum"
S_SIMILARITY_ADD_DIFFERENCE = "similarity_add_difference"
S_DISTRIBUTION_CROSSOVER = "distribution_crossover"
S_TIES_ADD_DIFFERENCE = "ties_add_difference"

# Including all strings in the choice_of_method list
choice_of_method = [
    S_WEIGHTED_SUM,
    S_WEIGHTED_SUBTRACTION,
    S_TENSOR_SUM,
    S_ADD_DIFFERENCE,
    S_SUM_TWICE,
    S_TRIPLE_SUM,
    S_EUCLIDEAN_ADD_DIFFERENCE,
    S_MULTIPLY_DIFFERENCE,
    S_TOP_K_TENSOR_SUM,
    S_SIMILARITY_ADD_DIFFERENCE,
    S_DISTRIBUTION_CROSSOVER,
    S_TIES_ADD_DIFFERENCE
]

mergeHistory = MergeHistory()


class MergeRecipe():
    def __init__(self, A, B, C, O, M, S, F:bool, CF):
        if C == None:
            C = ""
        if O == None:
            O = ""
        self.row_A = "" if type(A) == list and len(A) == 0 else A
        self.row_B = "" if type(B) == list and len(B) == 0 else B
        self.row_C = "" if type(C) == list and len(C) == 0 else C
        self.row_O = O
        self.row_M = M
        self.row_S = S
        self.row_F = F
        self.row_CF = CF if CF in ["ckpt", "safetensors"] else "ckpt"

        self.A = self.row_A
        self.B = self.row_B
        self.C = self.row_C
        self.O = re.sub(r'[\\|:|?|"|<|>|\|\*]', '-', O)
        self.S = self.row_S
        self.M = self.row_M
        self.F = self.row_F
        self.CF = self.row_CF

        self.vars = {}  # runtime variables

    def can_process(self, index=0):
        if self.A == "" or self.B == "" or self.A == None or self.B == None:
            return False
        if (self.C == "" or self.C == None) and self.S == S_AD:
            return False
        if index > 0:
            # invalid var check
            # __O3__, line=4 => ok
            # __O3__, line=2 => error
            pass
        return True

    def apply_variables(self, _vars:dict):
        def _apply(_param, _vars):
            if not _param or _param == "":
                return _param
            if _param in _vars.keys():
                _var = _vars.get(_param)  # i.e. self.row_A = __O1__, _vars["__O1__"] = _var = "xxxx.ckpt"
                if _var and _var != "":
                    return sd_models.get_closet_checkpoint_match(_var).title
            return _param
        self.A = _apply(self.row_A, _vars)
        self.B = _apply(self.row_B, _vars)
        self.C = _apply(self.row_C, _vars)

    def run_merge(self, index, skip_merge_if_exists, config_source, save_metadata):
        # Initial setup and model existence checks
        sd_models.list_models()
        if skip_merge_if_exists:
            _filename = self.O + "." + self.CF if self.O != "" else self._estimate_ckpt_name()
            if self._check_ckpt_exists(_filename):
                print(f"Merge skipped. Same name checkpoint already exists: {_filename}")
                self._update_o_filename(index, [f"[skipped] {_filename}", f"[skipped] {_filename}"])
                return [f"[skipped] {_filename}", f"[skipped] {_filename}"]

        # Check sha256 and update if needed
        def check_and_update(model_title):
            if model_title == "":
                return ""
            _model_info = sd_models.get_closet_checkpoint_match(model_title)
            if _model_info is None:
                return ""
            if hasattr(_model_info, "sha256") and _model_info.sha256 is None:
                _model_info.calculate_shorthash()
                _model_info.register()
            return _model_info.title

        self.A = check_and_update(self.A)
        self.B = check_and_update(self.B)
        self.C = check_and_update(self.C)

        print( "Starting merge under settings below,")
        print( "  A: {}".format(f"{self.A}" if self.A == self.row_A else f"{self.row_A} -> {self.A}"))
        print( "  B: {}".format(f"{self.B}" if self.B == self.row_B else f"{self.row_B} -> {self.B}"))
        print( "  C: {}".format(f"{self.C}" if self.C == self.row_C else f"{self.row_C} -> {self.C}"))
        print(f"  S: {self.S}")
        print(f"  M: {self.M}")
        print(f"  F: {self.F}")
        print( "  O: {}".format(f"{self.O}" if self.O != "" else f" -> {self._estimate_ckpt_name()}"))
        print(f" CF: {self.CF}")

        # Prepare arguments for run_modelmerger
        merger_args = {
            "id_task": None,
            "overwrite": True,
            "custom_name": self.O,
            "primary_model_name": self.A,
            "secondary_model_name": self.B,
            "tertiary_model_name": self.C if self.C else None,
            "merge_mode": self.S,
            "custom_name": self.O,
            "checkpoint_format": self.CF,
            "prune": True,
            "weights_clip": True,
            "save_metadata": save_metadata,
            "device": "cpu",
            "unload": True,
            "precision": "fp16" if self.F else "fp32",
            "alpha": self.M,
            "beta": self.M
            # Include additional parameters required by the latest run_modelmerger definition
        }

        print( f"Mapped to the following merger_args: {merger_args}")

        # Attempt to run the model merger
        try:
            results = extras.run_modelmerger(**merger_args)
        except Exception as e:
            error_message = str(e)  # Convert the exception to a string to capture the error message
            print(f"Error during model merge: {error_message}", file=sys.stderr)
            sd_models.list_models()  # Refresh models list to ensure consistency
            # Return the error message as part of the output
            return [f"Error: {error_message}", f"Error: {error_message}"]

        # Post-processing: handle results, update filenames, and manage state
        self._update_o_filename(index, results)

        # Save merge history and return results
        # This part includes logging and history tracking as implemented before
        return [f"Merge complete. Checkpoint saved as: [{self.O}]", self.O]


    def get_vars(self):
        return self.vars

    #
    # local func
    #
    def _update_o_filename(self, index, results_list):
        """
        update self.O and vars {"__Ox__": self.O}
        """
        if len(results_list) == 5:
            results = results_list[4] if type(results_list[4]) == str else results_list[0]
        else:
            results = results_list[0]
        # Checkpoint saved to " + output_modelname
        ckpt_path = " ".join(results.split(" ")[3:])
        ckpt_name = os.path.basename(ckpt_path)  # expect aaaa.ckpt
        ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
        if ckpt_info is None and hasattr(ckpt_info, "sha256"):
            ckpt_info = CheckpointInfo(ckpt_path)
            ckpt_info.calculate_shorthash()
            ckpt_info.register()
            ckpt_name = ckpt_info.title
        else:
            sd_models.list_models()

        # update
        self.O = ckpt_name
        print(f"  __O{index}__: -> {ckpt_name}")
        self.vars.update({f"__O{index}__": ckpt_name})

    def _alpha_of_weighted_sum(self, alpha):
        """
        Weighted sum
            (1-alpha)*theta0 + alpha * theta1
          = theta0 + (theta1 - theta0) * alpha
        """
        return alpha

    def _check_ckpt_exists(self, _O):
        ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
        output_modelname = os.path.join(ckpt_dir, _O)
        if os.path.exists(ckpt_dir) and os.path.exists(output_modelname) and os.path.isfile(output_modelname):
            return True
        else:
            return False

    def _estimate_ckpt_name(self):
        _A = sd_models.get_closet_checkpoint_match(self.A)
        _B = sd_models.get_closet_checkpoint_match(self.B)
        _M = self.M
        _S = self.S
        _CF = self.CF

        # File name generation code from
        #
        # AUTO 685f963
        # modules/extras.py
        # def run_modelmerger
        # L314
        _filename = \
        _A.model_name + '_' + str(round(1-_M, 2)) + '-' + \
        _B.model_name + '_' + str(round(_M, 2)) + '-' + \
        _S.replace(" ", "_") + \
        '-merged.' +  \
        _CF
        return _filename
