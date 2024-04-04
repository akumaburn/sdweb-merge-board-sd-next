import os

from modules import sd_models, extras, shared
try:
    from modules import hashes
    from modules.sd_models import CheckpointInfo
except:
    pass

from scripts.multimerge.recipe import MergeRecipe

class MergeOperation:
    def __init__(self):
        self.recipes = {}
        self.last_output_id = None  # Track the last output identifier
        self.second_last_output_id = None  # Track the second-to-last output identifier

    def can_process(self):
        _ret = True
        for _index, _recipe in self.recipes.items():
            _recipe: MergeRecipe = _recipe
            _ret = _ret and _recipe.can_process()
        return _ret

    def add_merge(self, index, A, B, C, M, S, F, O, CF):
        if index and index != "" and index >= 0:
            _recipe = MergeRecipe(A, B, C, O, M, S, F, CF)
            if _recipe.can_process():
                self.recipes.update({index: _recipe})

    def get_process_num(self):
        return len(self.recipes)

    def run_merge(self, skip_merge_if_exists=False, config_source=0, save_metadata=False):
        _ret_all = []
        _vars = {}
        for _index, _recipe in sorted(self.recipes.items()):
            _recipe: MergeRecipe = _recipe
            # Apply current variables
            _recipe.apply_variables(_vars)
            # Run merge
            _ret = _recipe.run_merge(_index, skip_merge_if_exists, config_source, save_metadata)
            _ret_all.append(_ret)
            # Update vars
            _vars.update(_recipe.get_vars())

            # Use the second-to-last output id to retrieve its path and delete the file
            if self.second_last_output_id:
                ckpt_info = sd_models.get_closet_checkpoint_match(self.second_last_output_id)
                if ckpt_info and hasattr(ckpt_info, "path") and os.path.exists(ckpt_info.path):
                    os.remove(ckpt_info.path)
                    print(f"Deleted second-to-last output: {ckpt_info.path}")

            # Update the output identifier tracking
            self.second_last_output_id = self.last_output_id
            self.last_output_id = _ret[1] if len(_ret) > 1 else None

        return _ret_all