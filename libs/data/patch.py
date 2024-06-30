from dataclasses import dataclass
from typing import Any, List

@dataclass
class CandidatePatch():

	patch_index: int = None
	set_index: int = None
	
	patch: str = None

	multi_chunk_key: str = None
	start_line: int = None
	end_line: int = None

	module_id: str = None
	module_name: str = None
	module_num: str = None

	buggy_dir: str = None
	module_dir: str = None

	file_key: str = None
	file_name: str = None
	file_ext: str = None

	buggy_code: str = None

	exp_ins: int = None
	per_ins: int = None

	exp_other: int = None
	per_other: int = None
	
	diffs: Any = None
	action_judgements: Any = None

	tf_tokens: List[str] = None
	tf_token_length: int = None

	criteria_tokens: List[str] = None
	criteria_token_length: int = None

	action_score: int = None
	ngram_score: int = None

	score_total: int = None

@dataclass
class OptimizationOption:
	datasets_type: str = None

	json_path: str = None
	multi_chunk_file: str = None

	patches_dir: str = None
	patches_file: str = None

	filtered_patches_dir: str = None
	filtered_patches_file: str = None
	
	ranked_patches_dir: str = None
	ranked_patches_file: str = None

	combination_patches_dir: str = None