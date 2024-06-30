import sys
sys.path.append(".")

# Settings
from shutil import copyfile
from functools import reduce
from itertools import product
from typing import Dict, List
import os
from logging import Logger
from pathlib import Path
from libs.data.patch import CandidatePatch, OptimizationOption
from libs.components.apr_directory import APRDirectory
from libs.components.patch_optimizer import compare_diff, filter_candidate_patches, get_top_ks, rank_candidate_patches, map_patches_by_index
from libs.components.custom_logger import get_custom_logger
from libs.components.file_processor import read_java, read_json, write_java, write_json
from libs.data.defects4j import Defects4JModule
from libs.components.patch_optimizer import get_extended_propotional_top_ks, get_propotional_top_k_ranges, get_propotional_top_ks

current_dir = os.getcwd()
parent_dir = Path(current_dir).parent.absolute()
grand_parent_dir = parent_dir.parent.absolute()
apr_dir = APRDirectory(parent_dir)

step_name = "Step6"
file_name = "Optimize_Patch"

logger_name = "_".join([step_name, file_name])
error_logger_name = "_".join([step_name, file_name, "Error"])

logger: Logger = get_custom_logger(logger_name, apr_dir.logs_dir)
error_logger: Logger = get_custom_logger(error_logger_name, apr_dir.logs_dir)

del_token = "<del/>"

# Arguments
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--combination_type', type=str, default="buggy_blocks")
parser.add_argument('--is_propotional_top_k', type=bool, default=True)
parser.add_argument('--without_optimization', type=bool, default=False)
parser.add_argument('--with_calculation', type=bool, default=False)
parser.add_argument('--print_log', type=bool, default=False)
parser.add_argument('--combination_patch_max', type=int, default=20000)
parser.add_argument('--num_candidate_patches', type=int, default=1000)

args = parser.parse_args()

combination_type=args.combination_type
is_propotional_top_k = args.is_propotional_top_k
without_optimization = args.without_optimization
with_calculation = args.with_calculation
print_log = args.print_log
combination_patch_max = args.combination_patch_max
num_candidate_patches = args.num_candidate_patches

# Executions
def main():
    try:
        if combination_type == "buggy_blocks":
            optimize_patches_for_buggy_blocks()
        elif combination_type == "multi_chunks":
            optimize_patches_for_multi_chunks()
        elif combination_type == "without_special_tokens":
            optimize_patches_without_special_tokens()
        elif combination_type == "without_optimization":
            optimize_patches_without_optimization()
        else:
            raise Exception("Enter a supported combineation type.")
    except Exception as ex:
        error_logger.error("Main Exception")
        error_logger.error(ex, exc_info=True, stack_info=True)

def patches_by_file(acc: dict, cur: CandidatePatch):
    file_name = cur.file_name

    if file_name not in acc:
        acc[file_name] = []
    
    acc[file_name] += [cur]

    return acc

def copy_origin_files(modules_dir: str, combination_patches_dir: str, candidate_patches: List[CandidatePatch], module_id: str, combination_index: int):
    for patch in candidate_patches:
        # print(patch.file_name)

        file_name = patch.file_name
        file_ext =  patch.file_ext

        # src_file_name, _ = write_java(combination_patches_dir, file_name, [])
        # logger.info("{}.java".format(src_file_name))
        combination_patch_dir = os.path.join(combination_patches_dir, module_id, str(combination_index))

        write_java(combination_patch_dir, file_name, [])

        src_buggy_file_path = os.path.join(modules_dir, module_id, patch.buggy_dir, patch.module_dir, "{}{}".format(file_name, file_ext))
        tgt_buggy_file_path = os.path.join(combination_patch_dir, "{}_Origin{}".format(file_name, file_ext))

        copyfile(src_buggy_file_path, tgt_buggy_file_path)

        logger.info("Source Buggy File Path : {}".format(src_buggy_file_path))
        logger.info("Target Buggy File Path : {}".format(tgt_buggy_file_path))
        
def apply_patches(combination_patches_dir: str, candidate_patches: List[CandidatePatch], module_id: str, combination_index: int):
    file_names: Dict[str, List[CandidatePatch]] = reduce(patches_by_file, candidate_patches, {})

    combination_patch_dir = os.path.join(combination_patches_dir, module_id, str(combination_index))

    # global_index = 0
    for file_name, file_patches in file_names.items():
        new_lines = []

        sorted_file_patches = sorted(file_patches, key=lambda x: x.start_line)

        # print(sorted_file_patches)

        for index, file_patch in enumerate(sorted_file_patches):
            origin_file_name = "{}_Origin".format(file_name)
            revised_file_name = file_name

            lines = read_java(combination_patch_dir, origin_file_name)

            # file_patch = candidate_patches[global_index]

            if index == 0:
                start_line = 0
                end_line = file_patch.start_line - 1
            else:
                prev_file_patch = sorted_file_patches[index - 1]
                start_line = prev_file_patch.end_line
                end_line = file_patch.start_line - 1
            
            new_lines += lines[start_line:end_line]

            logger.info("Index {} : Lines {} - {}".format(index, start_line, end_line))

            new_patch = file_patch.patch

            if new_patch != del_token:
                new_lines += [new_patch]

            final_end_line = file_patch.end_line

            if index == (len(sorted_file_patches) - 1):
                new_lines += lines[final_end_line:]
                logger.info("Index {} : Lines {} - ".format(index, final_end_line))
                logger.info("Index {} is the end".format(index))

        src_file_name, item_len = write_java(combination_patch_dir, revised_file_name, new_lines)
        logger.info("{} : {} data".format(src_file_name, item_len))
                
        # Generate Diff
        origin_path = os.path.join(combination_patch_dir, origin_file_name)
        revised_path = os.path.join(combination_patch_dir, revised_file_name)
        compare_diff(origin_path, revised_path, combination_patch_dir, revised_file_name)

        logger.info("Original Path : {}".format(origin_path))
        logger.info("Revised Path : {}".format(revised_path))
        
        # global_index += 1
        combination_patch_json = {}
        for file_name, file_patches in file_names.items():
            combination_patch_json[file_name] = {}

            sorted_file_patches = sorted(file_patches, key=lambda x: x.start_line)

            for index, file_patch in enumerate(sorted_file_patches):
                combination_patch_json[file_name][index] = file_patch.__dict__

    write_json(combination_patch_dir, "combination_patch", combination_patch_json)

def get_combine_patches_by_same_index(top_k_patch_sets: List[List[CandidatePatch]]):
   
    mapped_combination_patches: List[List[CandidatePatch]] = []

    row_len = len(top_k_patch_sets)
    col_len = len(top_k_patch_sets[0])

    index = 0
    for _ in range(col_len):
        mapped_combination_patches.append([])

        for j in range(row_len):
            mapped_combination_patches[index].append(top_k_patch_sets[j][index])
        index += 1
    
    return mapped_combination_patches

def get_combination_patches(top_k_patch_sets: List[List[CandidatePatch]]):
    return list(product(*top_k_patch_sets))

def optimize_patches(option: OptimizationOption, is_combined = True, with_calculation=True):
    repair_dir = apr_dir.repair_dir

    module_file = "defects4j_modules"

    modules: dict = read_json(repair_dir, module_file)

    modules = dict(sorted(modules.items(), key=lambda x: (x[1]['module_name'], int(x[1]['module_num']))))

    patches_dir = option.patches_dir

    bins_dir = apr_dir.bins_dir
    logs_dir = apr_dir.logs_dir

    if is_combined and with_calculation:
        logger.info("Starts to filter candidate patches.")
        if print_log:
            ft = filter_candidate_patches(bins_dir, option, logs_dir, logger_name, logger_name, print_log=print_log)
            logger.info("Filtering : \n{}",format(ft))
        else:
            filter_candidate_patches(bins_dir, option, logs_dir, logger_name, logger_name)
        logger.info("Finished to filter candidate patches.")

        logger.info("Starts to rank candidate patches.")
        if print_log:
            rk = rank_candidate_patches(bins_dir, option, logs_dir, logger_name, logger_name, alpha=5, beta=5, print_log=print_log)
            logger.info("Ranking : \n{}",format(rk))
        else:
            rank_candidate_patches(bins_dir, option, logs_dir, logger_name, logger_name, alpha=5, beta=5)
        logger.info("Finished to rank candidate patches.")
    
    if is_combined == False:
        logger.info("Starts to map candidate patches.")
        if print_log:
            mp = map_patches_by_index(bins_dir, option, logs_dir, logger_name, logger_name, print_log=print_log)
            logger.info("Mapping : \n{}",format(mp))
        else:
            map_patches_by_index(bins_dir, option, logs_dir, logger_name, logger_name)
        logger.info("Finished to map candidate patches.")

    if without_optimization:
        return
        
    # combination_patch_max = 20000 # 10000
    # num_candidate_patches = 1000 # 500

    combination_patches_dir = option.combination_patches_dir

    for module_id, module_value in modules.items():

        """ if module_id not in ["Chart_18", "Chart_22", "Math_6", "Mockito_6",  "Closure_85"]:
            continue """
            
        logger.info("Module ID : {}".format(module_id))

        module = Defects4JModule(**module_value)

        multi_chunks: dict = module.multi_chunks

        num_multi_chunks = len(multi_chunks)

        if is_propotional_top_k == True:
            b_loc_totals = []
            for multi_chunk_key, multi_chunk_item in multi_chunks.items():
                total = multi_chunk_item['buggy_line_total']
            
                logger.info("{} : {} lines".format(multi_chunk_key, total))

                b_loc_totals.append(total)

            logger.info("nc : {}, MC : {}".format(num_multi_chunks, combination_patch_max))
            logger.info("buggy location totals : {}".format(b_loc_totals))

            top_ks = get_propotional_top_ks(num_multi_chunks, combination_patch_max, num_candidate_patches, b_loc_totals)
            logger.info("top ks : {}".format(top_ks))

            top_k_max_multiplier = 2
            top_k_ranges = get_propotional_top_k_ranges(num_multi_chunks, combination_patch_max, num_candidate_patches, top_ks, top_k_max_multiplier)
            logger.info("top k ranges : {}".format(top_k_ranges))
            
            top_ks = get_extended_propotional_top_ks(combination_patch_max, top_k_ranges)
            logger.info("extended top ks : {}".format(top_ks))
        else:
            top_ks = get_top_ks(num_multi_chunks, num_candidate_patches, combination_patch_max)
            logger.info("top ks : {}".format(top_ks))

        patch_sets = []

        top_k_index = 0
        for multi_chunk_key, _ in multi_chunks.items():

            module_multi_chunk_dir = os.path.join(patches_dir, module_id, multi_chunk_key)

            try:
                if is_combined:
                    ranked_patch_data: dict = read_json(module_multi_chunk_dir, option.ranked_patches_file.split(".")[0])
                    ranked_patch_data: dict = sorted(ranked_patch_data.items(), key = lambda item: CandidatePatch(**item[1]).score_total, reverse = True)

                    # for top_k in top_ks:
                    current_top_k = top_ks[top_k_index]

                    logger.info("{} : current top k - {}".format(multi_chunk_key, current_top_k))

                    top_k_patches: List[CandidatePatch] = list(map(lambda item: CandidatePatch(**item[1]), ranked_patch_data))[:current_top_k]
                    patch_sets.append(top_k_patches)
                else:
                    patch_data: dict = read_json(module_multi_chunk_dir, option.ranked_patches_file.split(".")[0])

                    candidate_patches: List[CandidatePatch] = list(map(lambda item: CandidatePatch(**item[1]), patch_data.items()))
                    patch_sets.append(candidate_patches)

            except Exception as ex:
                error_logger.error("Module ID : {}".format(module_id))
                error_logger.error(ex, exc_info=True, stack_info=True)
                patch_sets.append([])
            finally:
                top_k_index += 1

        combination_patches = get_combination_patches(patch_sets) if is_combined else get_combine_patches_by_same_index(patch_sets)

        module_clones_dir = os.path.join(grand_parent_dir, "raw_dataset", "defects4j_modules")

        for combination_index, combination_patch in enumerate(combination_patches):
            try:
                candidate_patches = combination_patch
                
                copy_origin_files(module_clones_dir, combination_patches_dir, candidate_patches, module_id, combination_index + 1)
                apply_patches(combination_patches_dir, candidate_patches, module_id, combination_index + 1)
            except Exception as ex:
                error_logger.error("Module ID : {}".format(module_id))
                error_logger.error(ex, exc_info=True, stack_info=True)
                continue

def optimize_patches_without_optimization():
    logger.info("Starts with optimizing patches. (Patch Optimization X)")
    option: OptimizationOption = OptimizationOption()

    option.datasets_type = "defects4j"
    option.json_path = os.path.join(apr_dir.repair_dir, "defects4j_modules.json")

    option.multi_chunk_file = "multi_chunk.txt"

    option.patches_dir = apr_dir.candidate_patches_dir
    option.patches_file = "candidate_patches_buggy_block.txt"

    option.ranked_patches_dir = apr_dir.candidate_patches_dir
    option.ranked_patches_file = "candidate_patches_buggy_block.json"

    option.combination_patches_dir = apr_dir.combination_patches_mapping_dir

    optimize_patches(option, is_combined=False)
    logger.info("Finished to optimize patches. (Patch Optimization X)")

def optimize_patches_without_special_tokens():
    logger.info("Starts with optimizing patches. (Speical Tokens X)")
    option: OptimizationOption = OptimizationOption()

    option.datasets_type = "defects4j"
    option.json_path = os.path.join(apr_dir.repair_dir, "defects4j_modules.json")

    option.multi_chunk_file = "multi_chunk.txt"

    option.patches_dir = apr_dir.candidate_patches_dir + "_without_special_tokens"
    option.patches_file = "candidate_patches_without_special_tokens.txt"

    option.filtered_patches_dir = apr_dir.candidate_patches_dir + "_without_special_tokens"
    option.filtered_patches_file = "filtered_candidate_patches_buggy_block_without_special_tokens.txt"

    option.ranked_patches_dir = apr_dir.candidate_patches_dir + "_without_special_tokens"
    option.ranked_patches_file = "ranked_candidate_patches_buggy_block_without_special_tokens.json"

    option.combination_patches_dir = apr_dir.combination_patches_dir # + "_without_special_tokens"

    optimize_patches(option, with_calculation=with_calculation)
    logger.info("Finished to optimize patches. (Speical Tokens X)")

def optimize_patches_for_multi_chunks():
    logger.info("Starts with optimizing patches. (Multi-Chunks)")
    option: OptimizationOption = OptimizationOption()

    option.datasets_type = "defects4j"
    option.json_path = os.path.join(apr_dir.repair_dir, "defects4j_modules.json")

    option.multi_chunk_file = "multi_chunk.txt"

    option.patches_dir = apr_dir.candidate_patches_dir + "_multi_chunks"
    option.patches_file = "candidate_patches.txt"

    option.filtered_patches_dir = apr_dir.candidate_patches_dir + "_multi_chunks"
    option.filtered_patches_file = "filtered_candidate_patches.txt"

    option.ranked_patches_dir = apr_dir.candidate_patches_dir + "_multi_chunks"
    option.ranked_patches_file = "ranked_candidate_patches.json"

    option.combination_patches_dir = apr_dir.combination_patches_dir # + "_multi_chunks"

    optimize_patches(option, with_calculation=with_calculation)
    logger.info("Finished to optimize patches. (Multi-Chunks)")

def optimize_patches_for_buggy_blocks():
    logger.info("Starts with optimizing patches. (Buggy Blocks)")

    option: OptimizationOption = OptimizationOption()

    option.datasets_type = "defects4j"
    option.json_path = os.path.join(apr_dir.repair_dir, "defects4j_modules.json")

    option.multi_chunk_file = "multi_chunk.txt"

    option.patches_dir = apr_dir.candidate_patches_dir
    option.patches_file = "candidate_patches_buggy_block.txt"

    option.filtered_patches_dir = apr_dir.candidate_patches_dir
    option.filtered_patches_file = "filtered_candidate_patches_buggy_block.txt"

    option.ranked_patches_dir = apr_dir.candidate_patches_dir
    option.ranked_patches_file = "ranked_candidate_patches_buggy_block.json"
    
    option.combination_patches_dir = apr_dir.combination_patches_dir

    optimize_patches(option, with_calculation=with_calculation)

    logger.info("Finished to optimize patches. (Buggy Blocks)")

if __name__ == "__main__":
    main()
