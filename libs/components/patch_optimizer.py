from functools import reduce
from itertools import product
import math
import os
import subprocess

from libs.data.patch import OptimizationOption

def get_top_ks(nc: int, num_candidate_patches: int, combination_patch_max: int):
    ks = []
    
    for patch_index in range(1, num_candidate_patches + 1):
        if (patch_index ** nc) <= combination_patch_max:
            ks.append(patch_index)
        else:
            break
    
    top_k = max(ks)

    return [top_k for _ in range(nc)]

def filter_candidate_patches(bins_dir: str, option: OptimizationOption, logs_dir: str, log_file: str, logger_name: str, print_log=False):
    className = "uos.selab.patches.PatchFiltering"
    jarPath = os.path.join(bins_dir, "custom_java_parser.jar")

    cmd = ['java', '-cp', jarPath, className, option.datasets_type, option.json_path, 
        option.multi_chunk_file, option.patches_dir, option.patches_file, option.filtered_patches_dir, option.filtered_patches_file,
        logs_dir, log_file, logger_name]

    if print_log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        return process.communicate()
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.wait()
        return (None, None)

def rank_candidate_patches(bins_dir: str, option: OptimizationOption, logs_dir: str, log_file: str, logger_name: str, print_log=False, alpha=5, beta=5):
    className = "uos.selab.patches.PatchRanking"
    jarPath = os.path.join(bins_dir, "custom_java_parser.jar")

    total = alpha + beta
    print(total)
    if total < 0 or total > 10:
        raise Exception("The sum of alpha and beta is between 0 and 10.")

    cmd = ['java', '-cp', jarPath, className, option.datasets_type, option.json_path, 
        option.multi_chunk_file, option.filtered_patches_dir, option.filtered_patches_file, option.ranked_patches_dir, option.ranked_patches_file,
        logs_dir, log_file, logger_name, str(alpha), str(beta)]

    if print_log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        return process.communicate()
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.wait()
        return (None, None)

def map_patches_by_index(bins_dir: str, option: OptimizationOption, logs_dir: str, log_file: str, logger_name: str, print_log=False):
    className = "uos.selab.patches.PatchMapping"
    jarPath = os.path.join(bins_dir, "custom_java_parser.jar")

    cmd = ['java', '-cp', jarPath, className, option.datasets_type, option.json_path, 
        option.multi_chunk_file, option.patches_dir, option.patches_file, option.ranked_patches_dir, option.ranked_patches_file,
        logs_dir, log_file, logger_name]

    if print_log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        return process.communicate()
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.wait()
        return (None, None)

def compare_diff(origin_path: str, revised_path: str, stored_dir: str, revised_file_name: str):
    cmd = ["diff", "-u", "-w", "{}.java".format(origin_path), "{}.java".format(revised_path)]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, _ = process.communicate()

    result = output.decode('utf-8')

    diff_path = os.path.join(stored_dir, "{}_diff".format(revised_file_name))
    with open(diff_path, 'w') as diff:
        diff.write(result)


##
def get_propotional_top_ks(nc, MC, SP, b_locs):
  top_ks = []

  for i, line1 in enumerate(b_locs):
    denominator = 1
  
    for j, line2 in enumerate(b_locs):
        if (j != i):
            denominator *= line2
    
    propotional_MC = (MC * (line1 ** (nc - 1))) / denominator
    propotional_MC = int(propotional_MC)

    top_k = 1

    for k in range(1, SP + 1):
        if (k ** nc) <= propotional_MC:
            top_k = k
        else:
            break

    top_ks.append(top_k)
  
  return top_ks

def get_propotional_top_k_ranges(nc, MC, SP, top_ks, top_k_max_multiplier):
  top_k_ranges = []

  for idx, _ in enumerate(top_ks):
    top_k = top_ks[idx]

    remainder_top_ks = top_ks[:idx] + top_ks[idx+1:]
    remainder_product = reduce(lambda acc, value: acc * value, remainder_top_ks, 1)

    # print(product_val)

    max_top_k = top_k

    limit1 = math.ceil(SP * (top_k / sum(top_ks)))
    for current_top_k in range(top_k, limit1 + 1):
        # current_val = (max_top_k * remainder_product)
        if (max_top_k * remainder_product) <= MC:
            max_top_k = current_top_k
        else:
            break

    limit2 = top_k * top_k_max_multiplier
    max_top_k = min(max_top_k, limit2)
    print(idx, ":", max_top_k)
    top_k_ranges.append([i for i in range(top_k, max_top_k + 1)])

  return top_k_ranges


def get_extended_propotional_top_ks(MC, top_k_ranges):
    combs1 = product(*top_k_ranges)
    combs2 = filter(lambda x: reduce(lambda acc, value: acc * value, x, 1) <= MC, combs1)
    combs3 = sorted(combs2, key=lambda x: (reduce(lambda acc, value: acc * value, x, 1), -math.sqrt(__get_variance(x)), sum(x)), reverse=True)
    
    return iter(combs3).__next__()

def __get_variance(vals):
    vsum = 0
    mean = sum(vals) / len(vals)

    for val in vals:
        vsum = vsum + (val - mean) ** 2

    return vsum / len(vals)

