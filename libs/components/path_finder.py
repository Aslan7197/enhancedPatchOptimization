import glob
from logging import Logger
import os
import shutil
import subprocess
from typing import Any, Iterator, List

from libs.data.fault_path import FaultPath
from libs.data.manysstubs4j import ManySStuBs4J, ManySStuBs4JDiff

buggy_dir, fixed_dir = ("F_dir", "P_dir")
file_ext = ".java"
buggy_ext = ".buggy.lines"

def get_chunk_total(lines: dict):
    # print(lines)
    line_len = len(lines.keys())
    
    if line_len <= 1:
        return line_len
    
    total = 0

    omitted_lines = dict(filter(lambda x: x[1]['is_omission'] == True, lines.items()))

    total += len(omitted_lines.keys())

    other_lines = dict(filter(lambda x: x[1]['is_omission'] == False, lines.items()))
    other_line_len = len(other_lines.keys())

    if (other_line_len <= 1):
        total += other_line_len
    else:
        other_lines = sorted(map(int, other_lines.keys()))
        
        prev = other_lines[0]
        # total += 1
        
        for line in other_lines[1:]:
            if abs(line - prev) >= 2:
                total += 1
                prev = line
    
    return total

def find_bugs2fix_fault_paths(faults_dir: str):
    fault_paths = {}

    recursive_fault_dir = os.path.join(faults_dir, '**/**')
    for file_path in glob.iglob(recursive_fault_dir, recursive=True):
        if (buggy_dir in file_path) and (file_ext in file_path):
            new_file_path = file_path.replace(faults_dir + "/", "")
            new_file_paths = new_file_path.split("/")

            module_id = new_file_paths[0]
            file_name = new_file_paths[-1].replace(file_ext, "")

            if "Test" not in file_name:
                module_paths = new_file_paths[2:-1]
                module_dir = "/".join(module_paths)

                fixed_path = os.path.join(faults_dir, module_id, fixed_dir, module_dir, file_name + file_ext)
                # print(fixed_path)
                if os.path.isfile(fixed_path) == True:
                    file_key = "{}_{}_{}".format(module_id, "_".join(module_paths), file_name)

                    fault_path = FaultPath(
                        module_id=module_id, module_dir=module_dir,
                        file_name=file_name, file_ext=file_ext,
                        buggy_dir=buggy_dir, fixed_dir=fixed_dir,
                        # file_path=file_path
                    )
                    fault_paths[file_key] = fault_path.__dict__

                    """ fault_paths[file_key] = {
                        'module_id': module_id,
                        'module_dir': module_dir,
                        'buggy_dir': buggy_dir,
                        'fixed_dir': fixed_dir,
                        'file_name': file_name,
                        'file_ext': file_ext
                    } """

    return fault_paths

def find_bugs2fix_sub_fault_paths(faults_dir: str, fault_data: Iterator[str], start: int, end: int, fault_paths: List[tuple]):
    # fault_paths = {}

    for file_path in fault_data[start:end]:
        if (buggy_dir in file_path) and (file_ext in file_path):
            new_file_path = file_path.replace(faults_dir + "/", "")
            new_file_paths = new_file_path.split("/")

            module_id = new_file_paths[0]
            file_name = new_file_paths[-1].replace(file_ext, "")

            if "Test" not in file_name:
                module_paths = new_file_paths[2:-1]
                module_dir = "/".join(module_paths)

                fixed_path = os.path.join(faults_dir, module_id, fixed_dir, module_dir, file_name + file_ext)
                # print(fixed_path)
                if os.path.isfile(fixed_path) == True:
                    file_key = "{}_{}_{}".format(module_id, "_".join(module_paths), file_name)

                    fault_path = FaultPath(
                        module_id=module_id, module_dir=module_dir,
                        file_name=file_name, file_ext=file_ext,
                        buggy_dir=buggy_dir, fixed_dir=fixed_dir,
                        # file_path=file_path
                    )
                    # fault_paths[file_key] = fault_path.__dict__
                    fault_paths.append((file_key, fault_path.__dict__))

                    """ fault_paths[file_key] = {
                        'module_id': module_id,
                        'module_dir': module_dir,
                        'buggy_dir': buggy_dir,
                        'fixed_dir': fixed_dir,
                        'file_name': file_name,
                        'file_ext': file_ext
                    } """

    return fault_paths

def find_bugs2fix_fault_paths2(dataset: str, bins_dir: str, extracted_dir: str, stored_dir: str, logs_dir: str, log_file: str, logger_name: str):
    # faults_json_path = os.path.join(localization_dir, fault_paths_file)

    className = "uos.selab.paths.PathExtracter"
    jarPath = os.path.join(bins_dir, "custom_java_parser.jar")

    cmd = ['java', '-cp', jarPath, className, dataset, extracted_dir, stored_dir, logs_dir, log_file, logger_name]

    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.wait()

def find_defects4j_fault_paths(faults_dir: str):
    # print(faults_dir)
    json_data={}

    # logger = getLogger("Localize.defects4j")
    # logger.info("Loclize Defects4J")
    # logger.info("Faults Directory : {}".format(faults_dir))

    for origin_faults_file_path in glob.iglob(faults_dir + '/**/**', recursive=True):
        """ buggy_ext = ".buggy.lines"
        file_ext = ".java" """

        if buggy_ext in origin_faults_file_path:
            faults_file_path = origin_faults_file_path.replace(faults_dir + "/", "")
            faults_file_paths = faults_file_path.replace(buggy_ext, "").split("-")
            print(faults_dir)
            print(faults_file_path)
            module_name = faults_file_paths[0]
            module_num = faults_file_paths[1]
            module_number_len = len(module_num)

            # logger.info("File Path : {}".format(faults_file_path))

            if module_number_len >= 1 and module_number_len <= 3:

                module_id = module_name + "_" + module_num

                """ module_buggy_dirs = {
                    'Chart': 'source',
                    'Closure': 'src',
                    'Lang35': 'src/main/java',
                    'Lang36': 'src/java',
                    'Math84': 'src/main/java',
                    'Math85': 'src/java',
                    'Mockito': 'src',
                    'Time': 'src/main/java'
                } """
                
                with open(origin_faults_file_path, 'r', encoding='latin-1') as loc_file:
                    file_lines = loc_file.readlines()
                    print(module_id)
                    for file_line in file_lines:
                        if len(file_line) <= 0 or file_line is None:
                            continue

                        # print(file_line)
                        lines = file_line.split("#")
                        line_len = len(lines)

                        file_path = lines[0]
                        # print(lines)
                        buggy_line_nums = lines[1].split("-")

                        buggy_line_num = int(buggy_line_nums[0])
                        buggy_start_line_num = buggy_line_num
                        buggy_end_line_num = int(buggy_line_nums[1]) if len(buggy_line_nums) > 1 else buggy_line_num
                        
                        body = lines[2].lstrip().rstrip().strip('\n')

                        file_path = file_path.replace(file_ext, "")
                        file_paths = file_path.split("/")

                        module_dir = "/".join(file_paths[:-1])
                        
                        file_name = file_paths[-1]
                        file_key = "{}_{}_{}".format(module_id, "_".join(file_paths[:-1]), file_name)
                        print(file_key)
                        
                        if file_key not in json_data:

                            json_data[file_key] =  {
                                'module_id': module_id,
                                'module_name': module_name,
                                'module_num': module_num,
                                'module_dir': module_dir,
                                'buggy_dir': "",
                                'file_key': file_key,
                                'file_name': file_name,
                                'file_ext': file_ext,
                                'chunk_total': 0,
                                'buggy_lines': {},
                                'buggy_line_total': 0
                            }

                        buggy_lines: dict = json_data[file_key]['buggy_lines']

                        if buggy_line_num not in buggy_lines:
                            buggy_line_info = {
                                'buggy_start_line': buggy_start_line_num,
                                'buggy_end_line': buggy_end_line_num,
                                'is_duplicated': buggy_start_line_num != buggy_end_line_num,
                                'replaced_start_line': None,
                                'replaced_end_line': None,
                                'is_reduplicated': False,
                                'body': body,
                                'is_omission': body == 'FAULT_OF_OMISSION',
                                'replaced_body': None,
                                'is_empty': False
                            }

                            if line_len >= 5:
                                replaced_line_nums = lines[3].split("-")

                                replaced_line_num = int(replaced_line_nums[0])
                                replaced_start_line_num = replaced_line_num
                                replaced_end_line_num = int(replaced_line_nums[1]) if len(replaced_line_nums) > 1 else replaced_start_line_num

                                replaced_body = lines[4].lstrip().rstrip().strip('\n')

                                buggy_line_info.update({
                                    'replaced_start_line': replaced_start_line_num,
                                    'replaced_end_line': replaced_end_line_num,
                                    'is_reduplicated': replaced_start_line_num != replaced_end_line_num,
                                    'replaced_body': replaced_body
                                })
                            
                                if line_len == 6:
                                    buggy_line_info.update({
                                        'is_empty': True
                                    })

                            buggy_lines.update({
                                buggy_line_num: buggy_line_info
                            })
                            
                        json_data[file_key]['chunk_total'] = get_chunk_total(buggy_lines)
                        json_data[file_key]['buggy_line_total'] = len(buggy_lines.keys())


    return json_data

def __move_commit(commit_id: str, timeout=120):        
    is_timeout=False
    
    cmd = ["git", "checkout", "-f", commit_id]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        is_timeout = True
    
    if is_timeout or output is None:
        return False
    
    out_res = output.decode('utf-8')

    return "error:" not in out_res

def __is_copied(src_path: str, tgt_dir: str, tgt_path: str):
    try:
        # if not os.path.exists(tgt_path):
        os.makedirs(tgt_dir, exist_ok=True)
        shutil.copyfile(src_path, tgt_path)
        return True
    except Exception as ex:
        print("ex : ", ex)
        return False

def find_manysstubs4j_faults(fault_data: List[Any], fault_dir: str, stored_dir: str, limitedProjects: dict, logger: Logger = None):
    fault_paths = {}
    fault_diffs = {}

    # fault_data = get_fault_data(os.path.join(fault_dir, 'sstubsSmall.json'))

    for fault in fault_data:
        try:
            many4J = ManySStuBs4J(**fault)

            project_name = many4J.projectName

            if limitedProjects.get(project_name) == None:
                continue

            module_id = many4J.fixCommitSHA1
            module_parent_id = many4J.fixCommitParentSHA1

            file_path = many4J.bugFilePath
            new_module_paths = file_path.replace(file_ext, "").split("/")

            module_dir_paths =  new_module_paths[:-1]

            file_name = new_module_paths[-1]

            module_dir = "/".join(module_dir_paths)

            project_path = os.path.join(fault_dir, project_name)
            os.chdir(project_path)

            is_buggy_moved = __move_commit(module_parent_id)

            if is_buggy_moved == False:
                continue

            buggy_src_path = os.path.join(project_path, file_path)

            buggy_tgt_dir = os.path.join(stored_dir, module_id, buggy_dir, module_dir)
            buggy_tgt_path = os.path.join(buggy_tgt_dir, file_name + file_ext)
            is_buggy_copied = __is_copied(buggy_src_path, buggy_tgt_dir, buggy_tgt_path)

            if is_buggy_copied == False:
                continue

            is_fixed_moved = __move_commit(module_id)

            if is_fixed_moved == False:
                continue

            fixed_src_path = os.path.join(project_path, file_path)

            fixed_tgt_dir = os.path.join(stored_dir, module_id, fixed_dir, module_dir)
            fixed_tgt_path = os.path.join(fixed_tgt_dir, file_name + file_ext)
            is_fixed_copied = __is_copied(fixed_src_path, fixed_tgt_dir, fixed_tgt_path)

            if is_fixed_copied == False:
                continue
            
            buggy_key = "_".join([module_id] + new_module_paths)

            bugLineNum = many4J.bugLineNum
            fixLineNum = many4J.fixLineNum
            fault_diff = ManySStuBs4JDiff(
                action="CHANGE",
                buggy_start_line=bugLineNum, buggy_end_line=bugLineNum, buggy_size=1, 
                fixed_start_line=fixLineNum, fixed_end_line=fixLineNum, fixed_size=1
            )

            if buggy_key not in fault_diffs:
                fault_diffs[buggy_key] = {}

            fault_diffs[buggy_key].update({
                str(bugLineNum): fault_diff.__dict__
            })

            file_path = os.path.join(module_id, buggy_dir, module_dir)
            fault_path = FaultPath(
                module_id=module_id, module_dir=module_dir,
                file_name=file_name, file_ext=file_ext,
                buggy_dir=buggy_dir, fixed_dir=fixed_dir,
                file_path=file_path
            )

            fault_paths[buggy_key] = fault_path.__dict__

            if logger is not None:
                logger.info("module id : {}".format(module_id))
        except Exception as ex:
            if logger is not None:
                logger.error("module id : ".format(module_id))
                logger.error(ex, exc_info=True, stack_info=True)
            continue

        # print(many4J.projectUrl)
    
    return fault_paths, fault_diffs