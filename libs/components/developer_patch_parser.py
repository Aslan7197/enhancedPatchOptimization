from typing import List

# diff --git : The start of a diff file
# @@ : The start of a diff 
# + : Buggy Line, - : Fixed Line
def get_diff_file_starts(lines: List[str]):
    starts = []

    for index, line in enumerate(lines):
        if "diff" in line and "--git" in line:
            starts.append(index)

    return sorted(starts)

def get_diff_file_ends(starts: List[int], line_len: int):
    start_len = len(starts)

    if start_len == 1:
       return [line_len - 1]

    ends = []

    for index, _ in enumerate(starts):
        if index == 0:
            continue
        
        ends.append(starts[index] - 1)

    ends.append(line_len - 1)

    return sorted(ends)

def get_diff_file_path(line: str):
    tokens = line.split(" ")

    return tokens[2][2:]

def get_diff_starts_per_file(lines: List[str], start: int, end: int):
    starts = []

    for index, line in enumerate(lines):
        if index >= start and index <= end and line.startswith("@@"):
            starts.append(index + 1)
    
    return starts

def get_diff_ends_per_file(starts: List[int], end: int):
    start_len = len(starts)

    if start_len == 1:
       return [end]

    ends = []

    for index, _ in enumerate(starts):
        if index == 0:
            continue
        
        ends.append(starts[index] - 1 - 1)

    ends.append(end)

    return sorted(ends)

def get_start_buggy_line_per_diff(line: str):
    if (line == None or len(line) <= 0):
        raise Exception("Line is empty.")
    
    tokens = line.split(" ")

    start_line = None

    for token in tokens:
        if token.startswith("+"):
            sub_tokens = token.strip().replace("+", "").split(",")

            start_line = int(sub_tokens[0])

    if start_line == None:
        raise Exception("Start Line is empty.")
    
    return start_line

def is_buggy_line(line: str):
    return line.startswith("+")

def is_fixed_line(line: str):
    return line.startswith("-")

""" def is_comment(line: str):
    new_line = line.strip()

    return new_line.startswith("//") or new_line.startswith("/*") or new_line.startswith("*") or new_line.startswith("*/") or (not new_line.startswith("*") and new_line.endswith("*/"))

def is_blank_or_empty(line: str):
    new_line = line.strip()

    return new_line == "" or len(new_line) == 0

def is_fixed_block_end(line: str):
    return line[1:].strip() == "}" """

def get_symbol(line: str):
    return "+" if is_buggy_line(line) else "-" if is_fixed_line(line) else ""

def get_adjusted_lines(lines: List[str], start_line: int):
    line_len = len(lines)

    if (line_len == 0):
        raise Exception("Lines are empty.")

    if (line_len == 1):
        return [start_line]
    
    new_lines = [start_line]

    prev_symbol = get_symbol(lines[0])
    prev_line = start_line

    for index, line in enumerate(lines):
        if index == 0:
            continue
        
        current_symbol = get_symbol(line)

        if (prev_symbol == "-"):
            new_lines.append(prev_line)
        else:
            new_lines.append(prev_line + 1)
            prev_line += 1
        
        prev_symbol = current_symbol
    
    return new_lines

def get_buggy_chunks(lines: List[str]):
    line_len = len(lines)

    if (line_len == 0):
        raise Exception("Lines are empty.")
    
    starts = []
    ends = []

    prev_symbol =  get_symbol(lines[0])
    # prev_line = start_line

    for index, line in enumerate(lines):
        current_symbol = get_symbol(line)

        if index == 0:
            continue

        if (prev_symbol == "" and (current_symbol == "-" or current_symbol == "+")):
            starts.append(index)
        
        if (prev_symbol == "+" and current_symbol == "-"):
            ends.append(index - 1)
            starts.append(index)
        
        if ((prev_symbol == "+" or prev_symbol == "-") and current_symbol == ""):
            ends.append(index - 1)
        
        prev_symbol = current_symbol

    assert (len(starts) == len(ends))

    return (starts, ends)

def get_info_lines_per_chunk(file_path: str, sub_lines: List[str], adjusted_lines: List[str]):
    new_info_lines = []

    line_len, buggy_line_len, fixed_line_len = (len(sub_lines), 0, 0)

    for sub_line in sub_lines:
        if is_buggy_line(sub_line):
            buggy_line_len += 1
        
        if is_fixed_line(sub_line):
            fixed_line_len += 1

    # is_all_buggy = line_len == buggy_line_len
    is_all_fixed = line_len == fixed_line_len

    if is_all_fixed:
        new_info_lines.append("{}#{}#{}".format(file_path, adjusted_lines[0], "FAULT_OF_OMISSION"))
    else:
        for sub_line, adjusted_line in zip(sub_lines, adjusted_lines):
            if is_buggy_line(sub_line):
                new_info_lines.append("{}#{}#{}".format(file_path, adjusted_line, sub_line.replace("+", "").replace("\n", "").strip()))


    return new_info_lines

def parse_developer_patch(lines: List[str]):
    # lines = get_txt_lines()

    line_len = len(lines)

    file_starts = get_diff_file_starts(lines)
    file_ends = get_diff_file_ends(file_starts, line_len)

    if (len(file_starts)) != len(file_ends):
        raise Exception("File indexes for diffs are different.")

    """ print(file_starts)
    print(file_ends)
    print() """
    info_lines = []

    for start, end in zip(file_starts, file_ends):
        diff_starts = get_diff_starts_per_file(lines, start, end)

        # print(diff_starts)

        diff_ends = get_diff_ends_per_file(diff_starts, end)
        # print(diff_ends)

        file_path = get_diff_file_path(lines[start])
        # print(file_path)


        index = 0
        for diff_start, diff_end in zip(diff_starts, diff_ends):
            # buggy_lines = get_buggy_lines_per_diff(lines, diff_start, diff_end)
            """ print(buggy_lines)
            print() """

            # fixed_lines = get_fixed_lines_per_diff(lines, diff_start, diff_end)
            """ print(fixed_lines)
            print()

            print("diff {}".format(index)) """
            start_buggy_line = get_start_buggy_line_per_diff(lines[diff_start - 1])
            """ print(start_buggy_line)
            print() """

            sub_lines = lines[diff_start: diff_end + 1]

            adjusted_lines = get_adjusted_lines(sub_lines, start_buggy_line)

            chunk_starts, chunk_ends  = get_buggy_chunks(sub_lines)

            """ print(adjusted_lines)
            print()

            print(chunk_starts)
            print(chunk_ends)
            print() """

            for chunk_start, chunk_end in zip(chunk_starts, chunk_ends):
                """ if is_comment(sub_lines[chunk_end + 1]):
                    continue """

                new_sub_lines = sub_lines[chunk_start: chunk_end + 1]

                """ if len(new_sub_lines) == 1:
                    first_line = new_sub_lines[0]
                    
                    if is_fixed_line(first_line) and is_fixed_block_end(first_line):
                        continue """

                new_adjusted_lines = adjusted_lines[chunk_start: chunk_end + 1]

                new_info_lines = get_info_lines_per_chunk(file_path, new_sub_lines, new_adjusted_lines)
                info_lines.extend(new_info_lines)
            
        
            index += 1
        
    return info_lines

# Chart_14, Chart_26, Closure_103, Closure_110, Math_66, Lang_7, Mockito_17, Time_17, 검증 => additional validation
