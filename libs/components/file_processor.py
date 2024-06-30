import json
import os

def make_dirs(dir):
    # if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

# Json
def read_json(json_dir, json_file, is_checked = True):
    json_file_path = os.path.join(json_dir, json_file + ".json")

    if is_checked:
        make_dirs(json_dir)

    with open(json_file_path, 'r', encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    
    return json_data

def write_json(json_dir, json_file, json_data, indent=4, short_keys=True):
    json_file_path = os.path.join(json_dir, json_file + ".json")

    with open(json_file_path, 'w', newline='', encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=indent, sort_keys=short_keys)

# Txt for Learning
def write_train_txt(file_dir, file_name, src_tokens, tgt_tokens):
    src_token_len = len(src_tokens)
    tgt_token_len = len(tgt_tokens)

    src_file_name = "src-{}.txt".format(file_name)
    tgt_file_name = "tgt-{}.txt".format(file_name)

    assert src_token_len == tgt_token_len
    
    make_dirs(file_dir)
    
    src_count = 0
    tgt_count = 0
    with open(os.path.join(file_dir, src_file_name), "w", encoding="utf-8") as src_file, open(os.path.join(file_dir, tgt_file_name), "w", encoding="utf-8") as tgt_file:
        for index in range(src_token_len):
            try:
                src_code = src_tokens[index]
                tgt_code = tgt_tokens[index]

                src_file.write(src_code + "\n")
                tgt_file.write(tgt_code + "\n")

                src_count += 1
                tgt_count += 1
            except Exception as ex:
                print("index : {}".format(index))
                print("exception : {}".format(ex))
                # src_countcount += 1
                continue

    assert src_token_len == tgt_token_len

    return (src_file_name, tgt_file_name, src_token_len, tgt_token_len, src_count, tgt_count)


def write_test_txt(file_dir, file_name, tokens):
    token_len = len(tokens)

    src_file_name = "{}.txt".format(file_name)

    make_dirs(file_dir)

    with open(os.path.join(file_dir, src_file_name), "w", encoding="utf-8") as src_file:
        src_file.write(" ".join(tokens) + "\n")

    return (src_file_name, token_len) 

def read_txt(file_dir, file_name):
    with open(os.path.join(file_dir, file_name + ".txt"), 'r', encoding="utf-8") as file:
        lines = file.readlines()

    return lines

def write_txt(file_dir, file_name, lines):
    make_dirs(file_dir)

    with open(os.path.join(file_dir, file_name + ".txt"), "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

# Text
def read_test_txt(file_dir, file_name):
    txt_file_path = os.path.join(file_dir, "{}.txt".format(file_name))

    with (open(txt_file_path, 'r', encoding="utf-8")) as txt_file:
        txt_lines = txt_file.readlines()

    return txt_lines

def write_txt(file_dir, file_name, items):
    item_len = len(items)

    src_file_name = "{}.txt".format(file_name)
    
    make_dirs(file_dir)

    with open(os.path.join(file_dir, src_file_name), "w", encoding="utf-8") as src_file:
        for item in items:
            src_file.write(item + "\n")
    
    return (src_file_name, item_len)

# Java
def read_java(file_dir, file_name):
    java_file_path = os.path.join(file_dir, "{}.java".format(file_name))

    with (open(java_file_path, 'r', encoding="utf-8")) as txt_file:
        txt_lines = txt_file.readlines()

    return txt_lines

def write_java(file_dir, file_name, items):
    item_len = len(items)

    src_file_name = "{}.java".format(file_name)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(os.path.join(file_dir, src_file_name), "w", encoding="utf-8") as src_file:
        for item in items:
            src_file.write(item)
    
    return (src_file_name, item_len)

# Developer Patch
def read_develop_patch(file_dir, file_name):
    with open(os.path.join(file_dir, file_name), "r", newline="\n", encoding='latin-1') as patch_file:
        lines = patch_file.readlines()
    
    return lines

def write_develper_patch_lines(file_dir, file_name, items):
    with open(os.path.join(file_dir, file_name), "w", encoding='latin-1') as src_file:
        for item in items:
            src_file.write(item + "\n")
    
    return (file_name, len(items))