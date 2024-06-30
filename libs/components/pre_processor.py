import json
from logging import getLogger
from typing import List
import javalang

logger = getLogger("Step3_Preprocess_BuggyBlock.pre")

def get_jaccard_similarity(list1, list2):
    s1, s2 = (set(list1), set(list2))

    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def get_sorted_field_contexts(field_key: str, context_keys: List[str]):
    # field_name = field_key.split("_")[1]

    getter1, getter2, setter = ("_get", "_is", "_set")

    # capitalized_field = field_name.capitalize()

    # getter_field1, getter_field2, setter_field  = (getter1 + capitalized_field, getter2 + capitalized_field, setter + capitalized_field)
    
    context_items = []

    for context_key in context_keys:
        context_item = (context_key, -3)

        if "constructor" in context_key:
            context_item = (context_key, 0)
        # elif field_name.lower() in context_key.lower():
        elif setter in context_key:
            context_item = (context_key, -1)
        elif getter1 in context_key or getter2 in context_key:
            context_item = (context_key, -2)
        # else:
        #    context_item = (context_key, -3)

        context_items.append(context_item)
    
    logger.info("context_items : {}".format(context_items))
    # context_items = sorted(context_items, key=lambda item: item[1])
    return list(map(lambda x: x[0], sorted(context_items, key=lambda item: item[1], reverse=True)))

def find_buggy_field_contexts(multi_buggy_chunk_key: str, buggy_contexts: dict, multi_chunks: dict) -> dict:
    logger.info("multi_buggy_chunk_key (field) : \n{}".format(multi_buggy_chunk_key))

    contexts: dict = buggy_contexts.get(multi_buggy_chunk_key)

    if contexts is not None:
        context_keys = get_sorted_field_contexts(multi_buggy_chunk_key, contexts.keys())
        logger.info("context_keys : {}".format(context_keys))

        for context_key in context_keys:        
            similar_context = multi_chunks.get(context_key)
            
            if similar_context is not None:
                logger.info("similar_context (field): \n{}".format(json.dumps(similar_context, sort_keys=False, indent=4)))
                return similar_context
    
    logger.info("similar_context (field) : \nNone")
    return {}

def find_buggy_method_contexts(multi_buggy_chunk_key: str, buggy_contexts: dict, multi_chunks: dict) -> dict:
    logger.info("multi_buggy_chunk_key (method) : \n{}".format(multi_buggy_chunk_key))

    fields_dict = {}

    for field_key, contexts in buggy_contexts.items():
        context_keys = contexts.keys()

        for context_key in context_keys:
            if context_key not in fields_dict:
                fields_dict[context_key] = [field_key]
            else:
                fields_item = fields_dict[context_key]

                if field_key not in fields_item:
                    fields_dict[context_key].append(field_key)
    # logger.info("fields_dict : \n{}".format(json.dumps(fields_dict, sort_keys=False, indent=4)))

    multi_chunk_context_fields = fields_dict.get(multi_buggy_chunk_key)

    # logger.info("multi_chunk_context_fields : \n{}".format(json.dumps(multi_chunk_context_fields, sort_keys=False, indent=4)))

    max_jaccard_similarity = 0.0
    similar_context = {}

    if multi_chunk_context_fields is None:
        return similar_context

    for context_key, context_fields in fields_dict.items():
        if context_key != multi_buggy_chunk_key:
            jaccard_similarity = get_jaccard_similarity(multi_chunk_context_fields, context_fields)

            if jaccard_similarity > max_jaccard_similarity:
                # logger.info("similar_context (method) : \n{}".format(json.dumps(similar_context, sort_keys=False, indent=4)))   
                similar_context = multi_chunks[context_key]

    logger.info("similar_context (method) : \n{}".format(json.dumps(similar_context, sort_keys=False, indent=4)))     
    return similar_context

def find_lines(lines, start_line, end_line):
    new_lines = []

    for line in range(start_line, end_line + 1):
        line = lines[str(line)]
        new_lines.append(line)
    
    return new_lines

def find_buggy_diffs(diffs: dict, start_line: int, end_line: int, is_buggy=True):
    new_diffs = {}
    
    for diff_key, diff_item in diffs.items():
        diff_start_line = int(diff_key)
        
        diff_line_condition = diff_start_line >= start_line and diff_start_line <= end_line
        
        if diff_line_condition:
            diff_end_line: int = diff_item['buggy_end_line'] if is_buggy else diff_item['fixed_end_line']
            for diff_line in range(diff_start_line, diff_end_line + 1):
                new_diffs[diff_line] = diff_item
    
    return new_diffs

def find_buggy_lines(buggy_lines: dict, start_line: int, end_line: int):
    new_buggy_lines = {}
    
    for buggy_line_key, buggy_line_item in buggy_lines.items():
        buggy_line_num = int(buggy_line_key)
        
        buggy_line_condition = buggy_line_num >= start_line and buggy_line_num <= end_line
        
        if buggy_line_condition:
            new_buggy_lines[buggy_line_num] = buggy_line_item
    
    return new_buggy_lines
