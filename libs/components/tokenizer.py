from typing import Any, List
import javalang

def tokenize_buggy_lines_by_diffs(buggy_diffs: dict, lines: List[Any], add_special_tokens = True):
    new_tokens = []

    for line in lines:
        line_num = int(line['line_number'])

        is_all_comment = bool(line['is_all_comment'])
        is_blank = bool(line['is_blank'])

        if is_all_comment or is_blank:
            body_tokens = []
        else:
            body = str(line['body'])
            body_tokens = list(javalang.tokenizer.tokenize(body))
            body_tokens = list(map(lambda x: str(x.value), body_tokens))

        buggy_diff = buggy_diffs.get(line_num)
        buggy_line_condition = buggy_diff is not None

        if buggy_line_condition:
            if add_special_tokens:
                action = buggy_diff['action']

                insert_action = action == "INSERT"

                if insert_action:
                    omit_token1 = "<omit>"
                    omit_token2 = "</omit>"
                    body_tokens = [omit_token1] + body_tokens + [omit_token2]
                else:
                    bug_token1 = "<bug>"
                    bug_token2 = "</bug>"
                    body_tokens = [bug_token1] + body_tokens + [bug_token2]
    
        new_tokens.extend(body_tokens)

    return new_tokens

# Buggy Lines를 기준으로 재작성
def tokenize_buggy_lines_by_information_lines(buggy_lines: dict, information_lines: List[Any], add_special_tokens = True, is_strengthed=True):
    new_tokens = []

    omit_token1, omit_token2 = ("<omit>", "</omit>")
    bug_token1, bug_token2 = ("<bug>", "</bug>")

    if is_strengthed == True:
        duplicated_line_nums = []

        for line in information_lines:
            line_num = int(line['line_number'])

            if bool(line['is_all_comment']) or bool(line['is_blank']):
                continue
            
            if line_num in duplicated_line_nums:
                continue

            body = str(line['body'])
            body_tokens = list(javalang.tokenizer.tokenize(body))
            body_tokens = list(map(lambda x: str(x.value), body_tokens))

            buggy_line = buggy_lines.get(line_num)

            if buggy_line is not None:
                if add_special_tokens:
                    if buggy_line['is_omission'] == True:
                        ins_body = str(buggy_line['replaced_body'])
                        ins_body_tokens = list(javalang.tokenizer.tokenize(ins_body))
                        ins_body_tokens = list(map(lambda x: str(x.value), ins_body_tokens))
                        body_tokens = [omit_token1] + ins_body_tokens + [omit_token2]
                    else:
                        other_body = str(buggy_line['body'])
                        other_body_tokens = list(javalang.tokenizer.tokenize(other_body))
                        other_body_tokens = list(map(lambda x: str(x.value), other_body_tokens))
                        body_tokens = [bug_token1] + other_body_tokens + [bug_token2]

                buggy_start_line = int(buggy_line['buggy_start_line'])
                buggy_end_line = int(buggy_line['buggy_end_line'])

                duplicated_line_nums.extend([i for i in range(buggy_start_line, buggy_end_line+1)])
            
            new_tokens.extend(body_tokens)
    else:
        for line in information_lines:
            line_num = int(line['line_number'])

            if bool(line['is_all_comment']) or bool(line['is_blank']):
                body_tokens = []
            else:
                body = str(line['body'])
                body_tokens = list(javalang.tokenizer.tokenize(body))
                body_tokens = list(map(lambda x: str(x.value), body_tokens))

            buggy_line = buggy_lines.get(line_num)
            buggy_line_condition = buggy_line is not None

            if buggy_line_condition:
                if add_special_tokens:
                    is_omission = buggy_line['is_omission'] == True
                    # is_duplicated = buggy_line['is_duplicated'] == True

                    if is_omission:
                        body_tokens = [omit_token1] + body_tokens + [omit_token2]
                    else:
                        body_tokens = [bug_token1] + body_tokens + [bug_token2]
            
            new_tokens.extend(body_tokens)

    return new_tokens

def tokenize_fixed_lines(lines: List[Any]):
    new_tokens = []

    for line in lines:
        is_all_comment = bool(line['is_all_comment'])
        is_blank = bool(line['is_blank'])

        if is_all_comment == False and is_blank == False:
            body = str(line['body'])
            body_tokens = list(javalang.tokenizer.tokenize(body))
            body_tokens = list(map(lambda x: str(x.value), body_tokens))

            new_tokens.extend(body_tokens)
    
    return new_tokens

def map_tokens(src_tokens, tgt_tokens):
    map_src_tokens = []
    map_tgt_tokens = []

    token_map = {}
    for src_buggy_key, src_token_item in src_tokens.items():
        tgt_token_item = tgt_tokens.get(src_buggy_key)

        if (src_token_item is not None) and (tgt_token_item is not None):
            try:
                src_code = " ".join(src_token_item).encode(encoding="utf-8", errors="strict").decode(encoding="utf-8", errors="strict")
                tgt_code = " ".join(tgt_token_item).encode(encoding="utf-8", errors="strict").decode(encoding="utf-8", errors="strict")

                if len(src_code) > 0 and len(tgt_code) > 0:
                    map_src_tokens.append(src_code)
                    map_tgt_tokens.append(tgt_code)

                    token_map[src_buggy_key] = {
                        'source': src_code,
                        'target': tgt_code
                    }
            except Exception: # as ex:
                # print(ex)
                continue

    return (map_src_tokens, map_tgt_tokens, token_map)