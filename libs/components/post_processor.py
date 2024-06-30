
def revert_to_java(prediction: str):
    tokens = prediction.strip().split(" ")
  
    codeLine = ""
    delimiter = JavaDelimiter()
    for i in range(len(tokens)):
        if(tokens[i] == "<unk>"):
            return ""
        if(i+1 < len(tokens)):
            # DEL = delimiters
            # ... = method_referece
            # STR = token with alphabet in it
            if (not is_delimiter(tokens[i])):
                if (not is_delimiter(tokens[i+1])): # STR (i) + STR (i+1)
                    codeLine = codeLine + tokens[i] + " "
                else: # STR(i) + DEL(i+1)
                    codeLine = codeLine + tokens[i]
            else:
                if (tokens[i] == delimiter.varargs): # ... (i) + ANY (i+1)
                    codeLine = codeLine + tokens[i] + " "
                elif (tokens[i] == delimiter.biggerThan): # > (i) + ANY(i+1)
                    codeLine = codeLine + tokens[i] + " "
                elif (tokens[i] == delimiter.rightBrackets and i > 0):
                    if (tokens[i-1] == delimiter.leftBrackets): # [ (i-1) + ] (i)
                        codeLine = codeLine + tokens[i] + " "
                    else: # DEL not([) (i-1) + ] (i)
                        codeLine = codeLine + tokens[i]
                else: # DEL not(... or ]) (i) + ANY
                    codeLine = codeLine + tokens[i]
        else:
            codeLine = codeLine + tokens[i]

    return codeLine


def is_delimiter(token: str):
    return not token.upper().isupper()


class JavaDelimiter:
    @property
    def varargs(self):
        return "..."

    @property
    def rightBrackets(self):
        return "]"

    @property
    def leftBrackets(self):
        return "["

    @property
    def biggerThan(self):
        return ">"