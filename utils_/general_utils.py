import os
import re
import numpy as np


def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_memory_usage():
    # Importing the library to measure RAM usage
    import psutil
    return psutil.virtual_memory()[2]


def replace_all_occurences_in_string(complete_string, old_word, new_word, identify_overlapping_occurences=False):
    """
    This function replaces all occurences of the {old_word} in {complete_string} by the {new_word}.
    """
    
    if identify_overlapping_occurences:
        indices = [m.start() for m in re.finditer('(?={})'.format(old_word), complete_string)]
    else:
        indices = [m.start() for m in re.finditer(old_word, complete_string)]
    
    if len(indices) > 0:
        chunks_of_string = [complete_string[:indices[0]] + new_word]
        for i in range(1, len(indices)):
            chunks_of_string += [complete_string[indices[i-1]+len(old_word):indices[i]] + new_word]
        chunks_of_string += [complete_string[indices[-1]+len(old_word):]]
        complete_string = ''.join(chunks_of_string)
    
    return complete_string