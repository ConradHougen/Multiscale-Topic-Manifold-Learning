import numpy as np
import random
import re

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, find
from sparse_dot_topn import awesome_cossim_topn

class AuthorDisambiguator:
    def __init__(self):
        # Dictionaries for mapping author names to unique IDs
        # One ID may be mapped to multiple author names
        self.authorName_to_authorId = defaultdict(int)
        self.authorId_to_authorName = defaultdict(list)


# Takes a string that is a list of authors separated by a specific character
# Removes all characters between '[' and ']'
# Returns list of strings, where each string is an author name
def scrub_author_list(str_in, separator=';'):
    auth_list = str_in.split(separator)
    auth_out = []
    for auth in auth_list:
        auth = re.sub('(\\[.*\\])', '', auth)
        auth = re.sub('\\[.*', '', auth)
        if '[' in auth or ']' in auth:
            continue
        auth = auth.strip()
        auth_out.append(auth)
    return auth_out


def ngrams(string, n=3):
    string = string.encode("ascii", errors="ignore").decode()
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()
    string = re.sub(' +', ' ', string).strip()
    string = ' ' + string + ' '
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams_list = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams_list]


def compute_author_similarity(A, B, ntop, lower_bound=0.0):
    A = A.tocsr()
    B = B.tocsr()
    return awesome_cossim_topn(A, B, ntop, lower_bound=lower_bound)


def assign_author_ids(author_lists, fmatches):
    N = sum(len(author_list) for author_list in author_lists)
    auth_ids_pool = random.sample(range(1000000, 9999999), N)
    next_id_idx = 0

    auth_to_authId = {}
    authId_to_auth = {}

    for author_list, fmatch in zip(author_lists, fmatches):
        for auth_idx, auth_name in enumerate(author_list):
            matching_inds = fmatch[1][fmatch[0] == auth_idx]
            matching_names = [author_list[i] for i in matching_inds]

            if auth_name not in auth_to_authId:
                authId = auth_ids_pool[next_id_idx]
                for name in matching_names:
                    if name not in auth_to_authId:
                        auth_to_authId[name] = authId
                        authId_to_auth.setdefault(authId, []).append(name)
                next_id_idx += 1

    return auth_to_authId, authId_to_auth