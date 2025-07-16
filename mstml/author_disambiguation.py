import numpy as np
import sparse_dot_topn.sparse_dot_topn as ct
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, find
from sparse_dot_topn import awesome_cossim_topn


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
    # string = fix_text(string) # fix text encoding issues
    string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
    string = string.lower()  # make lower case

    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)  # remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip()  # get rid of multiple spaces and replace with a single space
    string = ' ' + string +' '  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))


# Function to create dictionary of author names to author IDs
# author_lists is list of sublists of author names
# fmatches is an array of the resulting 3 arrays by N (num authors) from running find() on matrix of matches
def assign_author_ids(author_lists, fmatches):
    # Create list of random numbers to be used as author IDs at least as long as worst case (all unique authors)
    N = 0
    for author_list in author_lists:
        N += len(author_list)

    auth_ids_pool = random.sample(range(1000000, 9999999), N)
    next_id_idx = 0

    # Dictionaries to be returned
    auth_to_authId = {}
    authId_to_auth = {}

    # Iterate through full author list (prior to disambiguation)
    for author_list, fmatch in zip(author_lists, fmatches):

        author_indices = list(range(len(author_list)))

        for auth_idx, auth_name in zip(author_indices, author_list):
            # Find matches, using auth_idx
            matching_inds = fmatch[1][fmatch[0] == auth_idx]
            matching_names = [author_list[i] for i in matching_inds]

            # If this author name has already been assigned an ID, use it
            if auth_name in auth_to_authId:
                auth_id = auth_to_authId[auth_name]
            else:
                # Check if any of the matching names have already been assigned an ID
                existing_id = None
                for match_name in matching_names:
                    if match_name in auth_to_authId:
                        existing_id = auth_to_authId[match_name]
                        break

                if existing_id:
                    auth_id = existing_id
                else:
                    # Assign a new ID
                    auth_id = auth_ids_pool[next_id_idx]
                    next_id_idx += 1

            # Update dictionaries for this author and all matching authors
            for match_name in [auth_name] + matching_names:
                auth_to_authId[match_name] = auth_id
                if auth_id not in authId_to_auth:
                    authId_to_auth[auth_id] = []
                if match_name not in authId_to_auth[auth_id]:
                    authId_to_auth[auth_id].append(match_name)

    return auth_to_authId, authId_to_auth


# Function to perform author disambiguation using TF-IDF and cosine similarity
def disambiguate_authors(author_lists, threshold=0.8):
    """
    Disambiguate authors using TF-IDF vectorization and cosine similarity.
    
    Args:
        author_lists: List of lists, where each inner list contains author names for a document
        threshold: Similarity threshold for considering authors as the same person
        
    Returns:
        Tuple of (auth_to_authId, authId_to_auth) dictionaries
    """
    # Flatten all author names
    all_authors = []
    for author_list in author_lists:
        all_authors.extend(author_list)
    
    # Remove duplicates while preserving order
    unique_authors = list(dict.fromkeys(all_authors))
    
    if not unique_authors:
        return {}, {}
    
    # Create n-grams for each author name
    author_ngrams = [' '.join(ngrams(author)) for author in unique_authors]
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(min_df=1, analyzer='word', token_pattern=r'\S+')
    tfidf_matrix = vectorizer.fit_transform(author_ngrams)
    
    # Find matches using cosine similarity
    fmatches = []
    for author_list in author_lists:
        if not author_list:
            fmatches.append((np.array([]), np.array([]), np.array([])))
            continue
            
        # Get indices of authors in this list
        author_indices = [unique_authors.index(author) for author in author_list if author in unique_authors]
        
        if not author_indices:
            fmatches.append((np.array([]), np.array([]), np.array([])))
            continue
        
        # Create submatrix for this document's authors
        sub_tfidf = tfidf_matrix[author_indices]
        
        # Compute similarity matrix
        similarity_matrix = awesome_cossim_top(sub_tfidf, sub_tfidf.T, 
                                             ntop=len(author_indices), 
                                             lower_bound=threshold)
        
        # Find matches
        matches = find(similarity_matrix)
        fmatches.append(matches)
    
    # Assign author IDs
    return assign_author_ids(author_lists, fmatches)


# Function to create author disambiguation mappings from a dataframe
def create_author_mappings(df, author_column='authors', separator=';'):
    """
    Create author disambiguation mappings from a dataframe.
    
    Args:
        df: Pandas dataframe containing author information
        author_column: Name of the column containing author names
        separator: Character used to separate multiple authors
        
    Returns:
        Tuple of (auth_to_authId, authId_to_auth, author_lists)
    """
    # Extract and clean author lists
    author_lists = []
    for _, row in df.iterrows():
        if pd.isna(row[author_column]):
            author_lists.append([])
        else:
            authors = scrub_author_list(str(row[author_column]), separator)
            author_lists.append(authors)
    
    # Perform disambiguation
    auth_to_authId, authId_to_auth = disambiguate_authors(author_lists)
    
    return auth_to_authId, authId_to_auth, author_lists


# Function to add author IDs to dataframe
def add_author_ids_to_dataframe(df, auth_to_authId, author_lists, separator=','):
    """
    Add author ID column to dataframe.
    
    Args:
        df: Pandas dataframe
        auth_to_authId: Dictionary mapping author names to IDs
        author_lists: List of author name lists for each document
        separator: Character to use when joining multiple author IDs
        
    Returns:
        Modified dataframe with authorID column
    """
    df = df.copy()
    author_id_lists = []
    
    for author_list in author_lists:
        author_ids = []
        for author in author_list:
            if author in auth_to_authId:
                author_ids.append(str(auth_to_authId[author]))
        
        # Add trailing separator for consistency with original format
        if author_ids:
            author_id_string = separator.join(author_ids) + separator
        else:
            author_id_string = ""
            
        author_id_lists.append(author_id_string)
    
    df['authorID'] = author_id_lists
    return df


# Function to get statistics about author disambiguation
def get_disambiguation_stats(auth_to_authId, authId_to_auth):
    """
    Get statistics about the author disambiguation process.
    
    Args:
        auth_to_authId: Dictionary mapping author names to IDs
        authId_to_auth: Dictionary mapping author IDs to lists of names
        
    Returns:
        Dictionary containing disambiguation statistics
    """
    total_author_names = len(auth_to_authId)
    unique_author_ids = len(authId_to_auth)
    
    # Count how many author names were merged
    merged_names = sum(1 for names in authId_to_auth.values() if len(names) > 1)
    
    # Average names per unique author
    avg_names_per_author = total_author_names / unique_author_ids if unique_author_ids > 0 else 0
    
    # Most common author (by number of name variants)
    most_variants = max(len(names) for names in authId_to_auth.values()) if authId_to_auth else 0
    
    stats = {
        'total_author_names': total_author_names,
        'unique_author_ids': unique_author_ids,
        'disambiguation_ratio': unique_author_ids / total_author_names if total_author_names > 0 else 0,
        'merged_author_ids': merged_names,
        'avg_names_per_author': avg_names_per_author,
        'max_name_variants': most_variants
    }
    
    return stats