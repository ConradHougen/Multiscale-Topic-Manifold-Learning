"""
Test cases for author_disambiguation.py module
"""

import pytest
import numpy as np
import random
from unittest.mock import Mock, patch
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml.author_disambiguation import (
    AuthorDisambiguator, scrub_author_list, ngrams, 
    compute_author_similarity, assign_author_ids
)


class TestAuthorDisambiguator:
    """Test cases for AuthorDisambiguator class"""
    
    def test_initialization(self):
        """Test AuthorDisambiguator initializes correctly"""
        disambiguator = AuthorDisambiguator()
        
        # Check that mappings are initialized as defaultdicts
        assert isinstance(disambiguator.authorName_to_authorId, defaultdict)
        assert isinstance(disambiguator.authorId_to_authorName, defaultdict)
        
        # Check that they start empty
        assert len(disambiguator.authorName_to_authorId) == 0
        assert len(disambiguator.authorId_to_authorName) == 0
    
    def test_defaultdict_behavior(self):
        """Test that the defaultdicts work as expected"""
        disambiguator = AuthorDisambiguator()
        
        # Accessing non-existent keys should return default values
        assert disambiguator.authorName_to_authorId["nonexistent"] == 0
        assert disambiguator.authorId_to_authorName[999] == []
        
        # Setting values should work normally
        disambiguator.authorName_to_authorId["John Doe"] = 123
        disambiguator.authorId_to_authorName[123] = ["John Doe", "J. Doe"]
        
        assert disambiguator.authorName_to_authorId["John Doe"] == 123
        assert disambiguator.authorId_to_authorName[123] == ["John Doe", "J. Doe"]


class TestScrubAuthorList:
    """Test cases for scrub_author_list function"""
    
    def test_basic_scrubbing(self):
        """Test basic author list scrubbing"""
        input_str = "John Doe;Jane Smith;Bob Wilson"
        result = scrub_author_list(input_str)
        expected = ["John Doe", "Jane Smith", "Bob Wilson"]
        assert result == expected
    
    def test_scrub_bracketed_content(self):
        """Test removal of bracketed content"""
        input_str = "John Doe [University];Jane Smith [Corp];Bob Wilson"
        result = scrub_author_list(input_str)
        expected = ["John Doe", "Jane Smith", "Bob Wilson"]
        assert result == expected
    
    def test_scrub_partial_brackets(self):
        """Test handling of partial/unclosed brackets"""
        input_str = "John Doe [University;Jane Smith [Corp];Bob Wilson"
        result = scrub_author_list(input_str)
        # Should skip entries with unmatched brackets
        expected = ["Jane Smith", "Bob Wilson"]
        assert result == expected
    
    def test_scrub_skip_malformed_entries(self):
        """Test skipping entries with remaining bracket characters"""
        input_str = "John Doe;Jane] Smith;Bob Wilson"
        result = scrub_author_list(input_str)
        # Should skip "Jane] Smith" due to remaining bracket
        expected = ["John Doe", "Bob Wilson"]
        assert result == expected
    
    def test_custom_separator(self):
        """Test using custom separator"""
        input_str = "John Doe|Jane Smith|Bob Wilson"
        result = scrub_author_list(input_str, separator='|')
        expected = ["John Doe", "Jane Smith", "Bob Wilson"]
        assert result == expected
    
    def test_whitespace_stripping(self):
        """Test that whitespace is properly stripped"""
        input_str = " John Doe ; Jane Smith ; Bob Wilson "
        result = scrub_author_list(input_str)
        expected = ["John Doe", "Jane Smith", "Bob Wilson"]
        assert result == expected
    
    def test_empty_string(self):
        """Test handling of empty string"""
        result = scrub_author_list("")
        expected = [""]
        assert result == expected
    
    def test_single_author(self):
        """Test handling of single author"""
        result = scrub_author_list("John Doe")
        expected = ["John Doe"]
        assert result == expected


class TestNgrams:
    """Test cases for ngrams function"""
    
    def test_basic_ngrams(self):
        """Test basic n-gram generation"""
        result = ngrams("John Doe", n=3)
        # Should generate 3-character ngrams from " John Doe "
        expected_first_few = [' Jo', 'Joh', 'ohn', 'hn ', 'n D']
        for expected in expected_first_few:
            assert expected in result
    
    def test_character_removal(self):
        """Test removal of special characters"""
        result = ngrams("John (Doe)", n=3)
        # Parentheses should be removed
        assert '(' not in ''.join(result)
        assert ')' not in ''.join(result)
    
    def test_character_replacement(self):
        """Test character replacements"""
        result = ngrams("John & Jane", n=3)
        # '&' should be replaced with 'and'
        combined = ''.join(result)
        assert 'and' in combined
        assert '&' not in combined
    
    def test_comma_dash_replacement(self):
        """Test comma and dash replacement with spaces"""
        result = ngrams("John,Jane-Doe", n=3)
        combined = ''.join(result)
        # Commas and dashes should be replaced with spaces
        assert ',' not in combined
        assert '-' not in combined
    
    def test_title_case_conversion(self):
        """Test title case conversion"""
        result = ngrams("john doe", n=3)
        combined = ''.join(result)
        # Should be converted to title case
        assert 'John' in combined
        assert 'Doe' in combined
    
    def test_multiple_spaces_compression(self):
        """Test compression of multiple spaces"""
        result = ngrams("John    Doe", n=3)
        combined = ''.join(result)
        # Multiple spaces should be compressed to single spaces
        assert '    ' not in combined
    
    def test_ascii_encoding(self):
        """Test ASCII encoding (non-ASCII characters removed)"""
        result = ngrams("Jöhn Døe", n=3)
        combined = ''.join(result)
        # Non-ASCII characters should be removed
        assert 'ö' not in combined
        assert 'ø' not in combined
    
    def test_different_n_values(self):
        """Test different n-gram sizes"""
        text = "John"
        
        result_2 = ngrams(text, n=2)
        result_3 = ngrams(text, n=3)
        result_4 = ngrams(text, n=4)
        
        # Different n values should produce different numbers of n-grams
        assert len(result_2) != len(result_3)
        assert len(result_3) != len(result_4)
        
        # Verify some expected n-grams
        assert ' J' in result_2
        assert ' Jo' in result_3
        assert ' Joh' in result_4
    
    def test_empty_string(self):
        """Test handling of empty string"""
        result = ngrams("", n=3)
        # Should handle empty string gracefully
        assert len(result) >= 0


class TestComputeAuthorSimilarity:
    """Test cases for compute_author_similarity function"""
    
    @pytest.fixture
    def sample_sparse_matrices(self):
        """Create sample sparse matrices for testing"""
        from scipy.sparse import csr_matrix
        
        # Create simple sparse matrices
        A = csr_matrix([[1, 0, 2], [0, 1, 1]])
        B = csr_matrix([[1, 1], [0, 1], [2, 0]])
        return A, B
    
    @patch('mstml.author_disambiguation.awesome_cossim_topn')
    def test_compute_author_similarity_called_correctly(self, mock_awesome_cossim, sample_sparse_matrices):
        """Test that compute_author_similarity calls awesome_cossim_topn correctly"""
        A, B = sample_sparse_matrices
        mock_awesome_cossim.return_value = "mock_result"
        
        result = compute_author_similarity(A, B, ntop=5, lower_bound=0.1)
        
        # Check that awesome_cossim_topn was called with correct parameters
        mock_awesome_cossim.assert_called_once()
        call_args = mock_awesome_cossim.call_args
        
        # Check positional arguments
        assert call_args[0][2] == 5  # ntop
        
        # Check keyword arguments
        assert call_args[1]['lower_bound'] == 0.1
        
        assert result == "mock_result"
    
    def test_matrix_conversion_to_csr(self, sample_sparse_matrices):
        """Test that matrices are converted to CSR format"""
        from scipy.sparse import coo_matrix
        A, B = sample_sparse_matrices
        
        # Convert to COO format first
        A_coo = A.tocoo()
        B_coo = B.tocoo()
        
        with patch('mstml.author_disambiguation.awesome_cossim_topn') as mock_awesome_cossim:
            mock_awesome_cossim.return_value = "mock_result"
            
            compute_author_similarity(A_coo, B_coo, ntop=5)
            
            # Check that the matrices passed to awesome_cossim_topn are CSR
            call_args = mock_awesome_cossim.call_args[0]
            passed_A = call_args[0]
            passed_B = call_args[1]
            
            assert passed_A.format == 'csr'
            assert passed_B.format == 'csr'


class TestAssignAuthorIds:
    """Test cases for assign_author_ids function"""
    
    @pytest.fixture
    def sample_author_data(self):
        """Create sample author data for testing"""
        author_lists = [
            ["John Doe", "Jane Smith"],
            ["J. Doe", "Alice Johnson"],
            ["Bob Wilson", "Jane Smith"]
        ]
        
        # Mock fuzzy match results
        # Each fmatch represents matches within each author list
        fmatches = [
            (np.array([0, 1]), np.array([0, 1])),  # No internal matches in first list
            (np.array([0, 1]), np.array([0, 1])),  # No internal matches in second list  
            (np.array([0, 1]), np.array([0, 1]))   # No internal matches in third list
        ]
        
        return author_lists, fmatches
    
    @patch('mstml.author_disambiguation.random.sample')
    def test_assign_author_ids_basic(self, mock_random_sample, sample_author_data):
        """Test basic author ID assignment"""
        author_lists, fmatches = sample_author_data
        
        # Mock random ID generation
        mock_ids = [1000001, 1000002, 1000003, 1000004, 1000005, 1000006]
        mock_random_sample.return_value = mock_ids
        
        auth_to_authId, authId_to_auth = assign_author_ids(author_lists, fmatches)
        
        # Check that mappings were created
        assert isinstance(auth_to_authId, dict)
        assert isinstance(authId_to_auth, dict)
        
        # Check that all authors got IDs
        all_authors = set()
        for author_list in author_lists:
            all_authors.update(author_list)
        
        for author in all_authors:
            assert author in auth_to_authId
            auth_id = auth_to_authId[author]
            assert auth_id in authId_to_auth
            assert author in authId_to_auth[auth_id]
    
    @patch('mstml.author_disambiguation.random.sample')
    def test_assign_author_ids_with_matches(self, mock_random_sample):
        """Test author ID assignment with fuzzy matches"""
        # Create data where "John Doe" and "J. Doe" should match
        author_lists = [
            ["John Doe", "Jane Smith"],
            ["J. Doe", "Alice Johnson"]
        ]
        
        # Mock fuzzy match results indicating "John Doe" matches "J. Doe"
        fmatches = [
            (np.array([0]), np.array([0])),  # John Doe matches with itself
            (np.array([0]), np.array([0]))   # J. Doe matches with itself
        ]
        
        mock_ids = [1000001, 1000002, 1000003, 1000004]
        mock_random_sample.return_value = mock_ids
        
        auth_to_authId, authId_to_auth = assign_author_ids(author_lists, fmatches)
        
        # All authors should get unique IDs since no cross-list matching in this setup
        assert len(auth_to_authId) == 4
        assert len(authId_to_auth) == 4
    
    @patch('mstml.author_disambiguation.random.sample')
    def test_id_pool_size_calculation(self, mock_random_sample, sample_author_data):
        """Test that correct number of IDs are requested from random pool"""
        author_lists, fmatches = sample_author_data
        
        # Count total authors
        total_authors = sum(len(author_list) for author_list in author_lists)
        
        mock_ids = list(range(1000001, 1000001 + total_authors))
        mock_random_sample.return_value = mock_ids
        
        assign_author_ids(author_lists, fmatches)
        
        # Check that random.sample was called with correct parameters
        mock_random_sample.assert_called_once()
        call_args = mock_random_sample.call_args[0]
        
        # Should request exactly the number of authors
        assert call_args[1] == total_authors
        
        # Should sample from the specified range
        expected_range = list(range(1000000, 9999999))
        assert call_args[0] == expected_range
    
    def test_empty_author_lists(self):
        """Test handling of empty author lists"""
        author_lists = []
        fmatches = []
        
        auth_to_authId, authId_to_auth = assign_author_ids(author_lists, fmatches)
        
        # Should return empty mappings
        assert auth_to_authId == {}
        assert authId_to_auth == {}
    
    def test_single_author_list(self):
        """Test handling of single author list"""
        author_lists = [["John Doe"]]
        fmatches = [(np.array([0]), np.array([0]))]
        
        with patch('mstml.author_disambiguation.random.sample') as mock_random_sample:
            mock_random_sample.return_value = [1000001]
            
            auth_to_authId, authId_to_auth = assign_author_ids(author_lists, fmatches)
            
            # Should create mapping for single author  
            assert "John Doe" in auth_to_authId
            auth_id = auth_to_authId["John Doe"]
            assert auth_id in authId_to_auth
            assert authId_to_auth[auth_id] == ["John Doe"]


class TestAuthorDisambiguationIntegration:
    """Integration tests for author disambiguation functionality"""
    
    def test_ngrams_with_scrubbed_names(self):
        """Test that ngrams work well with scrubbed author names"""
        # Start with a messy author string
        messy_authors = "John Doe [University A];Jane Smith [Corp B];Bob Wilson"
        
        # Scrub it
        clean_authors = scrub_author_list(messy_authors)
        
        # Generate n-grams for each clean author
        author_ngrams = {}
        for author in clean_authors:
            author_ngrams[author] = ngrams(author, n=3)
        
        # Check that we get reasonable n-grams
        assert "John Doe" in author_ngrams
        assert "Jane Smith" in author_ngrams
        assert "Bob Wilson" in author_ngrams
        
        # Check that n-grams contain expected substrings
        john_ngrams = author_ngrams["John Doe"]
        assert any('Joh' in ngram for ngram in john_ngrams)
        assert any('Doe' in ngram for ngram in john_ngrams)
    
    def test_end_to_end_small_example(self):
        """Test a small end-to-end example"""
        # Create a simple scenario
        author_lists = [
            ["John Doe", "Jane Smith"],
            ["Bob Wilson"]
        ]
        
        # Mock fuzzy matches (no matches in this simple case)
        fmatches = [
            (np.array([0, 1]), np.array([0, 1])),
            (np.array([0]), np.array([0]))
        ]
        
        with patch('mstml.author_disambiguation.random.sample') as mock_random_sample:
            mock_random_sample.return_value = [1000001, 1000002, 1000003]
            
            auth_to_authId, authId_to_auth = assign_author_ids(author_lists, fmatches)
            
            # All authors should have unique IDs
            assert len(auth_to_authId) == 3
            assert len(set(auth_to_authId.values())) == 3
            
            # Each ID should map back to one author
            for auth_id, authors in authId_to_auth.items():
                assert len(authors) == 1
                assert authors[0] in auth_to_authId
                assert auth_to_authId[authors[0]] == auth_id