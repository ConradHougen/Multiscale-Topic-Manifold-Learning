"""
Test cases for dataframe_schema.py module
"""

import pytest
import datetime as dt
from unittest.mock import Mock

# Import with fallback for different environments
try:
    from mstml.dataframe_schema import MainDataSchema, FieldDef
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mstml.dataframe_schema import MainDataSchema, FieldDef
    except ImportError:
        # Direct import as fallback
        sys.path.insert(0, str(Path(__file__).parent.parent / "mstml"))
        from dataframe_schema import MainDataSchema, FieldDef


class TestFieldDef:
    """Test cases for FieldDef namedtuple"""
    
    def test_field_def_creation(self):
        """Test FieldDef can be created with required fields"""
        field = FieldDef("test_col", lambda x: x.get("test"), str)
        assert field.column_name == "test_col"
        assert field.type == str
        assert callable(field.extractor)
    
    def test_field_def_extractor_execution(self):
        """Test FieldDef extractor can be called"""
        field = FieldDef("test_col", lambda x: x.get("value", "default"), str)
        result = field.extractor({"value": "test"})
        assert result == "test"
        
        # Test with missing key
        result = field.extractor({})
        assert result == "default"


class TestMainDataSchema:
    """Test cases for MainDataSchema enum"""
    
    def test_schema_fields_exist(self):
        """Test that all expected schema fields are defined"""
        expected_fields = ['TITLE', 'DATE', 'RAW_TEXT', 'AUTHOR_NAMES', 
                          'AUTHOR_IDS', 'PREPROCESSED_TEXT']
        
        for field_name in expected_fields:
            assert hasattr(MainDataSchema, field_name)
    
    def test_colname_property(self):
        """Test colname property returns correct column name"""
        assert MainDataSchema.TITLE.colname == "title"
        assert MainDataSchema.DATE.colname == "date"
        assert MainDataSchema.RAW_TEXT.colname == "raw_text"
        assert MainDataSchema.AUTHOR_NAMES.colname == "author_names"
        assert MainDataSchema.AUTHOR_IDS.colname == "author_ids"
        assert MainDataSchema.PREPROCESSED_TEXT.colname == "preprocessed_text"
    
    def test_get_extractor_method(self):
        """Test get_extractor returns callable function"""
        for field in MainDataSchema:
            extractor = field.get_extractor()
            assert callable(extractor)
    
    def test_all_colnames_classmethod(self):
        """Test all_colnames returns list of column names"""
        colnames = MainDataSchema.all_colnames()
        expected = ["title", "date", "raw_text", "author_names", 
                   "author_ids", "preprocessed_text"]
        assert colnames == expected
    
    def test_all_fields_classmethod(self):
        """Test all_fields returns list of enum members"""
        fields = MainDataSchema.all_fields()
        assert len(fields) == 6
        assert all(isinstance(field, MainDataSchema) for field in fields)


class TestMainDataSchemaExtractors:
    """Test cases for MainDataSchema field extractors"""
    
    def test_title_extractor(self):
        """Test TITLE extractor handles various inputs"""
        extractor = MainDataSchema.TITLE.get_extractor()
        
        # Normal case
        entry = {"title": "Test Title"}
        assert extractor(entry) == "Test Title"
        
        # With whitespace
        entry = {"title": "  Test Title  "}
        assert extractor(entry) == "Test Title"
        
        # Missing title
        entry = {}
        assert extractor(entry) == ""
        
        # Empty title
        entry = {"title": ""}
        assert extractor(entry) == ""
    
    def test_date_extractor(self):
        """Test DATE extractor handles various date formats"""
        extractor = MainDataSchema.DATE.get_extractor()
        
        # Valid ISO date
        entry = {"date": "2023-12-25T10:30:00Z"}
        result = extractor(entry)
        expected = dt.date(2023, 12, 25)
        assert result == expected
        
        # Valid date only
        entry = {"date": "2023-12-25"}
        result = extractor(entry)
        expected = dt.date(2023, 12, 25)
        assert result == expected
        
        # Missing date
        entry = {}
        result = extractor(entry)
        assert result is None
        
        # Invalid date format should raise exception
        entry = {"date": "invalid-date"}
        with pytest.raises(ValueError):
            extractor(entry)
    
    def test_raw_text_extractor(self):
        """Test RAW_TEXT extractor handles various inputs"""
        extractor = MainDataSchema.RAW_TEXT.get_extractor()
        
        # Normal case
        entry = {"raw_text": "This is raw text"}
        assert extractor(entry) == "This is raw text"
        
        # With whitespace
        entry = {"raw_text": "  This is raw text  "}
        assert extractor(entry) == "This is raw text"
        
        # Missing raw_text
        entry = {}
        assert extractor(entry) == ""
        
        # Empty raw_text
        entry = {"raw_text": ""}
        assert extractor(entry) == ""
    
    def test_author_names_extractor(self):
        """Test AUTHOR_NAMES extractor handles various author formats"""
        extractor = MainDataSchema.AUTHOR_NAMES.get_extractor()
        
        # Normal case with valid authors
        entry = {
            "authors": [
                {"name": "John Doe"},
                {"name": "Jane Smith"}
            ]
        }
        result = extractor(entry)
        assert result == ["John Doe", "Jane Smith"]
        
        # Authors with missing name field
        entry = {
            "authors": [
                {"name": "John Doe"},
                {"affiliation": "University"},  # Missing name
                {"name": "Jane Smith"}
            ]
        }
        result = extractor(entry)
        assert result == ["John Doe", "Jane Smith"]
        
        # Non-dict authors
        entry = {
            "authors": [
                {"name": "John Doe"},
                "Not a dict",  # Invalid format
                {"name": "Jane Smith"}
            ]
        }
        result = extractor(entry)
        assert result == ["John Doe", "Jane Smith"]
        
        # Missing authors field
        entry = {}
        result = extractor(entry)
        assert result == []
        
        # Empty authors list
        entry = {"authors": []}
        result = extractor(entry)
        assert result == []
    
    def test_author_ids_extractor(self):
        """Test AUTHOR_IDS extractor returns None (placeholder)"""
        extractor = MainDataSchema.AUTHOR_IDS.get_extractor()
        entry = {"some": "data"}
        result = extractor(entry)
        assert result is None
    
    def test_preprocessed_text_extractor(self):
        """Test PREPROCESSED_TEXT extractor returns None (placeholder)"""
        extractor = MainDataSchema.PREPROCESSED_TEXT.get_extractor()
        entry = {"some": "data"}
        result = extractor(entry)
        assert result is None


class TestMainDataSchemaTypes:
    """Test cases for MainDataSchema field types"""
    
    def test_field_types(self):
        """Test that field types are correctly defined"""
        assert MainDataSchema.TITLE.value.type == str
        assert MainDataSchema.DATE.value.type == dt.date
        assert MainDataSchema.RAW_TEXT.value.type == str
        assert MainDataSchema.AUTHOR_NAMES.value.type == list
        assert MainDataSchema.AUTHOR_IDS.value.type == list
        assert MainDataSchema.PREPROCESSED_TEXT.value.type == list


@pytest.fixture
def sample_json_entry():
    """Sample JSON entry for testing"""
    return {
        "title": "Sample Research Paper",
        "date": "2023-12-25T10:30:00Z",
        "raw_text": "This is the abstract of the paper.",
        "authors": [
            {"name": "John Doe", "affiliation": "University A"},
            {"name": "Jane Smith", "affiliation": "University B"}
        ]
    }


class TestMainDataSchemaIntegration:
    """Integration tests for MainDataSchema with real-like data"""
    
    def test_full_extraction_workflow(self, sample_json_entry):
        """Test extracting all fields from a complete entry"""
        results = {}
        for field in MainDataSchema:
            extractor = field.get_extractor()
            results[field.colname] = extractor(sample_json_entry)
        
        # Verify extracted values
        assert results["title"] == "Sample Research Paper"
        assert results["date"] == dt.date(2023, 12, 25)
        assert results["raw_text"] == "This is the abstract of the paper."
        assert results["author_names"] == ["John Doe", "Jane Smith"]
        assert results["author_ids"] is None
        assert results["preprocessed_text"] is None
    
    def test_malformed_entry_handling(self):
        """Test schema handles malformed entries gracefully"""
        malformed_entry = {
            "title": 123,  # Wrong type but should be converted to string
            "date": "not-a-date",  # Invalid date format
            "authors": "not-a-list"  # Wrong type
        }
        
        # Title should work (converted to string and stripped)
        title_result = MainDataSchema.TITLE.get_extractor()(malformed_entry)
        assert title_result == "123"
        
        # Date should raise exception
        with pytest.raises(ValueError):
            MainDataSchema.DATE.get_extractor()(malformed_entry)
        
        # Authors should return empty list (no valid authors)
        author_result = MainDataSchema.AUTHOR_NAMES.get_extractor()(malformed_entry)
        assert author_result == []