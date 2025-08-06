"""
Test cases for data_loaders.py module
"""

import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import datetime as dt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mstml.data_loaders import (
    DataLoader, JsonDataLoader, get_project_root_directory, 
    get_data_directory
)
from mstml.dataframe_schema import MainDataSchema
from mstml.text_preprocessing import TextPreprocessor
from mstml.author_disambiguation import AuthorDisambiguator


class TestUtilityFunctions:
    """Test utility functions in data_loaders module"""
    
    def test_get_project_root_directory(self):
        """Test get_project_root_directory returns correct path"""
        root_dir = get_project_root_directory()
        assert isinstance(root_dir, Path)
        assert root_dir.exists()
        assert root_dir.name == "Multiscale-Topic-Manifold-Learning"
    
    def test_get_data_directory(self):
        """Test get_data_directory returns correct path"""
        data_dir = get_data_directory()
        assert isinstance(data_dir, Path)
        assert data_dir.name == "data"
        # Note: data directory may not exist yet


class TestDataLoaderBase:
    """Test cases for DataLoader abstract base class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_input_file(self, temp_dir):
        """Create sample input file for testing"""
        input_file = temp_dir / "input.json"
        input_file.write_text('{"title": "test"}')
        return input_file
    
    def test_dataloader_cannot_be_instantiated(self):
        """Test DataLoader is abstract and cannot be instantiated directly"""
        with pytest.raises(TypeError):
            DataLoader("input.json", "test_dataset")
    
    def test_setup_dataset_dirs(self, temp_dir):
        """Test setup_dataset_dirs creates correct directory structure"""
        # Mock get_data_directory to return our temp dir
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            dirs = DataLoader.setup_dataset_dirs("test_dataset")
        
        # Check returned dict structure
        assert "root" in dirs
        assert "original" in dirs
        assert "clean" in dirs
        assert "networks" in dirs
        
        # Check directories were created
        assert dirs["root"].exists()
        assert dirs["original"].exists()
        assert dirs["clean"].exists()
        assert dirs["networks"].exists()
        
        # Check directory names
        assert dirs["root"].name == "test_dataset"
        assert dirs["original"].name == "original"
        assert dirs["clean"].name == "clean"
        assert dirs["networks"].name == "networks"
    
    def test_setup_dataset_dirs_overwrite_false(self, temp_dir):
        """Test setup_dataset_dirs respects overwrite=False"""
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            # Create directories first time
            dirs1 = DataLoader.setup_dataset_dirs("test_dataset")
            
            # Create marker file
            marker_file = dirs1["original"] / "marker.txt"
            marker_file.write_text("test")
            
            # Try to create again with overwrite=False (default)
            dirs2 = DataLoader.setup_dataset_dirs("test_dataset", overwrite=False)
            
            # Marker file should still exist
            assert marker_file.exists()
    
    def test_setup_dataset_dirs_overwrite_true(self, temp_dir):
        """Test setup_dataset_dirs respects overwrite=True"""
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            # Create directories first time
            dirs1 = DataLoader.setup_dataset_dirs("test_dataset")
            
            # Create directories again with overwrite=True should work
            dirs2 = DataLoader.setup_dataset_dirs("test_dataset", overwrite=True)
            
            # Should still have correct structure
            assert dirs2["original"].exists()


class ConcreteDataLoader(DataLoader):
    """Concrete implementation of DataLoader for testing"""
    
    def _load_raw_data(self):
        self.raw_data = [{"title": "test", "date": "2023-01-01", "raw_text": "content"}]
    
    def _preprocess(self):
        rows = []
        for entry in self.raw_data:
            row = {field.colname: field.get_extractor()(entry) for field in MainDataSchema}
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self._apply_preprocessors()


class TestDataLoaderConcrete:
    """Test cases using concrete DataLoader implementation"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_input_file(self, temp_dir):
        """Create sample input file for testing"""
        input_file = temp_dir / "input.json"
        input_file.write_text('{"title": "test"}')
        return input_file
    
    @pytest.fixture
    def data_loader(self, sample_input_file, temp_dir):
        """Create ConcreteDataLoader instance for testing"""
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            loader = ConcreteDataLoader(
                str(sample_input_file), 
                "test_dataset",
                overwrite=True
            )
        return loader
    
    def test_dataloader_initialization(self, data_loader):
        """Test DataLoader initializes correctly"""
        assert data_loader._dataset_name == "test_dataset"
        assert data_loader._overwrite is True
        assert isinstance(data_loader.text_preprocessor, TextPreprocessor)
        assert isinstance(data_loader.author_disambiguator, AuthorDisambiguator)
    
    def test_dataloader_properties_before_setup(self, data_loader):
        """Test DataLoader properties before setup"""
        assert data_loader.dataset_dirs is None
        assert data_loader.input_path is None
        assert data_loader.df is None
        assert data_loader._valid_mask is None
    
    def test_dataloader_run_pipeline(self, data_loader):
        """Test complete DataLoader run pipeline"""
        # Mock the preprocessors to avoid complex dependencies
        data_loader.text_preprocessor = Mock()
        data_loader.text_preprocessor.update_dataframe = Mock(return_value=["processed"])
        data_loader.author_disambiguator = Mock()
        data_loader.author_disambiguator.update_dataframe = Mock(return_value=[["author1"]])
        
        # Run the pipeline
        data_loader.run()
        
        # Check that directories were set up
        assert data_loader.dataset_dirs is not None
        assert data_loader.dataset_root_dir.exists()
        assert data_loader.dataset_original_dir.exists()
        assert data_loader.dataset_clean_dir.exists()
        
        # Check that data was loaded and processed
        assert data_loader.df is not None
        assert len(data_loader.df) > 0
        assert data_loader._valid_mask is not None
    
    def test_get_clean_df_before_validation(self, data_loader):
        """Test get_clean_df raises error before validation"""
        with pytest.raises(RuntimeError, match="Validation not run yet"):
            data_loader.get_clean_df()
    
    def test_get_na_df_before_validation(self, data_loader):
        """Test get_na_df raises error before validation"""
        with pytest.raises(RuntimeError, match="Validation not run yet"):
            data_loader.get_na_df()
    
    def test_resolve_input_path_already_in_original(self, data_loader, temp_dir):
        """Test _resolve_input_path when file is already in original dir"""
        # Set up directories first
        data_loader._prepare_environment()
        
        # Create file in original directory
        original_file = data_loader.dataset_original_dir / "test.json"
        original_file.write_text('{"test": "data"}')
        
        # Should return same path
        result = data_loader._resolve_input_path(original_file)
        assert result == original_file
    
    def test_resolve_input_path_local_file(self, data_loader, temp_dir):
        """Test _resolve_input_path copies local file to original"""
        # Set up directories first
        data_loader._prepare_environment()
        
        # Create file outside original directory
        external_file = temp_dir / "external.json"
        external_file.write_text('{"test": "data"}')
        
        # Should copy to original directory
        result = data_loader._resolve_input_path(external_file)
        expected_path = data_loader.dataset_original_dir / "external.json"
        
        assert result == expected_path
        assert expected_path.exists()
        assert expected_path.read_text() == '{"test": "data"}'
    
    def test_resolve_input_path_file_not_found(self, data_loader):
        """Test _resolve_input_path raises error for non-existent file"""
        # Set up directories first
        data_loader._prepare_environment()
        
        non_existent = Path("/non/existent/file.json")
        
        with pytest.raises(FileNotFoundError):
            data_loader._resolve_input_path(non_existent)


class TestJsonDataLoader:
    """Test cases for JsonDataLoader"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create sample JSON lines file for testing"""
        json_file = temp_dir / "sample.jsonl"
        sample_data = [
            {
                "title": "First Paper",
                "date": "2023-01-01",
                "raw_text": "This is the first paper content.",
                "authors": [
                    {"name": "John Doe"},
                    {"name": "Jane Smith"}
                ]
            },
            {
                "title": "Second Paper", 
                "date": "2023-01-02",
                "raw_text": "This is the second paper content.",
                "authors": [
                    {"name": "Alice Johnson"}
                ]
            }
        ]
        
        with open(json_file, 'w') as f:
            for entry in sample_data:
                json.dump(entry, f)
                f.write('\n')
        
        return json_file
    
    @pytest.fixture 
    def json_loader(self, sample_json_file, temp_dir):
        """Create JsonDataLoader instance for testing"""
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            loader = JsonDataLoader(
                str(sample_json_file),
                "test_json_dataset",
                overwrite=True
            )
        return loader
    
    def test_json_loader_initialization(self, json_loader):
        """Test JsonDataLoader initializes correctly"""
        assert isinstance(json_loader, JsonDataLoader)
        assert isinstance(json_loader, DataLoader)
        assert json_loader._dataset_name == "test_json_dataset"
    
    def test_json_loader_load_raw_data(self, json_loader, temp_dir):
        """Test JsonDataLoader loads raw JSON data correctly"""
        # Set up environment and prepare input
        json_loader._prepare_environment()
        json_loader._prepare_input()
        
        # Load raw data
        json_loader._load_raw_data()
        
        # Check raw data was loaded
        assert json_loader.raw_data is not None
        assert len(json_loader.raw_data) == 2
        assert json_loader.raw_data[0]["title"] == "First Paper"
        assert json_loader.raw_data[1]["title"] == "Second Paper"
    
    def test_json_loader_preprocess(self, json_loader, temp_dir):
        """Test JsonDataLoader preprocessing"""
        # Mock preprocessors to avoid complex dependencies
        json_loader.text_preprocessor = Mock()
        json_loader.text_preprocessor.update_dataframe = Mock(
            return_value=["processed1", "processed2"]
        )
        json_loader.author_disambiguator = Mock()
        json_loader.author_disambiguator.update_dataframe = Mock(
            return_value=[["author1", "author2"], ["author3"]]
        )
        
        # Set up and load data
        json_loader._prepare_environment()
        json_loader._prepare_input()
        json_loader._load_raw_data()
        
        # Preprocess
        json_loader._preprocess()
        
        # Check DataFrame was created
        assert json_loader.df is not None
        assert len(json_loader.df) == 2
        
        # Check required columns exist
        expected_cols = MainDataSchema.all_colnames()
        for col in expected_cols:
            assert col in json_loader.df.columns
        
        # Check some data values
        assert json_loader.df.iloc[0]["title"] == "First Paper"
        assert json_loader.df.iloc[1]["title"] == "Second Paper"
    
    def test_json_loader_full_pipeline(self, json_loader):
        """Test JsonDataLoader complete pipeline"""
        # Mock preprocessors
        json_loader.text_preprocessor = Mock()
        json_loader.text_preprocessor.update_dataframe = Mock(
            return_value=["processed1", "processed2"]
        )
        json_loader.author_disambiguator = Mock()
        json_loader.author_disambiguator.update_dataframe = Mock(
            return_value=[["author1"], ["author2"]]
        )
        
        # Run complete pipeline
        json_loader.run()
        
        # Check final state
        assert json_loader.df is not None
        assert json_loader._valid_mask is not None
        assert json_loader.dataset_dirs is not None
        
        # Check files were created
        assert (json_loader.dataset_clean_dir / "main_df.pkl").exists()
        assert (json_loader.dataset_clean_dir / "clean_df.pkl").exists()
        assert (json_loader.dataset_clean_dir / "na_df.pkl").exists()
        assert (json_loader.dataset_clean_dir / "metadata.json").exists()
    
    def test_json_loader_malformed_entries(self, temp_dir):
        """Test JsonDataLoader handles malformed JSON entries"""
        # Create file with some malformed entries
        json_file = temp_dir / "malformed.jsonl"
        with open(json_file, 'w') as f:
            # Valid entry
            json.dump({"title": "Good Entry", "date": "2023-01-01"}, f)
            f.write('\n')
            # Malformed entry (missing required extractable fields)
            json.dump({"invalid": "entry"}, f)
            f.write('\n')
            # Another valid entry
            json.dump({"title": "Another Good Entry", "date": "2023-01-02"}, f)
            f.write('\n')
        
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            loader = JsonDataLoader(str(json_file), "malformed_test", overwrite=True)
        
        # Mock preprocessors
        loader.text_preprocessor = Mock()
        loader.text_preprocessor.update_dataframe = Mock(return_value=["", ""])
        loader.author_disambiguator = Mock()
        loader.author_disambiguator.update_dataframe = Mock(return_value=[[], []])
        
        # Should handle malformed entries gracefully
        loader.run()
        
        # Should still have some valid data
        assert loader.df is not None
        assert len(loader.df) >= 1  # At least some valid entries


@pytest.fixture
def mock_validate_dataset_name():
    """Mock validate_dataset_name function"""
    with patch('mstml.data_loaders.validate_dataset_name') as mock:
        yield mock


class TestDataLoaderValidation:
    """Test cases for DataLoader validation functionality"""

    @patch('mstml.data_loaders.validate_dataset_name')
    def test_dataset_name_validation_called(self, mock_validate_dataset_name, temp_dir):
        """Test that dataset name validation is called during initialization"""
        sample_file = temp_dir / "test.json"
        sample_file.write_text('{}')
        
        with patch('mstml.data_loaders.get_data_directory', return_value=temp_dir):
            ConcreteDataLoader(str(sample_file), "test_name")
        
        mock_validate_dataset_name.assert_called_once_with("test_name")