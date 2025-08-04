"""
DataLoader Registry System for extensible file format support.

This module provides a registration system that allows DataLoader classes to
declare their supported file formats, enabling automatic format detection
and loader selection without hardcoding format logic.
"""

from typing import Dict, Set, List, Type, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .data_loaders import DataLoader


class DataLoaderRegistry:
    """
    Registry for mapping file extensions to DataLoader classes.
    
    Provides a clean way to add new file format support without modifying
    existing code. DataLoader classes register themselves with their
    supported file extensions.
    """
    
    _extension_to_loader: Dict[str, Type['DataLoader']] = {}
    _loader_to_extensions: Dict[Type['DataLoader'], Set[str]] = {}
    
    @classmethod
    def register_loader(cls, loader_class: Type, extensions: List[str]) -> None:
        """
        Register a DataLoader class with its supported file extensions.
        
        Args:
            loader_class: The DataLoader class to register
            extensions: List of file extensions (with or without leading dots)
        """
        # Normalize extensions (ensure they have leading dots and are lowercase)
        normalized_extensions = set()
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            normalized_extensions.add(ext.lower())
        
        # Register the mappings
        for ext in normalized_extensions:
            if ext in cls._extension_to_loader:
                existing_loader = cls._extension_to_loader[ext]
                if existing_loader != loader_class:
                    raise ValueError(
                        f"Extension '{ext}' is already registered to {existing_loader.__name__}. "
                        f"Cannot register to {loader_class.__name__}."
                    )
            cls._extension_to_loader[ext] = loader_class
        
        cls._loader_to_extensions[loader_class] = normalized_extensions
    
    @classmethod
    def get_loader_class(cls, extension: str) -> Optional[Type]:
        """
        Get the DataLoader class for a given file extension.
        
        Args:
            extension: File extension (with or without leading dot)
            
        Returns:
            DataLoader class or None if extension not supported
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        return cls._extension_to_loader.get(extension.lower())
    
    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """
        Get all supported file extensions.
        
        Returns:
            Set of supported file extensions (with leading dots)
        """
        return set(cls._extension_to_loader.keys())
    
    @classmethod
    def are_extensions_compatible(cls, extensions: Set[str]) -> bool:
        """
        Check if a set of file extensions are compatible (handled by same loader).
        
        Args:
            extensions: Set of file extensions to check
            
        Returns:
            True if all extensions are handled by the same loader class
        """
        if not extensions:
            return False
        
        # Normalize extensions
        normalized_extensions = set()
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            normalized_extensions.add(ext.lower())
        
        # Get unique loader classes for these extensions
        loader_classes = set()
        for ext in normalized_extensions:
            loader_class = cls.get_loader_class(ext)
            if loader_class is None:
                return False  # Unsupported extension
            loader_classes.add(loader_class)
        
        # Compatible if all extensions map to the same loader class
        return len(loader_classes) == 1
    
    @classmethod
    def get_loader_for_extensions(cls, extensions: Set[str]) -> Optional[Type]:
        """
        Get the common DataLoader class for a set of extensions.
        
        Args:
            extensions: Set of file extensions
            
        Returns:
            DataLoader class if all extensions are compatible, None otherwise
        """
        if not cls.are_extensions_compatible(extensions):
            return None
        
        # Get the loader for any extension (they're all the same)
        first_ext = next(iter(extensions))
        return cls.get_loader_class(first_ext)
    
    @classmethod
    def discover_supported_files(cls, directory: Path) -> List[str]:
        """
        Discover all supported files in a directory.
        
        Args:
            directory: Directory to scan for supported files
            
        Returns:
            Sorted list of supported filenames
        """
        if not directory.exists():
            return []
        
        supported_extensions = cls.get_supported_extensions()
        discovered_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                discovered_files.append(file_path.name)
        
        return sorted(discovered_files)
    
    @classmethod
    def get_extension_info(cls) -> Dict[str, str]:
        """
        Get information about all registered extensions and their loaders.
        
        Returns:
            Dictionary mapping extensions to loader class names
        """
        return {ext: loader.__name__ for ext, loader in cls._extension_to_loader.items()}
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered loaders (mainly for testing)."""
        cls._extension_to_loader.clear()
        cls._loader_to_extensions.clear()


def register_data_loader(*extensions: str):
    """
    Decorator for registering DataLoader classes with their supported extensions.
    
    Usage:
        @register_data_loader('json', 'jsonl')
        class JsonDataLoader(DataLoader):
            pass
    
    Args:
        *extensions: File extensions supported by this loader
    """
    def decorator(loader_class: Type) -> Type:
        DataLoaderRegistry.register_loader(loader_class, list(extensions))
        return loader_class
    return decorator


