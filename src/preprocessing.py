import re
from typing import List

class TextPreprocessor:
    """
    Handles text normalization and tokenization operations.
    """
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Converts raw text to a list of normalized tokens.
        Operations: Lowercasing, removing non-alphabetic characters.
        
        Args:
            text (str): The raw input document.
            
        Returns:
            List[str]: A sequence of processed alphabetical tokens.
        """
        text = str(text).lower()
        # Extract sequences of alphabetic characters only
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens