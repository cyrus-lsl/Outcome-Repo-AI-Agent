"""
Configuration management for the Measurement Instrument Assistant.
Validates environment variables and provides configuration settings.
"""
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Application configuration with validation"""
    
    # File paths
    EXCEL_FILE_PATH: str = os.getenv("EXCEL_FILE_PATH", "measurement_instruments.xlsx")
    EXCEL_SHEET_NAME: str = os.getenv("EXCEL_SHEET_NAME", "Measurement Instruments")
    
    # LLM / API Configuration (neutral names so we can use Hugging Face, Llama, etc.)
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "moonshotai/Kimi-K2-Instruct-0905")
    
    # Application Settings
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "8"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "500"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
    
    # Semantic Search Settings
    SEMANTIC_SEARCH_ENABLED: bool = os.getenv("SEMANTIC_SEARCH_ENABLED", "true").lower() == "true"
    SEMANTIC_MODEL: str = os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
    SEMANTIC_THRESHOLD: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.1"))
    
    # Performance Settings
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration settings.
        Returns (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate LLM_API_KEY (we keep reading HF_TOKEN too for backward compatibility)
        api_key = os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN")
        if isinstance(api_key, str):
            api_key = api_key.strip().strip('"\'')
        cls.LLM_API_KEY = api_key
        
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY (or legacy HF_TOKEN) environment variable is required")
        
        # Validate Excel file exists
        if not os.path.exists(cls.EXCEL_FILE_PATH):
            errors.append(f"Excel file not found: {cls.EXCEL_FILE_PATH}")
        
        # Validate numeric settings
        if cls.MAX_RESULTS < 1 or cls.MAX_RESULTS > 50:
            errors.append(f"MAX_RESULTS must be between 1 and 50, got {cls.MAX_RESULTS}")
        
        if cls.MAX_QUERY_LENGTH < 10 or cls.MAX_QUERY_LENGTH > 2000:
            errors.append(f"MAX_QUERY_LENGTH must be between 10 and 2000, got {cls.MAX_QUERY_LENGTH}")
        
        if cls.SEMANTIC_THRESHOLD < 0 or cls.SEMANTIC_THRESHOLD > 1:
            errors.append(f"SEMANTIC_THRESHOLD must be between 0 and 1, got {cls.SEMANTIC_THRESHOLD}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Configuration validated successfully")
        else:
            logger.error(f"Configuration validation failed: {', '.join(errors)}")
        
        return is_valid, errors
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of configuration (excluding sensitive data)"""
        return {
            "excel_file": cls.EXCEL_FILE_PATH,
            "excel_sheet": cls.EXCEL_SHEET_NAME,
            "max_results": cls.MAX_RESULTS,
            "semantic_search_enabled": cls.SEMANTIC_SEARCH_ENABLED,
            "caching_enabled": cls.ENABLE_CACHING,
            "llm_api_key_set": bool(cls.LLM_API_KEY),
        }

