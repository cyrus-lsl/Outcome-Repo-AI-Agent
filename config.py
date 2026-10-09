"""Configuration for the Measurement Instrument Assistant."""
import logging
import os
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Config:
    EXCEL_FILE_PATH: str = os.getenv('EXCEL_FILE_PATH', 'data/measurement_instruments.xlsx')
    EXCEL_SHEET_NAME: str = os.getenv('EXCEL_SHEET_NAME', 'Measurement Instruments')
    PROJECT_USAGE_PATH: str = os.getenv('PROJECT_USAGE_PATH', 'data/project_usage.xlsx')
    HANDBOOK_PATH: str = os.getenv('HANDBOOK_PATH', 'handbook/user_guidelines.md')

    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = os.getenv('LLM_BASE_URL', 'https://router.huggingface.co/v1')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'moonshotai/Kimi-K2-Instruct-0905')
    ROUTER_MODEL: str = os.getenv('ROUTER_MODEL', os.getenv('LLM_MODEL', 'moonshotai/Kimi-K2-Instruct-0905'))

    MAX_RESULTS: int = int(os.getenv('MAX_RESULTS', '8'))
    MAX_QUERY_LENGTH: int = int(os.getenv('MAX_QUERY_LENGTH', '500'))
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    PREFILTER_TOP_N: int = int(os.getenv('PREFILTER_TOP_N', '25'))
    BM25_PASS1_TOP_N: int = int(os.getenv('BM25_PASS1_TOP_N', '50'))
    BM25_PASS2_TOP_N: int = int(os.getenv('BM25_PASS2_TOP_N', '25'))

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        errors = []
        api_key = os.getenv('LLM_API_KEY') or os.getenv('HF_TOKEN')
        if isinstance(api_key, str):
            api_key = api_key.strip().strip('"\'')
        cls.LLM_API_KEY = api_key

        if not cls.LLM_API_KEY:
            errors.append('LLM_API_KEY (or HF_TOKEN) is required for web AI search')

        for label, path in (
            ('Excel file', cls.EXCEL_FILE_PATH),
            ('Project usage file', cls.PROJECT_USAGE_PATH),
        ):
            if not str(path).startswith(('http://', 'https://')) and not os.path.exists(path):
                errors.append(f'{label} not found: {path}')

        if not 1 <= cls.MAX_RESULTS <= 50:
            errors.append(f'MAX_RESULTS must be 1–50, got {cls.MAX_RESULTS}')

        return len(errors) == 0, errors
