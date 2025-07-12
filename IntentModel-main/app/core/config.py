"""
Configuration settings for Leadpoet Intent Model v1.1
Loads settings from environment variables with sensible defaults.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from urllib.parse import quote_plus


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_ENV: str = Field(default="development", description="Application environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Database Configuration
    DB_HOST: str = Field(default="localhost", description="Database host")
    DB_PORT: int = Field(default=5432, description="Database port")
    DB_NAME: str = Field(default="leadpoet", description="Database name")
    DB_USER: str = Field(default="leadpoet", description="Database user")
    DB_PASSWORD: str = Field(default="", description="Database password")
    DB_SSL_MODE: str = Field(default="prefer", description="Database SSL mode")
    
    # Redis Configuration
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    KAFKA_TOPIC_MINER_SNIPPETS: str = Field(default="miner.snippets", description="Kafka topic for miner snippets")
    KAFKA_TOPIC_OUTCOME_EVENTS: str = Field(default="outcome.events", description="Kafka topic for outcome events")
    
    # API Keys
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    PDL_API_KEY: str = Field(default="", description="PDL API key")
    CLEARBIT_API_KEY: str = Field(default="", description="Clearbit API key")
    
    # Scoring Configuration (from BRD Section 9)
    BM25_THRESHOLD: float = Field(default=8.0, description="BM25 threshold for LLM fallback")
    TIME_DECAY_TAU: int = Field(default=90, description="Time decay tau in days")
    MAX_LEADS_PER_QUERY: int = Field(default=2000, description="Maximum leads per query")
    MIN_LEADS_PER_QUERY: int = Field(default=200, description="Minimum leads per query")
    
    # Scoring Weights Configuration
    FIT_SCORE_INDUSTRY_WEIGHT: float = Field(0.4, description="Weight for industry match in fit score")
    FIT_SCORE_SIZE_WEIGHT: float = Field(0.3, description="Weight for company size match in fit score")
    FIT_SCORE_REGION_WEIGHT: float = Field(0.3, description="Weight for region match in fit score")
    FINAL_SCORE_FIT_WEIGHT: float = Field(0.6, description="Weight for fit score in final score calculation")
    FINAL_SCORE_INTENT_WEIGHT: float = Field(0.4, description="Weight for intent score in final score calculation")
    
    # Boost Configuration
    CHURN_BOOST_VALUE: float = Field(20.0, description="Boost value for churn signal (out of 100)")
    JOB_POSTING_BOOST_VALUE: float = Field(15.0, description="Boost value for job posting signal (out of 100)")
    
    # LLM Configuration
    LLM_API_KEY: str = Field(default="", description="OpenAI API key")
    LLM_MODEL: str = Field(default="gpt-4o", description="LLM model to use")
    LLM_MAX_TOKENS: int = Field(default=1000, description="Maximum tokens for LLM responses")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature setting")
    LLM_COST_PER_CALL: float = Field(default=0.01, description="Cost per LLM API call in USD")
    LLM_MAX_CALL_RATIO: float = Field(default=0.3, description="Maximum LLM call ratio")
    
    # Performance Configuration (from BRD Section 10)
    LATENCY_P95_THRESHOLD: int = Field(default=400, description="P95 latency threshold in ms")
    LATENCY_P99_THRESHOLD: int = Field(default=550, description="P99 latency threshold in ms")
    COST_AVG_THRESHOLD: float = Field(default=0.002, description="Average cost threshold per lead")
    COST_P99_THRESHOLD: float = Field(default=0.004, description="P99 cost threshold per lead")
    
    # Retrieval Configuration
    RETRIEVAL_TIMEOUT_MS: int = Field(default=50, description="Retrieval timeout in ms")
    EXEMPLAR_SEARCH_TIMEOUT_MS: int = Field(default=18, description="Exemplar search timeout in ms")
    
    # Scoring Configuration
    SCORING_TIMEOUT_MS: int = Field(default=10, description="Scoring timeout in ms")
    BM25_SCORING_TIMEOUT_MS: int = Field(default=4, description="BM25 scoring timeout in ms")
    LLM_SCORING_TIMEOUT_MS: int = Field(default=120, description="LLM scoring timeout in ms")
    
    # Monitoring Configuration
    METRICS_ENABLED: bool = Field(default=True, description="Enable metrics collection")
    COST_TELEMETRY_ENABLED: bool = Field(default=True, description="Enable cost telemetry")
    PROMETHEUS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics")
    
    # Security Configuration
    CORS_ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], 
        description="Allowed CORS origins"
    )
    CORS_ALLOWED_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
        description="Allowed HTTP methods for CORS"
    )
    CORS_ALLOWED_HEADERS: List[str] = Field(
        default=["*"], 
        description="Allowed headers for CORS"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=False, 
        description="Allow credentials in CORS requests"
    )
    TRUSTED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"], 
        description="Trusted hosts for Host header validation"
    )
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_ERROR_THRESHOLD: int = Field(default=5, description="Circuit breaker error threshold (failures before open)")
    CIRCUIT_BREAKER_WINDOW_SIZE: int = Field(default=60, description="Circuit breaker window size in seconds")
    
    # Cost Configuration
    BASE_COST_PER_LEAD: float = Field(default=0.001, description="Base cost per lead in USD")
    COST_LLM_GPT4O_INPUT_RATE: float = Field(default=0.005, description="LLM GPT-4o input cost per 1K tokens")
    COST_LLM_GPT4O_OUTPUT_RATE: float = Field(default=0.015, description="LLM GPT-4o output cost per 1K tokens")
    COST_API_REQUEST_RATE: float = Field(default=0.0001, description="API request cost per request")
    COST_INFRASTRUCTURE_PER_LEAD_RATE: float = Field(default=0.0005, description="Infrastructure cost per lead processed")
    COST_DAILY_BUDGET_USD: float = Field(default=100.0, description="Daily budget in USD")
    
    # Query Performance Tracking
    QUERY_PERF_CTR_WEIGHT: float = Field(0.6, description="Weight for CTR in query performance score")
    QUERY_PERF_CONVERSION_WEIGHT: float = Field(0.4, description="Weight for conversion rate in query performance score")
    QUERY_PERF_SCORE_THRESHOLD: float = Field(0.5, description="Threshold below which a query is flagged for low performance")
    
    # Simhash Plagiarism Check
    SIMHASH_BIT_DIFFERENCE_THRESHOLD: int = Field(3, description="Bit difference threshold for flagging near-duplicates")
    
    # LightGBM Cold-Start Model
    LGBM_MODEL_PATH: str = Field("models/lgbm_cold_start_model.txt", description="Path to the pre-trained LightGBM model file")
    FEATURE_ENCODER_PATH: str = Field("models/feature_encoder.joblib", description="Path to the pre-trained feature encoder file")
    
    # Prompt Parser Configuration
    PROMPT_PARSER_MODEL: str = Field(default="gpt-4o", description="Model for prompt parsing")
    PROMPT_PARSER_TIMEOUT: int = Field(default=5, description="Timeout for prompt parsing in seconds")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components."""
        if self.DB_PASSWORD:
            return f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode={self.DB_SSL_MODE}"
        else:
            return f"postgresql://{self.DB_USER}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode={self.DB_SSL_MODE}"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.REDIS_PASSWORD:
            return f"redis://:{quote_plus(self.REDIS_PASSWORD)}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def validate_settings(self) -> None:
        """Validate critical settings."""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        # Validate model files exist
        if not os.path.exists(self.LGBM_MODEL_PATH):
            logger.warning(f"LGBM model file not found at path: {self.LGBM_MODEL_PATH}. Using mock.")
        
        if not os.path.exists(self.FEATURE_ENCODER_PATH):
            logger.warning(f"Feature encoder file not found at path: {self.FEATURE_ENCODER_PATH}. Using mock.")
        
        if self.APP_ENV == "production":
            if not self.DB_PASSWORD:
                raise ValueError("DB_PASSWORD is required in production")
            if not self.PDL_API_KEY:
                raise ValueError("PDL_API_KEY is required in production")
            if not self.CLEARBIT_API_KEY:
                raise ValueError("CLEARBIT_API_KEY is required in production")
            
            # Security validation for production
            if "*" in self.CORS_ALLOWED_ORIGINS:
                raise ValueError("CORS_ALLOWED_ORIGINS cannot contain wildcards in production")
            if "*" in self.TRUSTED_HOSTS:
                raise ValueError("TRUSTED_HOSTS cannot contain wildcards in production")
            if self.CORS_ALLOW_CREDENTIALS and "*" in self.CORS_ALLOWED_ORIGINS:
                raise ValueError("CORS_ALLOW_CREDENTIALS cannot be True with wildcard origins")


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_settings()
except ValueError as e:
    import logging
    logging.warning(f"Configuration validation warning: {e}") 