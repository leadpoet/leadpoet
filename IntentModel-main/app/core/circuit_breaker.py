import pybreaker
from loguru import logger

from app.core.config import settings

# Create a listener to log state changes
class CircuitBreakerLogger(pybreaker.CircuitBreakerListener):
    def state_change(self, cb, old_state, new_state):
        logger.warning(f"CircuitBreaker '{cb.name}' state changed from {old_state} to {new_state}")

# --- Validation for Circuit Breaker Settings ---
if not isinstance(settings.CIRCUIT_BREAKER_ERROR_THRESHOLD, int) or settings.CIRCUIT_BREAKER_ERROR_THRESHOLD <= 0:
    raise ValueError("CIRCUIT_BREAKER_ERROR_THRESHOLD must be a positive integer.")

if not isinstance(settings.CIRCUIT_BREAKER_WINDOW_SIZE, (int, float)) or settings.CIRCUIT_BREAKER_WINDOW_SIZE <= 0:
    raise ValueError("CIRCUIT_BREAKER_WINDOW_SIZE must be a positive number.")

# Define the circuit breaker for the LLM service
llm_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.CIRCUIT_BREAKER_ERROR_THRESHOLD,
    reset_timeout=settings.CIRCUIT_BREAKER_WINDOW_SIZE,
    listeners=[CircuitBreakerLogger()],
    name="llm_service"
)

logger.info("LLM circuit breaker initialized with fail_max={}, reset_timeout={}", 
            settings.CIRCUIT_BREAKER_ERROR_THRESHOLD, settings.CIRCUIT_BREAKER_WINDOW_SIZE) 