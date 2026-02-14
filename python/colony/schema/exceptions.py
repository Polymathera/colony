

class MaxRetriesExceededError(Exception):
    """The maximum number of retries has been exceeded"""
    pass


class InferenceJobError(Exception):
    """An error occurred during an inference job."""
    pass

class AuthorizationError(Exception):
    """
    Raised when an authorization error occurs.
    """
    pass

class AuthenticationError(Exception):
    """Exception raised for authentication errors."""
    pass

class AlertError(Exception):
    """Base exception for alert-related errors"""
    pass

class GraphStorageError(Exception):
    """Base exception for graph storage errors"""
    pass

class DependencyConflictError(GraphStorageError):
    """Raised when there's a conflict in dependency versions"""
    pass

class CacheError(GraphStorageError):
    """Raised when there's an error with the cache"""
    pass

class SecurityError(GraphStorageError):
    """Raised when there's a security-related error"""
    pass

class SecurityError(Exception):
    """Exception raised for security-related issues"""
    pass

class CircuitBreakerOpenError(Exception):
    """Exception raised when the circuit breaker is open"""
    pass

class ShardingError(Exception):
    """Base class for all sharding-related errors"""
    pass
