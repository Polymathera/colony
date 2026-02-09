"""
Key improvements:
1. Extended Content Types:
    - Cloud provider credentials
    - Authentication tokens
    - Personal/financial data
    - Infrastructure details
    - Internal communications
    - Business-specific data
2. Enhanced Actions:
    - Encryption
    - Anonymization
    - Masking
    - Logging
3. Sophisticated Matching:
    - Context preservation
    - Pattern ignoring
    - Custom patterns
    - Match deduplication
4. Alert Management:
    - Batching
    - Deduplication
    - Severity levels
    - Configurable webhooks
5. Monitoring & Metrics:
    - Prometheus integration
    - Detailed logging
    - Performance tracking
"""

import base64
import hashlib
import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from re import Pattern
import aiohttp

from ...metrics.common import BaseMetricsMonitor
from .types import SecurityError

logger = logging.getLogger(__name__)


class SensitiveContentType(Enum):
    # Authentication & Authorization
    API_KEY = "api_key"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    JWT_TOKEN = "jwt_token"
    OAUTH_TOKEN = "oauth_token"
    SESSION_KEY = "session_key"

    # Credentials & Secrets
    AWS_CREDENTIALS = "aws_credentials"
    GCP_CREDENTIALS = "gcp_credentials"
    AZURE_CREDENTIALS = "azure_credentials"
    SSH_KEY = "ssh_key"
    PGP_KEY = "pgp_key"
    CERTIFICATE = "certificate"

    # Personal Information
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    PASSPORT = "passport"
    ADDRESS = "address"
    DOB = "date_of_birth"

    # Infrastructure
    INTERNAL_URL = "internal_url"
    IP_ADDRESS = "ip_address"
    DATABASE_URL = "database_url"
    SERVER_PATH = "server_path"
    HOSTNAME = "hostname"

    # Business Data
    CUSTOMER_DATA = "customer_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    PROPRIETARY_CODE = "proprietary_code"

    # Communication
    INTERNAL_COMMUNICATION = "internal_communication"
    SLACK_WEBHOOK = "slack_webhook"
    TEAMS_WEBHOOK = "teams_webhook"

    # Custom
    CUSTOM_PATTERN = "custom_pattern"


class ContentAction(Enum):
    SKIP = "skip"  # Skip file containing sensitive content
    REDACT = "redact"  # Replace with placeholder
    HASH = "hash"  # Replace with content hash
    ALERT = "alert"  # Process normally but alert
    ENCRYPT = "encrypt"  # Encrypt sensitive content
    ANONYMIZE = "anonymize"  # Replace with anonymous but valid data
    MASK = "mask"  # Partially mask content
    LOG = "log"  # Log occurrence only


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    webhook_url: str | None = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    alert_template: str = "Security alert: {type} found in {file}"
    batch_alerts: bool = True
    batch_interval_seconds: int = 300
    alert_deduplication: bool = True
    max_alerts_per_interval: int = 100


@dataclass
class SecurityPolicy:
    content_types: list[SensitiveContentType]
    action: ContentAction
    custom_patterns: list[Pattern] = field(default_factory=list)
    alert_config: AlertConfig | None = None
    ignore_patterns: list[Pattern] = field(default_factory=list)
    ignore_files: set[str] = field(default_factory=set)
    ignore_directories: set[str] = field(default_factory=set)
    max_matches_per_file: int = 100
    context_lines: int = 3  # Lines of context around matches


@dataclass
class SecurityMatch:
    content_type: SensitiveContentType
    matched_text: str
    file_path: str
    line_number: int
    context: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SecurityProcessorMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing SecurityProcessorMetricsMonitor instance {id(self)}...")
        self.sensitive_content_matches = self.create_counter(
            "security_sensitive_content_matches_total",
            "Number of sensitive content matches found",
            labelnames=["content_type", "action"],
        )
        self.alerts_sent = self.create_counter(
            "security_alerts_sent_total",
            "Number of security alerts sent",
            labelnames=["severity"],
        )
        self.processing_time = self.create_histogram(
            "security_processing_duration_seconds",
            "Time taken to process content"
        )


class SecurityProcessor:
    """Handles sensitive content according to policies"""

    def __init__(self, policies: list[SecurityPolicy]):
        self.policies = policies
        self._compile_patterns()
        self._alert_buffer = []
        self._last_alert_time = datetime.now(timezone.utc)
        self.metrics = SecurityProcessorMetricsMonitor()

    def _compile_patterns(self):
        """Compile regex patterns for sensitive content detection"""
        self.patterns = {
            # Authentication & Authorization
            SensitiveContentType.API_KEY: [
                re.compile(r'(?i)api[_-]key.*?[\'"][a-z0-9]{32,}[\'"]'),
                re.compile(
                    r'(?i)(api[_-]key|access[_-]key|secret[_-]key)[^\n]{0,20}[=:][^\n]{0,20}[\'"]([\w\-=]+)[\'"]'
                ),
            ],
            SensitiveContentType.PASSWORD: [
                re.compile(r'(?i)password.*?[\'"][^\'\"]{8,}[\'"]'),
                re.compile(
                    r'(?i)(password|passwd|pwd)[^\n]{0,20}[=:][^\n]{0,20}[\'"]([\w\-=]+)[\'"]'
                ),
            ],
            SensitiveContentType.PRIVATE_KEY: [
                re.compile(
                    r"-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----[^-]*-----END[^-]*-----"
                ),
                re.compile(r'(?i)private[_-]key.*?[\'"][a-z0-9/+=]{64,}[\'"]'),
            ],
            # Cloud Credentials
            SensitiveContentType.AWS_CREDENTIALS: [
                re.compile(r'(?i)aws[_-]access[_-]key[_-]id.*?[\'"][A-Z0-9]{20}[\'"]'),
                re.compile(
                    r'(?i)aws[_-]secret[_-]access[_-]key.*?[\'"][A-Za-z0-9/+=]{40}[\'"]'
                ),
            ],
            SensitiveContentType.GCP_CREDENTIALS: [
                re.compile(r'(?i)"type": "service_account".*?"private_key"'),
                re.compile(
                    r'(?i)google[_-]oauth.*?[\'"][0-9]-[A-Za-z0-9_]{32}\.apps\.googleusercontent\.com[\'"]'
                ),
            ],
            # Personal Information
            SensitiveContentType.EMAIL: [
                re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            ],
            SensitiveContentType.PHONE_NUMBER: [
                re.compile(r"\b\+?1?\d{10,13}\b"),
                re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            ],
            SensitiveContentType.CREDIT_CARD: [
                re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
            ],
            SensitiveContentType.SSN: [
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            ],
            # Infrastructure
            SensitiveContentType.INTERNAL_URL: [
                re.compile(
                    r"(?i)https?://(?:localhost|127\.0\.0\.1|192\.168\.|10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)[\w\-./]+"
                ),
                re.compile(r"(?i)https?://[a-z0-9-]+\.internal[\w\-./]+"),
            ],
            SensitiveContentType.DATABASE_URL: [
                re.compile(r'(?i)(?:postgres|mysql|mongodb|redis)://[^\s<>"]+'),
            ],
            # Communication
            SensitiveContentType.SLACK_WEBHOOK: [
                re.compile(
                    r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+"
                ),
            ],
        }

    async def process_content(self, content: str, file_path: str) -> str:
        """Process content according to security policies"""
        try:
            for policy in self.policies:
                if self._should_skip_file(file_path, policy):
                    continue

                matches = await self._find_sensitive_content(content, file_path, policy)
                if matches:
                    self._update_metrics(matches, policy.action)
                    content = await self._apply_policy(content, matches, policy)

            return content

        except Exception as e:
            logger.error(f"Error processing content for {file_path}: {e}")
            raise SecurityError(f"Security processing failed: {e!s}")

    def _should_skip_file(self, file_path: str, policy: SecurityPolicy) -> bool:
        """Check if file should be skipped based on policy"""
        if file_path in policy.ignore_files:
            return True

        for dir_pattern in policy.ignore_directories:
            if dir_pattern in file_path:
                return True

        return False

    async def _find_sensitive_content(
        self, content: str, file_path: str, policy: SecurityPolicy
    ) -> list[SecurityMatch]:
        """Find sensitive content matches"""
        matches = []
        lines = content.splitlines()

        for content_type in policy.content_types:
            # Check built-in patterns
            for pattern in self.patterns.get(content_type, []):
                for match in pattern.finditer(content):
                    if len(matches) >= policy.max_matches_per_file:
                        return matches

                    line_number = content.count("\n", 0, match.start()) + 1
                    context = self._get_context(
                        lines, line_number, policy.context_lines
                    )

                    # Check if match should be ignored
                    if not self._is_ignored_match(match.group(), policy):
                        matches.append(
                            SecurityMatch(
                                content_type=content_type,
                                matched_text=match.group(),
                                file_path=file_path,
                                line_number=line_number,
                                context=context,
                            )
                        )

            # Check custom patterns
            for pattern in policy.custom_patterns:
                for match in pattern.finditer(content):
                    if len(matches) >= policy.max_matches_per_file:
                        return matches

                    line_number = content.count("\n", 0, match.start()) + 1
                    context = self._get_context(
                        lines, line_number, policy.context_lines
                    )

                    if not self._is_ignored_match(match.group(), policy):
                        matches.append(
                            SecurityMatch(
                                content_type=SensitiveContentType.CUSTOM_PATTERN,
                                matched_text=match.group(),
                                file_path=file_path,
                                line_number=line_number,
                                context=context,
                            )
                        )

        return matches

    def _is_ignored_match(self, match_text: str, policy: SecurityPolicy) -> bool:
        """Check if match should be ignored"""
        for pattern in policy.ignore_patterns:
            if pattern.search(match_text):
                return True
        return False

    def _get_context(
        self, lines: list[str], line_number: int, context_lines: int
    ) -> str:
        """Get context lines around match"""
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        return "\n".join(lines[start:end])

    async def _apply_policy(
        self, content: str, matches: list[SecurityMatch], policy: SecurityPolicy
    ) -> str:
        """Apply security policy to matches"""
        if policy.action == ContentAction.SKIP:
            raise SecurityError(f"Sensitive content found: {len(matches)} matches")

        elif policy.action == ContentAction.REDACT:
            for match in matches:
                content = content.replace(
                    match.matched_text, f"[REDACTED_{match.content_type.value}]"
                )

        elif policy.action == ContentAction.HASH:
            for match in matches:
                content = content.replace(
                    match.matched_text,
                    f"[HASH_{hashlib.sha256(match.matched_text.encode()).hexdigest()[:8]}]",
                )

        elif policy.action == ContentAction.ENCRYPT:
            for match in matches:
                encrypted = self._encrypt_content(match.matched_text)
                content = content.replace(match.matched_text, encrypted)

        elif policy.action == ContentAction.ANONYMIZE:
            for match in matches:
                anonymized = self._anonymize_content(match)
                content = content.replace(match.matched_text, anonymized)

        elif policy.action == ContentAction.MASK:
            for match in matches:
                masked = self._mask_content(match)
                content = content.replace(match.matched_text, masked)

        if (
            policy.action in {ContentAction.ALERT, ContentAction.LOG}
            and policy.alert_config
        ):
            await self._handle_alert(matches, policy.alert_config)

        return content

    def _encrypt_content(self, content: str) -> str:
        """Encrypt sensitive content"""
        # TODO: Implementation depends on encryption requirements
        # This is a placeholder that uses base64 encoding
        return f"[ENCRYPTED_{base64.b64encode(content.encode()).decode()}]"

    def _anonymize_content(self, match: SecurityMatch) -> str:
        """Replace sensitive content with anonymous but valid data"""
        if match.content_type == SensitiveContentType.EMAIL:
            return f"user_{secrets.token_hex(4)}@example.com"
        elif match.content_type == SensitiveContentType.PHONE_NUMBER:
            return "+1-555-0123"
        elif match.content_type == SensitiveContentType.CREDIT_CARD:
            return "****-****-****-1234"
        # Add more anonymization rules
        return f"[ANONYMIZED_{match.content_type.value}]"

    def _mask_content(self, match: SecurityMatch) -> str:
        """Partially mask sensitive content"""
        content = match.matched_text
        if match.content_type == SensitiveContentType.EMAIL:
            username, domain = content.split("@")
            return f"{username[:2]}***@{domain}"
        elif match.content_type == SensitiveContentType.CREDIT_CARD:
            return re.sub(r"\d(?=\d{4})", "*", content)
        # Add more masking rules
        return re.sub(r".", "*", content[:-4]) + content[-4:]

    async def _handle_alert(
        self, matches: list[SecurityMatch], alert_config: AlertConfig
    ):
        """Handle security alerts"""
        if alert_config.batch_alerts:
            self._alert_buffer.extend(matches)

            # Check if it's time to send batch
            now = datetime.now(timezone.utc)
            if (
                now - self._last_alert_time
            ).total_seconds() >= alert_config.batch_interval_seconds:
                await self._send_batch_alerts(alert_config)
        else:
            await self._send_alerts(matches, alert_config)

    async def _send_batch_alerts(self, alert_config: AlertConfig):
        """Send batched alerts"""
        if not self._alert_buffer:
            return

        try:
            # Deduplicate alerts if configured
            if alert_config.alert_deduplication:
                unique_matches = self._deduplicate_matches(
                    self._alert_buffer, alert_config.max_alerts_per_interval
                )
            else:
                unique_matches = self._alert_buffer[
                    : alert_config.max_alerts_per_interval
                ]

            await self._send_alerts(unique_matches, alert_config)
            self._alert_buffer.clear()
            self._last_alert_time = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Error sending batch alerts: {e}")

    def _deduplicate_matches(
        self, matches: list[SecurityMatch], max_matches: int
    ) -> list[SecurityMatch]:
        """Deduplicate matches while preserving order"""
        seen = set()
        unique_matches = []

        for match in matches:
            key = (match.content_type, match.file_path)
            if key not in seen and len(unique_matches) < max_matches:
                seen.add(key)
                unique_matches.append(match)

        return unique_matches

    async def _send_alerts(
        self, matches: list[SecurityMatch], alert_config: AlertConfig
    ):
        """Send security alerts"""
        if not alert_config.webhook_url:
            return

        try:
            async with aiohttp.ClientSession() as session:
                alerts = [
                    {
                        "type": match.content_type.value,
                        "file": match.file_path,
                        "line": match.line_number,
                        "context": match.context,
                        "severity": alert_config.severity.value,
                        "timestamp": match.timestamp.isoformat(),
                    }
                    for match in matches
                ]

                payload = {
                    "alerts": alerts,
                    "total_matches": len(matches),
                    "severity": alert_config.severity.value,
                }

                async with session.post(
                    alert_config.webhook_url, json=payload, timeout=30
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send alerts: {response.status}")
                    else:
                        self.metrics.alerts_sent.labels(
                            alert_config.severity.value
                        ).inc(len(matches))

        except Exception as e:
            logger.error(f"Error sending alerts: {e}")

    def _update_metrics(self, matches: list[SecurityMatch], action: ContentAction):
        """Update Prometheus metrics"""
        for match in matches:
            self.metrics.sensitive_content_matches.labels(
                match.content_type.value, action.value
            ).inc()
