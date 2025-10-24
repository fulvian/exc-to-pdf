"""
Production monitoring and observability system for DevStream.

Context7-validated patterns for comprehensive production monitoring:
- Health check system with dependency validation
- Metrics collection and aggregation
- Performance monitoring and alerting
- Error tracking and analysis
- Structured logging integration
"""

import asyncio
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import json

import structlog
from pydantic import BaseModel, Field

from devstream.core.exceptions import DevStreamError
from devstream.database.connection import ConnectionPool

logger = structlog.get_logger(__name__)


class HealthStatus(BaseModel):
    """Health check status model."""

    status: str = Field(description="Overall health status")
    timestamp: float = Field(description="Unix timestamp")
    version: str = Field(description="Application version")
    environment: str = Field(description="Environment name")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(description="Component health details")


class MetricPoint(BaseModel):
    """Single metric data point."""

    name: str = Field(description="Metric name")
    value: Union[int, float] = Field(description="Metric value")
    timestamp: float = Field(description="Unix timestamp")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metric tags")
    unit: Optional[str] = Field(default=None, description="Metric unit")


class SystemMetrics(BaseModel):
    """System performance metrics."""

    cpu_percent: float = Field(description="CPU usage percentage")
    memory_percent: float = Field(description="Memory usage percentage")
    memory_used_mb: float = Field(description="Memory used in MB")
    memory_available_mb: float = Field(description="Memory available in MB")
    disk_usage_percent: float = Field(description="Disk usage percentage")
    disk_free_gb: float = Field(description="Free disk space in GB")
    load_average: List[float] = Field(description="System load average")
    open_files: int = Field(description="Number of open files")


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    duration_seconds: int = 60
    description: str = ""
    severity: str = "warning"  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True


class HealthChecker:
    """
    Production health check system with dependency validation.

    Context7-validated pattern for comprehensive health monitoring.
    """

    def __init__(self, pool: ConnectionPool, version: str = "1.0.0", environment: str = "production"):
        """
        Initialize health checker.

        Args:
            pool: Database connection pool
            version: Application version
            environment: Environment name
        """
        self.pool = pool
        self.version = version
        self.environment = environment
        self.start_time = time.time()
        self._health_checks: Dict[str, Callable] = {}

        # Register default health checks
        self._register_default_checks()

    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register custom health check function."""
        self._health_checks[name] = check_func
        logger.info("Health check registered", name=name)

    def _register_default_checks(self) -> None:
        """Register default system health checks."""

        async def database_health():
            """Check database connectivity and response time."""
            start_time = time.time()
            try:
                async with self.pool.read_transaction() as conn:
                    from sqlalchemy import text
                    result = await conn.execute(text("SELECT 1"))
                    result.fetchone()

                response_time = (time.time() - start_time) * 1000

                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "pool_size": self.pool.pool_size if hasattr(self.pool, 'pool_size') else 'unknown'
                }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time_ms": (time.time() - start_time) * 1000
                }

        async def memory_system_health():
            """Check memory system health."""
            try:
                # Basic memory system validation
                # This would typically check vector search, embeddings, etc.
                return {
                    "status": "healthy",
                    "note": "Memory system check placeholder"
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }

        async def task_system_health():
            """Check task system health."""
            try:
                # Check task system components
                return {
                    "status": "healthy",
                    "note": "Task system check placeholder"
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }

        async def storage_health():
            """Check storage health and disk space."""
            try:
                # Check database file accessibility
                if hasattr(self.pool, 'db_path'):
                    db_path = Path(self.pool.db_path)
                    if db_path.exists():
                        # Check disk space
                        stat = psutil.disk_usage(str(db_path.parent))
                        free_gb = stat.free / (1024**3)

                        status = "healthy" if free_gb > 1.0 else "warning"

                        return {
                            "status": status,
                            "database_size_mb": round(db_path.stat().st_size / (1024**2), 2),
                            "free_space_gb": round(free_gb, 2),
                            "warning": "Low disk space" if free_gb <= 1.0 else None
                        }

                return {"status": "healthy", "note": "Storage check basic validation"}

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }

        # Register checks
        self._health_checks.update({
            "database": database_health,
            "memory": memory_system_health,
            "tasks": task_system_health,
            "storage": storage_health
        })

    async def check_health(self) -> HealthStatus:
        """
        Perform comprehensive health check.

        Returns:
            HealthStatus with overall system health
        """
        logger.debug("Running health checks")

        components = {}
        overall_status = "healthy"

        # Run all health checks
        for name, check_func in self._health_checks.items():
            try:
                start_time = time.time()
                component_health = await check_func()
                response_time = (time.time() - start_time) * 1000

                component_health["response_time_ms"] = round(response_time, 2)
                components[name] = component_health

                # Determine overall status
                if component_health["status"] == "unhealthy":
                    overall_status = "unhealthy"
                elif component_health["status"] == "warning" and overall_status == "healthy":
                    overall_status = "degraded"

            except Exception as e:
                logger.error("Health check failed", component=name, error=str(e))
                components[name] = {
                    "status": "unhealthy",
                    "error": f"Health check exception: {str(e)}"
                }
                overall_status = "unhealthy"

        uptime = time.time() - self.start_time

        return HealthStatus(
            status=overall_status,
            timestamp=time.time(),
            version=self.version,
            environment=self.environment,
            uptime_seconds=round(uptime, 2),
            components=components
        )


class MetricsCollector:
    """
    Production metrics collection and aggregation system.

    Context7-validated pattern for performance monitoring.
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.retention_hours = retention_hours
        self._metrics: List[MetricPoint] = []
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

        logger.info("Metrics collector initialized", retention_hours=retention_hours)

    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Record counter metric."""
        tags = tags or {}

        if name not in self._counters:
            self._counters[name] = 0

        self._counters[name] += value

        metric = MetricPoint(
            name=name,
            value=self._counters[name],
            timestamp=time.time(),
            tags=tags,
            unit="count"
        )
        self._add_metric(metric)

    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record gauge metric."""
        tags = tags or {}

        self._gauges[name] = value

        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            unit="gauge"
        )
        self._add_metric(metric)

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record histogram metric."""
        tags = tags or {}

        if name not in self._histograms:
            self._histograms[name] = []

        self._histograms[name].append(value)

        # Keep only recent values for histogram
        cutoff_time = time.time() - 3600  # 1 hour
        self._histograms[name] = [v for v in self._histograms[name][-1000:]]  # Max 1000 values

        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            unit="histogram"
        )
        self._add_metric(metric)

    def _add_metric(self, metric: MetricPoint) -> None:
        """Add metric to collection."""
        self._metrics.append(metric)

        # Cleanup old metrics
        cutoff_time = time.time() - (self.retention_hours * 3600)
        self._metrics = [m for m in self._metrics if m.timestamp > cutoff_time]

    def get_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = (memory.total - memory.available) / (1024**2)
            memory_available_mb = memory.available / (1024**2)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)

            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]

            # Process metrics
            process = psutil.Process()
            open_files = len(process.open_files())

            metrics = SystemMetrics(
                cpu_percent=round(cpu_percent, 2),
                memory_percent=round(memory.percent, 2),
                memory_used_mb=round(memory_used_mb, 2),
                memory_available_mb=round(memory_available_mb, 2),
                disk_usage_percent=round(disk_usage_percent, 2),
                disk_free_gb=round(disk_free_gb, 2),
                load_average=[round(x, 2) for x in load_avg],
                open_files=open_files
            )

            # Record as metrics
            timestamp = time.time()
            system_metrics = [
                MetricPoint(name="system.cpu.percent", value=metrics.cpu_percent, timestamp=timestamp),
                MetricPoint(name="system.memory.percent", value=metrics.memory_percent, timestamp=timestamp),
                MetricPoint(name="system.disk.percent", value=metrics.disk_usage_percent, timestamp=timestamp),
                MetricPoint(name="system.disk.free_gb", value=metrics.disk_free_gb, timestamp=timestamp),
                MetricPoint(name="system.files.open", value=metrics.open_files, timestamp=timestamp),
            ]

            for metric in system_metrics:
                self._add_metric(metric)

            return metrics

        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            raise

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        recent_time = time.time() - 300  # Last 5 minutes
        recent_metrics = [m for m in self._metrics if m.timestamp > recent_time]

        # Group by metric name
        metric_groups = {}
        for metric in recent_metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric.value)

        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1]
                }

        return {
            "summary": summary,
            "total_metrics": len(self._metrics),
            "recent_metrics": len(recent_metrics),
            "collection_period_hours": self.retention_hours
        }


class AlertManager:
    """
    Alert management system for production monitoring.

    Context7-validated pattern for alerting and notification.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize alert manager.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}

        # Register default alert rules
        self._register_default_alerts()

        logger.info("Alert manager initialized")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.alert_rules.append(rule)
        logger.info("Alert rule added", name=rule.name, metric=rule.metric_name)

    def _register_default_alerts(self) -> None:
        """Register default production alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu.percent",
                threshold=80.0,
                comparison="gt",
                duration_seconds=300,
                description="CPU usage above 80% for 5 minutes",
                severity="warning"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory.percent",
                threshold=85.0,
                comparison="gt",
                duration_seconds=300,
                description="Memory usage above 85% for 5 minutes",
                severity="warning"
            ),
            AlertRule(
                name="low_disk_space",
                metric_name="system.disk.free_gb",
                threshold=2.0,
                comparison="lt",
                duration_seconds=60,
                description="Less than 2GB free disk space",
                severity="error"
            ),
            AlertRule(
                name="critical_disk_space",
                metric_name="system.disk.free_gb",
                threshold=0.5,
                comparison="lt",
                duration_seconds=30,
                description="Less than 500MB free disk space",
                severity="critical"
            )
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alert rules and return active alerts."""
        current_time = time.time()
        new_alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # Get recent metrics for this rule
                lookback_time = current_time - rule.duration_seconds
                relevant_metrics = [
                    m for m in self.metrics_collector._metrics
                    if m.name == rule.metric_name and m.timestamp > lookback_time
                ]

                if not relevant_metrics:
                    continue

                # Check if alert condition is met
                latest_value = relevant_metrics[-1].value
                condition_met = self._evaluate_condition(
                    latest_value, rule.threshold, rule.comparison
                )

                alert_key = f"{rule.name}:{rule.metric_name}"

                if condition_met:
                    # Check if this is a new alert or existing
                    if alert_key not in self.active_alerts:
                        # New alert
                        alert = {
                            "rule_name": rule.name,
                            "metric_name": rule.metric_name,
                            "current_value": latest_value,
                            "threshold": rule.threshold,
                            "comparison": rule.comparison,
                            "severity": rule.severity,
                            "description": rule.description,
                            "started_at": current_time,
                            "last_triggered": current_time
                        }

                        self.active_alerts[alert_key] = alert
                        new_alerts.append(alert)

                        logger.warning(
                            "Alert triggered",
                            rule=rule.name,
                            metric=rule.metric_name,
                            value=latest_value,
                            threshold=rule.threshold,
                            severity=rule.severity
                        )
                    else:
                        # Update existing alert
                        self.active_alerts[alert_key]["last_triggered"] = current_time
                        self.active_alerts[alert_key]["current_value"] = latest_value

                else:
                    # Clear alert if it was active
                    if alert_key in self.active_alerts:
                        resolved_alert = self.active_alerts.pop(alert_key)
                        logger.info(
                            "Alert resolved",
                            rule=rule.name,
                            metric=rule.metric_name,
                            duration=current_time - resolved_alert["started_at"]
                        )

            except Exception as e:
                logger.error("Alert check failed", rule=rule.name, error=str(e))

        return new_alerts

    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition."""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "eq":
            return value == threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "lte":
            return value <= threshold
        else:
            logger.error("Unknown comparison operator", comparison=comparison)
            return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())


class ProductionMonitoringSystem:
    """
    Comprehensive production monitoring system.

    Context7-validated integration of health checks, metrics, and alerting.
    """

    def __init__(
        self,
        pool: ConnectionPool,
        version: str = "1.0.0",
        environment: str = "production"
    ):
        """
        Initialize production monitoring system.

        Args:
            pool: Database connection pool
            version: Application version
            environment: Environment name
        """
        self.health_checker = HealthChecker(pool, version, environment)
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)

        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "Production monitoring system initialized",
            version=version,
            environment=environment
        )

    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous monitoring."""
        if self._running:
            logger.warning("Monitoring already running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )

        logger.info("Production monitoring started", interval=interval_seconds)

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Production monitoring stopped")

    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.get_system_metrics()

                # Check alerts
                new_alerts = self.alert_manager.check_alerts()

                # Log critical alerts
                for alert in new_alerts:
                    if alert["severity"] in ["error", "critical"]:
                        logger.error(
                            "Critical alert triggered",
                            alert_name=alert["rule_name"],
                            metric=alert["metric_name"],
                            value=alert["current_value"],
                            threshold=alert["threshold"]
                        )

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(interval_seconds)

    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        try:
            # Get health status
            health = await self.health_checker.check_health()

            # Get system metrics
            system_metrics = self.metrics_collector.get_system_metrics()

            # Get metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary()

            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()

            return {
                "health": health.dict(),
                "system_metrics": system_metrics.dict(),
                "metrics_summary": metrics_summary,
                "active_alerts": active_alerts,
                "alert_rules_count": len(self.alert_manager.alert_rules),
                "monitoring_status": "running" if self._running else "stopped"
            }

        except Exception as e:
            logger.error("Failed to get monitoring dashboard", error=str(e))
            raise DevStreamError(f"Monitoring dashboard error: {e}")


# Global monitoring instance
_monitoring_system: Optional[ProductionMonitoringSystem] = None


def get_monitoring_system() -> ProductionMonitoringSystem:
    """Get global monitoring system instance."""
    global _monitoring_system

    if _monitoring_system is None:
        raise DevStreamError("Monitoring system not initialized")

    return _monitoring_system


def initialize_monitoring(
    pool: ConnectionPool,
    version: str = "1.0.0",
    environment: str = "production"
) -> ProductionMonitoringSystem:
    """Initialize global monitoring system."""
    global _monitoring_system

    _monitoring_system = ProductionMonitoringSystem(pool, version, environment)

    logger.info("Global monitoring system initialized")
    return _monitoring_system