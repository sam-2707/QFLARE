"""
Distributed Tracing for QFLARE Federated Learning

This module provides comprehensive distributed tracing capabilities:
- Request tracing across federated learning operations
- Span tracking for training, aggregation, and communication
- Performance monitoring and bottleneck identification
- Cross-service correlation and dependency mapping
- Integration with OpenTelemetry standard
"""

import time
import logging
import threading
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Types of spans in QFLARE tracing."""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer" 
    CONSUMER = "consumer"
    INTERNAL = "internal"


class SpanStatus(Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Context information for a span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    trace_flags: int = 1  # Sampled by default


@dataclass
class Span:
    """Represents a single span in a trace."""
    context: SpanContext
    operation_name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: Optional[float] = None
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span."""
        self.end_time = datetime.utcnow()
        self.status = status
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span."""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception):
        """Mark the span as having an error."""
        self.status = SpanStatus.ERROR
        self.set_tag('error', True)
        self.set_tag('error.type', type(error).__name__)
        self.set_tag('error.message', str(error))
        self.log(f"Error: {str(error)}", level="error")


@dataclass
class Trace:
    """Represents a complete trace containing multiple spans."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    service_name: str = "qflare"
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans.append(span)
        
        # Update trace timing
        if not self.start_time or span.start_time < self.start_time:
            self.start_time = span.start_time
        
        if span.end_time:
            if not self.end_time or span.end_time > self.end_time:
                self.end_time = span.end_time
        
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def get_root_span(self) -> Optional[Span]:
        """Get the root span of the trace."""
        for span in self.spans:
            if span.context.parent_span_id is None:
                return span
        return None
    
    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get a span by its ID."""
        for span in self.spans:
            if span.context.span_id == span_id:
                return span
        return None


class QFLARETracer:
    """Distributed tracer for QFLARE federated learning operations."""
    
    def __init__(self, service_name: str = "qflare"):
        """Initialize the tracer."""
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
        self.active_spans: Dict[str, Span] = {}
        self.completed_traces: Dict[str, Trace] = {}
        self.span_processors: List[Callable[[Span], None]] = []
        self.trace_exporters: List[Callable[[Trace], None]] = []
        self._local = threading.local()
        
        # Configuration
        self.sampling_rate = 1.0  # Sample all traces by default
        self.max_traces_in_memory = 1000
        self.trace_retention_hours = 24
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def start_span(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL,
                   parent_context: Optional[SpanContext] = None,
                   tags: Dict[str, Any] = None) -> Span:
        """Start a new span."""
        # Generate IDs
        trace_id = parent_context.trace_id if parent_context else self._generate_trace_id()
        span_id = self._generate_span_id()
        parent_span_id = parent_context.span_id if parent_context else None
        
        # Create span context
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=parent_context.baggage.copy() if parent_context else {}
        )
        
        # Create span
        span = Span(
            context=context,
            operation_name=operation_name,
            kind=kind,
            start_time=datetime.utcnow(),
            tags=tags or {}
        )
        
        # Add default tags
        span.set_tag('service.name', self.service_name)
        span.set_tag('span.kind', kind.value)
        
        # Store active span
        self.active_spans[span_id] = span
        
        # Set as current span
        self._set_current_span(span)
        
        self.logger.debug(f"Started span: {operation_name} (trace_id={trace_id}, span_id={span_id})")
        
        return span
    
    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """Finish a span."""
        span.finish(status)
        
        # Remove from active spans
        if span.context.span_id in self.active_spans:
            del self.active_spans[span.context.span_id]
        
        # Process span
        for processor in self.span_processors:
            try:
                processor(span)
            except Exception as e:
                self.logger.error(f"Error in span processor: {e}")
        
        # Add to trace
        trace = self._get_or_create_trace(span.context.trace_id)
        trace.add_span(span)
        
        # Check if trace is complete and export
        if self._is_trace_complete(trace):
            self._export_trace(trace)
        
        self.logger.debug(f"Finished span: {span.operation_name} ({span.duration_ms:.2f}ms)")
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return getattr(self._local, 'current_span', None)
    
    def get_current_context(self) -> Optional[SpanContext]:
        """Get the current span context."""
        current_span = self.get_current_span()
        return current_span.context if current_span else None
    
    def inject_context(self, context: SpanContext, carrier: Dict[str, str]):
        """Inject span context into a carrier (e.g., HTTP headers)."""
        carrier['x-trace-id'] = context.trace_id
        carrier['x-span-id'] = context.span_id
        if context.parent_span_id:
            carrier['x-parent-span-id'] = context.parent_span_id
        
        # Add baggage
        for key, value in context.baggage.items():
            carrier[f'x-baggage-{key}'] = value
    
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from a carrier."""
        trace_id = carrier.get('x-trace-id')
        span_id = carrier.get('x-span-id')
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = carrier.get('x-parent-span-id')
        
        # Extract baggage
        baggage = {}
        for key, value in carrier.items():
            if key.startswith('x-baggage-'):
                baggage_key = key[10:]  # Remove 'x-baggage-' prefix
                baggage[baggage_key] = value
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )
    
    @contextmanager
    def trace(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL,
              tags: Dict[str, Any] = None):
        """Context manager for tracing an operation."""
        parent_context = self.get_current_context()
        span = self.start_span(operation_name, kind, parent_context, tags)
        
        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
    
    def add_span_processor(self, processor: Callable[[Span], None]):
        """Add a span processor."""
        self.span_processors.append(processor)
    
    def add_trace_exporter(self, exporter: Callable[[Trace], None]):
        """Add a trace exporter."""
        self.trace_exporters.append(exporter)
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self.completed_traces.get(trace_id)
    
    def get_active_traces(self) -> List[Trace]:
        """Get all active traces."""
        active_trace_ids = set(span.context.trace_id for span in self.active_spans.values())
        return [self._get_or_create_trace(trace_id) for trace_id in active_trace_ids]
    
    def get_tracing_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        return {
            'active_spans': len(self.active_spans),
            'completed_traces': len(self.completed_traces),
            'span_processors': len(self.span_processors),
            'trace_exporters': len(self.trace_exporters),
            'sampling_rate': self.sampling_rate,
            'service_name': self.service_name
        }
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex
    
    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex[:16]
    
    def _set_current_span(self, span: Span):
        """Set the current span in thread local storage."""
        self._local.current_span = span
    
    def _get_or_create_trace(self, trace_id: str) -> Trace:
        """Get or create a trace by ID."""
        if trace_id not in self.completed_traces:
            self.completed_traces[trace_id] = Trace(
                trace_id=trace_id,
                service_name=self.service_name
            )
        return self.completed_traces[trace_id]
    
    def _is_trace_complete(self, trace: Trace) -> bool:
        """Check if a trace is complete (no active spans)."""
        trace_active_spans = [
            span for span in self.active_spans.values()
            if span.context.trace_id == trace.trace_id
        ]
        return len(trace_active_spans) == 0
    
    def _export_trace(self, trace: Trace):
        """Export a completed trace."""
        for exporter in self.trace_exporters:
            try:
                exporter(trace)
            except Exception as e:
                self.logger.error(f"Error in trace exporter: {e}")
    
    def _start_cleanup_thread(self):
        """Start background thread for cleanup."""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_old_traces()
                    time.sleep(3600)  # Cleanup every hour
                except Exception as e:
                    self.logger.error(f"Error in trace cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_traces(self):
        """Clean up old traces from memory."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.trace_retention_hours)
        
        traces_to_remove = []
        for trace_id, trace in self.completed_traces.items():
            if trace.end_time and trace.end_time < cutoff_time:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.completed_traces[trace_id]
        
        # Also enforce max traces limit
        if len(self.completed_traces) > self.max_traces_in_memory:
            # Remove oldest traces
            sorted_traces = sorted(
                self.completed_traces.items(),
                key=lambda x: x[1].end_time or datetime.min
            )
            
            traces_to_remove = sorted_traces[:len(self.completed_traces) - self.max_traces_in_memory]
            for trace_id, _ in traces_to_remove:
                del self.completed_traces[trace_id]
        
        if traces_to_remove:
            self.logger.debug(f"Cleaned up {len(traces_to_remove)} old traces")


class FederatedLearningTracer:
    """Specialized tracer for federated learning operations."""
    
    def __init__(self, base_tracer: QFLARETracer):
        """Initialize FL tracer with base tracer."""
        self.tracer = base_tracer
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def trace_fl_round(self, round_id: str, algorithm: str = "fedavg"):
        """Trace a complete federated learning round."""
        with self.tracer.trace(
            f"fl_round_{round_id}",
            SpanKind.SERVER,
            tags={
                'fl.round_id': round_id,
                'fl.algorithm': algorithm,
                'fl.operation': 'round'
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_device_training(self, device_id: str, model_type: str = "cnn"):
        """Trace device training operation."""
        with self.tracer.trace(
            f"device_training_{device_id}",
            SpanKind.CLIENT,
            tags={
                'fl.device_id': device_id,
                'fl.model_type': model_type,
                'fl.operation': 'training'
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_model_aggregation(self, num_models: int, aggregation_method: str = "fedavg"):
        """Trace model aggregation operation."""
        with self.tracer.trace(
            "model_aggregation",
            SpanKind.INTERNAL,
            tags={
                'fl.num_models': num_models,
                'fl.aggregation_method': aggregation_method,
                'fl.operation': 'aggregation'
            }
        ) as span:
            yield span
    
    @contextmanager
    def trace_secure_communication(self, operation: str, device_id: str):
        """Trace secure communication operation."""
        with self.tracer.trace(
            f"secure_comm_{operation}",
            SpanKind.CLIENT,
            tags={
                'security.operation': operation,
                'security.device_id': device_id,
                'fl.operation': 'communication'
            }
        ) as span:
            yield span


# Trace exporters
class ConsoleTraceExporter:
    """Export traces to console/logs."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def __call__(self, trace: Trace):
        """Export a trace to console."""
        self.logger.info(f"Trace completed: {trace.trace_id} ({trace.duration_ms:.2f}ms)")
        
        for span in trace.spans:
            indent = "  " * self._get_span_depth(span, trace)
            self.logger.info(
                f"{indent}├─ {span.operation_name} ({span.duration_ms:.2f}ms) "
                f"[{span.status.value}]"
            )
    
    def _get_span_depth(self, span: Span, trace: Trace) -> int:
        """Calculate the depth of a span in the trace hierarchy."""
        depth = 0
        current_span = span
        
        while current_span.context.parent_span_id:
            parent = trace.get_span_by_id(current_span.context.parent_span_id)
            if not parent:
                break
            current_span = parent
            depth += 1
        
        return depth


class JSONTraceExporter:
    """Export traces to JSON files."""
    
    def __init__(self, output_dir: str = "/tmp/qflare_traces"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def __call__(self, trace: Trace):
        """Export a trace to JSON file."""
        trace_data = {
            'trace_id': trace.trace_id,
            'service_name': trace.service_name,
            'start_time': trace.start_time.isoformat() if trace.start_time else None,
            'end_time': trace.end_time.isoformat() if trace.end_time else None,
            'duration_ms': trace.duration_ms,
            'spans': []
        }
        
        for span in trace.spans:
            span_data = {
                'span_id': span.context.span_id,
                'parent_span_id': span.context.parent_span_id,
                'operation_name': span.operation_name,
                'kind': span.kind.value,
                'start_time': span.start_time.isoformat(),
                'end_time': span.end_time.isoformat() if span.end_time else None,
                'duration_ms': span.duration_ms,
                'status': span.status.value,
                'tags': span.tags,
                'logs': span.logs
            }
            trace_data['spans'].append(span_data)
        
        # Write to file
        import os
        filename = f"{trace.trace_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2)


# Global tracer instance
_tracer: Optional[QFLARETracer] = None
_fl_tracer: Optional[FederatedLearningTracer] = None


def get_tracer() -> QFLARETracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = QFLARETracer()
        # Add default exporters
        _tracer.add_trace_exporter(ConsoleTraceExporter())
    return _tracer


def get_fl_tracer() -> FederatedLearningTracer:
    """Get the federated learning tracer."""
    global _fl_tracer
    if _fl_tracer is None:
        _fl_tracer = FederatedLearningTracer(get_tracer())
    return _fl_tracer


def initialize_tracing(service_name: str = "qflare") -> QFLARETracer:
    """Initialize the global tracer."""
    global _tracer, _fl_tracer
    _tracer = QFLARETracer(service_name)
    _fl_tracer = FederatedLearningTracer(_tracer)
    
    # Add default exporters
    _tracer.add_trace_exporter(ConsoleTraceExporter())
    
    return _tracer


def shutdown_tracing():
    """Shutdown tracing."""
    global _tracer, _fl_tracer
    _tracer = None
    _fl_tracer = None