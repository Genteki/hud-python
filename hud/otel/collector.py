"""Global span collector for building in-memory traces.

This module provides a way to collect spans during execution
and retrieve them as a Trace object, enabling replay functionality
without modifying agent code.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from hud.types import Trace, TraceStep


# Global storage for collected spans by task_run_id
_TRACE_STORAGE: Dict[str, TraceCollector] = {}
_LOCK = threading.Lock()

# Context variable to track if collection is enabled
_collecting_enabled: ContextVar[bool] = ContextVar("collecting_enabled", default=False)


class TraceCollector:
    """Collects spans for a single task run."""
    
    def __init__(self, task_run_id: str):
        self.task_run_id = task_run_id
        self.spans: List[ReadableSpan] = []
        self._lock = threading.Lock()
    
    def add_span(self, span: ReadableSpan) -> None:
        """Thread-safe span addition."""
        with self._lock:
            self.spans.append(span)
    
    def to_trace(self) -> Trace:
        """Convert collected spans to a Trace object."""
        from .exporters import _span_to_dict
        from .hud_models import HudSpan
        
        trace = Trace()
        
        # Convert spans to TraceSteps
        for span in self.spans:
            try:
                # Use the same conversion logic as the exporter
                span_dict = _span_to_dict(span)
                hud_span = HudSpan.model_validate(span_dict)
                
                # The attributes field is already a TraceStep
                step = hud_span.attributes
                # Add timing from the span itself
                step.start_timestamp = hud_span.start_time
                step.end_timestamp = hud_span.end_time
                trace.append(step)
                
            except Exception as e:
                # Log but don't fail the whole trace
                import logging
                logging.getLogger(__name__).debug(f"Failed to convert span: {e}")
        
        return trace


class CollectingSpanExporter(SpanExporter):
    """A span exporter that collects spans in memory for replay."""
    
    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        """Collect spans if collection is enabled."""
        if not _collecting_enabled.get():
            return SpanExportResult.SUCCESS
            
        for span in spans:
            # Extract task_run_id from span
            task_run_id = span.attributes.get("hud.task_run_id") if span.attributes else None
            if not task_run_id:
                continue
                
            # Get or create collector
            with _LOCK:
                if task_run_id not in _TRACE_STORAGE:
                    _TRACE_STORAGE[task_run_id] = TraceCollector(task_run_id)
                collector = _TRACE_STORAGE[task_run_id]
            
            # Add span
            collector.add_span(span)
            
        return SpanExportResult.SUCCESS
    
    def shutdown(self) -> None:
        """Clean up resources."""
        with _LOCK:
            _TRACE_STORAGE.clear()


def enable_trace_collection(enabled: bool = True) -> None:
    """Enable or disable in-memory trace collection."""
    _collecting_enabled.set(enabled)


def get_trace(task_run_id: str) -> Optional[Trace]:
    """Retrieve collected trace for a task run ID.
    
    Returns None if no trace was collected or collection was disabled.
    """
    with _LOCK:
        collector = _TRACE_STORAGE.get(task_run_id)
        if collector:
            return collector.to_trace()
    return None


def clear_trace(task_run_id: str) -> None:
    """Clear collected trace for a task run ID."""
    with _LOCK:
        _TRACE_STORAGE.pop(task_run_id, None)


def install_collector() -> None:
    """Install the collecting span exporter.
    
    This should be called after configure_telemetry().
    """
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        exporter = CollectingSpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
