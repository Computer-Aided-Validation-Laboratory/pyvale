import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True, frozen=True)
class ProgressEvent:
    """Lightweight progress update emitted during identification.
    
    Attributes:
        message: A human-readable message describing the progress event.
        kind: A string describing the type of progress event.
        elapsed_seconds: The elapsed time in seconds since the start of the identification (if applicable).
    """

    message: str
    kind: str | None = None
    elapsed_seconds: float | None = None


def emit_progress(
    progress_report_function,
    event: ProgressEvent,
) -> None:
    """Send a progress event to a progress-reporting function.
    
    Arguments:
        progress_report_function: 
            A callback function that takes a ProgressEvent as an argument. 
            If None, the event is ignored.
        event: The ProgressEvent to be emitted.
    """

    if progress_report_function is None:
        return

    progress_report_function(event)


@dataclass(slots=True)
class ConsoleProgressReporter:
    """Print progress messages to the console.
        
    Attributes:
        min_interval_seconds: The minimum time interval (in seconds) between printed messages.
        _last_printed_at: The timestamp of the last printed message (used internally).
    """

    min_interval_seconds: float = 1.0
    _last_printed_at: float = field(default=0.0, init=False, repr=False)

    def report(
        self,
        event: ProgressEvent,
    ) -> None:
        """Print the event message, unless this event should be skipped."""

        if self._should_skip(event):
            return

        print(f"[{datetime.now():%H:%M}] {event.message}", flush=True)
        self._last_printed_at = time.perf_counter()

    def _should_skip(
        self,
        event: ProgressEvent,
    ) -> bool:
        """Avoid printing very frequent progress messages too often."""

        # If the minimum interval is set to zero or negative, we do not skip any messages.
        if self.min_interval_seconds <= 0.0:
            return False

        # Define a set of event types that are considered "noisy" and may be skipped if they occur too frequently.
        noisy_event_types = {
            "slice_started",
            "slice_finished",
            "evaluation",
        }

        # If the event type is not in the noisy set, we do not skip it.
        if event.kind not in noisy_event_types:
            return False

        # Calculate the time since the last printed message. If it's less than the minimum interval, we skip printing this event.
        seconds_since_last_print = time.perf_counter() - self._last_printed_at
        if seconds_since_last_print < self.min_interval_seconds:
            return True

        return False
