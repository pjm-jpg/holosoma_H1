from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from typing_extensions import Self

from holosoma.config_types.frequency import DecimationLike, is_frequency_string, resolve_decimation


class Phase(str, Enum):
    """Lifecycle points emitted by the active simulator loop.

    ``payload`` names the positional args ``emit`` forwards to callbacks. ``periodic`` marks phases
    that fire on a fixed clock (per-substep or per-frame ticks) — only these accept a frequency-string
    rate (``"30Hz"``) for :meth:`HookRegistry.add`'s ``every``. Episode events accept an int decimation
    ("every Nth event") but reject frequency strings (no clock to resolve against); CLOSE always
    fires once and rejects any decimation.

    ``FRAME_*`` bracket the outer tick (once per frame); ``*_STEP`` bracket each physics substep. The
    names are engine-agnostic — they don't assume what drives the outer tick (a policy, a bridge, ...).
    """

    payload: tuple[str, ...]
    periodic: bool

    FRAME_BEGIN = ("frame_begin", (), True)
    PRE_STEP = ("pre_step", (), True)
    POST_STEP = ("post_step", (), True)
    FRAME_END = ("frame_end", (), True)
    EPISODE_END = ("episode_end", ("env_id",), False)
    EPISODE_START = ("episode_start", ("env_id",), False)
    CLOSE = ("close", (), False)

    def __new__(cls, value: str, payload: tuple[str, ...], periodic: bool) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.payload = payload
        obj.periodic = periodic
        return obj


class HookRegistryError(RuntimeError):
    """Raised when the hook registry is misused."""


class HookCloseError(RuntimeError):
    """Raised after close hooks run if one or more hooks failed."""

    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        self.failures = failures
        details = ", ".join(f"{name}: {exc!r}" for name, exc in failures)
        super().__init__(f"{len(failures)} close hook(s) failed: {details}")


@dataclass
class _HookRecord:
    phase: Phase
    callback: Callable[..., Any]
    name: str
    every: int = 1
    """Resolved emission decimation: the callback runs once per ``every`` emissions of its phase."""
    _counter: int = 0
    """Emissions counted toward the next fire; only advances while enabled (disabled hooks leave the snapshot)."""
    enabled: bool = True
    removed: bool = False

    def due(self) -> bool:
        """Advance the per-hook counter and report whether this emission should fire the callback."""
        self._counter += 1
        if self._counter >= self.every:
            self._counter = 0
            return True
        return False


class HookHandle:
    """Identity handle returned by hook registration."""

    def __init__(self, registry: HookRegistry, record: _HookRecord) -> None:
        self._registry = registry
        self._record = record

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def phase(self) -> Phase:
        return self._record.phase

    @property
    def enabled(self) -> bool:
        return self._record.enabled and not self._record.removed

    @property
    def every(self) -> int:
        """Current resolved emission decimation (see :meth:`set_every`)."""
        return self._record.every

    def enable(self) -> None:
        self._registry._enable(self._record)

    def disable(self) -> None:
        self._registry._disable(self._record)

    def set_every(self, every: DecimationLike) -> None:
        """Change this hook's cadence live (int decimation or frequency string), resetting its counter.

        Same rules as ``every`` at registration. Safe to call while other phases emit, but not from
        inside this hook's own phase (mutating a phase mid-emit raises)."""
        self._registry._set_every(self._record, every)

    def remove(self) -> None:
        self._registry._remove(self._record)


class HookRegistry:
    """Small lifecycle hook registry with deterministic registration order."""

    def __init__(self, base_rates: Mapping[Phase, float] | None = None) -> None:
        self._records: dict[Phase, list[_HookRecord]] = {phase: [] for phase in Phase}
        self._snapshots: dict[Phase, tuple[_HookRecord, ...]] = dict.fromkeys(Phase, ())
        self._emitting: set[Phase] = set()
        self._closed = False
        self._base_rates: dict[Phase, float] = dict(base_rates or {})

    def add(
        self,
        phase: Phase,
        callback: Callable[..., Any],
        *,
        name: str | None = None,
        every: DecimationLike = 1,
    ) -> HookHandle:
        """Register a hook callback for a lifecycle phase.

        ``every`` sub-samples emissions: an int decimation runs the callback once per ``every``
        emissions; a frequency string (``"30Hz"``, ``">30Hz"``, ``"<30Hz"``) is resolved against the
        phase's base tick rate into that decimation. Frequency strings require a periodic phase and a
        known base rate; event phases (episode/close) accept only int decimations. Default ``1`` fires
        every emission.
        """
        self._assert_mutable(phase)
        record = _HookRecord(
            phase=phase,
            callback=callback,
            name=name or self._default_name(callback),
            every=self._resolve_every(phase, every, name),
        )
        self._records[phase].append(record)
        self._rebuild(phase)
        return HookHandle(self, record)

    def _resolve_every(self, phase: Phase, every: DecimationLike, name: str | None) -> int:
        """Turn a hook's ``every`` into an int decimation, resolving frequency strings vs the base rate."""
        field = f"hook {name or '<callback>'!r} on phase {phase.value!r} 'every'"
        if phase is Phase.CLOSE and every != 1:
            raise HookRegistryError(f"{field}: CLOSE fires once at teardown; decimating it would skip cleanup.")
        if is_frequency_string(every):
            if not phase.periodic:
                raise HookRegistryError(
                    f"{field}: phase {phase.value!r} is not periodic; use an int decimation, not {every!r}."
                )
            base_hz = self._base_rates.get(phase)
            if base_hz is None:
                raise HookRegistryError(
                    f"{field}: no base rate registered for periodic phase {phase.value!r}; "
                    f"cannot resolve frequency {every!r}. Pass base_rates to HookRegistry, or use an int."
                )
            return resolve_decimation(every, base_hz, field=field, log=True)
        return resolve_decimation(every, base_hz=1.0, field=field)

    def emit(self, phase: Phase, *args: Any) -> None:
        """Run each enabled hook due this emission, in registration order (CLOSE runs once, reversed)."""
        self._validate_payload(phase, args)

        if phase is Phase.CLOSE:
            if self._closed:
                return

            if phase in self._emitting:
                raise HookRegistryError("Recursive close hook emission is not supported")

            self._closed = True
            failures: list[tuple[str, Exception]] = []
            self._emitting.add(phase)
            try:
                for record in reversed(self._snapshots[phase]):
                    try:
                        record.callback()
                    except Exception as exc:  # noqa: PERF203
                        failures.append((record.name, exc))
            finally:
                self._emitting.remove(phase)

            if failures:
                raise HookCloseError(failures)
            return

        if phase in self._emitting:
            raise HookRegistryError(f"Recursive hook emission is not supported for phase {phase.value!r}")

        self._emitting.add(phase)
        try:
            for record in self._snapshots[phase]:
                if record.due():
                    record.callback(*args)
        finally:
            self._emitting.remove(phase)

    @staticmethod
    def _validate_payload(phase: Phase, args: tuple[Any, ...]) -> None:
        expected = phase.payload
        if len(args) == len(expected):
            return
        expected_text = ", ".join(expected) or "no arguments"
        raise HookRegistryError(f"Phase {phase.value!r} expects {expected_text}")

    def _enable(self, record: _HookRecord) -> None:
        if record.removed or record.enabled:
            return
        self._assert_mutable(record.phase)
        record.enabled = True
        self._rebuild(record.phase)

    def _disable(self, record: _HookRecord) -> None:
        if record.removed or not record.enabled:
            return
        self._assert_mutable(record.phase)
        record.enabled = False
        self._rebuild(record.phase)

    def _set_every(self, record: _HookRecord, every: DecimationLike) -> None:
        if record.removed:
            raise HookRegistryError(f"Cannot set cadence on removed hook {record.name!r}.")
        self._assert_mutable(record.phase)
        record.every = self._resolve_every(record.phase, every, record.name)
        record._counter = 0

    def _remove(self, record: _HookRecord) -> None:
        if record.removed:
            return
        self._assert_mutable(record.phase)
        record.enabled = False
        record.removed = True
        self._rebuild(record.phase)

    def _rebuild(self, phase: Phase) -> None:
        self._snapshots[phase] = tuple(
            record for record in self._records[phase] if record.enabled and not record.removed
        )

    def _assert_mutable(self, phase: Phase) -> None:
        if phase in self._emitting:
            raise HookRegistryError(f"Cannot mutate hooks while emitting phase {phase.value!r}")

    @staticmethod
    def _default_name(callback: Callable[..., Any]) -> str:
        return getattr(callback, "__qualname__", repr(callback))
