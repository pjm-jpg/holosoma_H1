"""Config types for simulator plugins.

A *plugin* is a bundle of behavior — a set of lifecycle hooks plus whatever state and
side effects they need — that an extension attaches to a running simulator without
subclassing a backend. Plugins are built on the lifecycle hook system (``simulator.hooks``,
:class:`~holosoma.simulator.base_simulator.hooks.Phase`) and may depend on other simulator
contracts (the virtual gantry, the clock, etc.). Each plugin pairs:

- a ``PluginConfig`` subclass (the CLI-visible, serializable knobs), registered under a
  name in ``holosoma.config_values.plugin.PLUGIN_REGISTRY``, and
- a runtime plugin class, returned by the config's :meth:`PluginConfig.get_cls`. There is
  no base class to inherit: any class constructed as ``cls(cfg, simulator)`` that
  registers its hooks on ``simulator.hooks`` works (duck-typed).

The config is resolved on the CLI as a dynamic-dict field (see ``RunSimConfig.plugin``);
``BaseSimulator.__init__`` then instantiates each ``get_cls()`` against itself.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable

from holosoma.config_types.frequency import DecimationLike, validate_decimation_like


@dataclass(frozen=True)
class PluginConfig(abc.ABC):
    """Base config for a simulator plugin.

    Subclass with the plugin's parameters as dataclass fields and implement
    :meth:`get_cls` to point at the runtime plugin class. Register an instance in
    ``PLUGIN_REGISTRY`` so it is selectable as ``plugin.<key>:<variant>`` on the CLI.
    """

    @abc.abstractmethod
    def get_cls(self) -> Callable[..., Any]:
        """Return the runtime plugin class this config configures.

        The class is constructed as ``cls(cfg, simulator)`` and is expected to register
        its hooks on ``simulator.hooks`` in ``__init__`` — no base class required.
        Import it lazily inside this method so that registering the config preset does
        not pull the (possibly heavy) runtime module at import time.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class NoOpPluginConfig(PluginConfig):
    """A plugin that does nothing, registered as the ``none`` preset.

    Selecting ``plugin.<key>:none`` disables that slot: its runtime class registers no
    hooks, so it is a genuine no-op.
    """

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.builtin_plugins import NoOpPlugin

        return NoOpPlugin


@dataclass(frozen=True)
class ClockPublishPluginConfig(PluginConfig):
    """Publish sim time as a ROS2 ``rosgraph_msgs/msg/Clock`` topic.

    A ROS2 example plugin. rclpy is an optional dependency (``holosoma[ros2]``); this
    config stays import-safe without ROS because :meth:`get_cls` defers the impl import.
    """

    topic: str = "/clock"
    """Topic to publish the clock on (ROS2 ``use_sim_time`` consumers expect ``/clock``)."""

    node_name: str = "holosoma_clock"
    """ROS2 node name for the publisher."""

    publish_every: DecimationLike = 1
    """How often to publish, on the PHYSICS rate (the clock is read right after each physics
    step). Either a decimation int (publish every Nth physics step) or a frequency string
    (``"100Hz"``, ``">100Hz"``, ``"<100Hz"``) resolved at install time against ``fps``."""

    def __post_init__(self) -> None:
        validate_decimation_like(self.publish_every, field="publish_every")

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import ClockPublishPlugin

        return ClockPublishPlugin


@dataclass(frozen=True)
class GantryControlPluginConfig(PluginConfig):
    """Control the virtual gantry over ROS2 via three independent standard-message topics.

    Each of position / length / enabled is its own subscription, so publishing to one
    topic changes only that property (the others are left untouched). A ROS2 example
    plugin; rclpy is optional (``holosoma[ros2]``), imported lazily via :meth:`get_cls`.
    """

    position_topic: str = "/gantry/position"
    """``geometry_msgs/msg/Point`` — new gantry anchor point ``(x, y, z)`` in world frame."""

    length_topic: str = "/gantry/length"
    """``std_msgs/msg/Float64`` — new elastic-band rest length."""

    enabled_topic: str = "/gantry/enabled"
    """``std_msgs/msg/Bool`` — enable (True) or disable (False) the gantry."""

    node_name: str = "holosoma_gantry_control"
    """ROS2 node name for the subscriber."""

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import GantryControlPlugin

        return GantryControlPlugin
