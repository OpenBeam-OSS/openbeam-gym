"""openbeam-gym: Gymnasium environments over OpenBeam's game MCP tools."""

from .env import OpenBeamRobotEnv, SceneConfig
from .mcp import CloudAccount, CloudGame, GameClient, LocalGame, McpError, ToolError
from .tasks import Pick, Reach, Task

__all__ = ["OpenBeamRobotEnv", "SceneConfig", "CloudAccount", "CloudGame", "GameClient", "LocalGame",
           "McpError", "ToolError", "Pick", "Reach", "Task"]
__version__ = "0.1.0"
