from abc import ABC, abstractmethod

from evalharness.models import AgentRunOutput, TaskCase


class AgentAdapter(ABC):
    agent_name: str

    @abstractmethod
    async def run_task(self, task: TaskCase) -> AgentRunOutput:
        pass
