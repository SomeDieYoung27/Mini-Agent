from typing import Any
from pydantic import BaseModel

class ToolResult(BaseModel):

    success: bool
    content: str =""
    error : str | None = None


class Tool:

    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @property

    def description(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    async def execute(self,args, **kwargs) -> ToolResult:
        raise NotImplementedError("Subclasses must implement this method")



    def to_schema(self) -> dict[str,Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai_schema(self) -> dict[str,Any]:

        return {
            "type" : "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }