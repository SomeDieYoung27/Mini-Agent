
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult
from .skill_loader import SkillLoader

class GetSkillTool(Tool):
       """Tool to get detailed information about a specific skill"""

       def __init__(self, skill_loader: SkillLoader):
        self.skill_loader = skill_loader

       @property
       def name(self) -> str:
        return "get_skill"

       @property
       def description(self) -> str:
        return "Get complete content and guidance for a specified skill, used for executing specific types of tasks"

       @property
       def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to get detailed information about",
                }
            },
            "required": ["skill_name"],
        }


        async def execute(self, skill_name: str) -> ToolResult:
        """Get detailed information about specified skill"""
        skill = self.skill_loader.get_skill(skill_name)

        if not skill:
            available = ", ".join(self.skill_loader.list_skills())
            return ToolResult(
                success=False,
                content="",
                error=f"Skill '{skill_name}' does not exist. Available skills: {available}",
            )

        # Return complete skill content
        result = skill.to_prompt()
        return ToolResult(success=True, content=result)



def create_skill_tools(  skills_dir: str = "./skills",) -> tuple[List[Tool], Optional[SkillLoader]]:

    loader = SkillLoader(skills_dir)
    skills = loader.discover_skills()
    print(f"✅ Discovered {len(skills)} Claude Skills")

    tools = [
        GetSkillTool(loader),
    ]

    return tools, loader
