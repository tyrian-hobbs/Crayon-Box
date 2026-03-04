"""
Agent Roles Module - Specialised agent implementations for the AI Village

This module contains concrete agent subclasses that extend the base LLMAgent
with role-specific behaviour, state, and prompting. It also provides the
create_agent factory function used by the orchestration layer.

To add a new agent role, subclass LLMAgent (or Agent directly), implement any
overrides, and add a routing condition to create_agent.
"""

import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import PromptTemplate

from agent_framework import Agent, AgentConfig, LLMAgent, MessageType

logger = logging.getLogger(__name__)

def create_agent(
    config: AgentConfig,
    llm: BaseLLM,
    callback_manager: Optional[CallbackManager] = None,
    role: Optional[str] = None
) -> Agent:
    """
    Factory function to create the appropriate agent class based on role.

    Matches the role string against ROLE_REGISTRY keys as substrings, in
    definition order. Falls back to LLMAgent if no match is found.

    Args:
        config: Configuration for the agent
        llm: Language model to use
        callback_manager: Optional callback manager
        role: Role string used for class routing. Defaults to config.role if not provided.

    Returns:
        The created agent
    """
    role_lower = (role or config.role).lower()

    agent_class = next(
        (cls for keyword, cls in ROLE_REGISTRY.items() if keyword in role_lower),
        LLMAgent
    )

    return agent_class(config, llm, callback_manager)



class ProjectManagerAgent(LLMAgent):
    """
    Specialized agent for project management

    This agent coordinates other agents, tracks progress, and ensures
    the team is moving toward its goals.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLLM,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(config, llm, callback_manager)

        # Project management specific state
        self.project_state = {
            "goals": [],
            "tasks": {},
            "team_members": {},
            "progress": {},
            "issues": [],
            "timeline": {}
        }

        # Initialize PM-specific memory keys
        self.memory.set_working_memory("project_state", self.project_state)

        logger.info(f"Initialized ProjectManagerAgent: {self.name}")

    def set_project_goals(self, goals: List[str]) -> None:
        """
        Set the goals for the project

        Args:
            goals: List of project goals
        """
        self.project_state["goals"] = goals
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} set goals: {goals}")

    def add_team_member(self, agent_name: str, agent_role: str) -> None:
        """
        Add a team member to the project

        Args:
            agent_name: Name of the agent
            agent_role: Role of the agent
        """
        self.project_state["team_members"][agent_name] = {
            "role": agent_role,
            "status": "active",
            "tasks_assigned": [],
            "tasks_completed": []
        }
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} added team member: {agent_name} ({agent_role})")

    def assign_task(self, task_id: str, task_description: str, assignee: str, deadline: Optional[str] = None) -> None:
        """
        Assign a task to a team member

        Args:
            task_id: Unique ID for the task
            task_description: Description of the task
            assignee: Name of the agent assigned to the task
            deadline: Optional deadline for the task
        """
        if assignee not in self.project_state["team_members"]:
            logger.warning(f"Cannot assign task to unknown team member: {assignee}")
            return

        self.project_state["tasks"][task_id] = {
            "description": task_description,
            "assignee": assignee,
            "status": "assigned",
            "deadline": deadline,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "updates": []
        }

        self.project_state["team_members"][assignee]["tasks_assigned"].append(task_id)
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} assigned task {task_id} to {assignee}")

    def update_task_status(self, task_id: str, status: str, update_message: Optional[str] = None) -> None:
        """
        Update the status of a task

        Args:
            task_id: ID of the task to update
            status: New status for the task
            update_message: Optional message describing the update
        """
        if task_id not in self.project_state["tasks"]:
            logger.warning(f"Cannot update unknown task: {task_id}")
            return

        task = self.project_state["tasks"][task_id]
        task["status"] = status
        task["updated_at"] = datetime.datetime.now().isoformat()

        if update_message:
            task["updates"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "message": update_message
            })

        if status == "completed":
            assignee = task["assignee"]
            if assignee in self.project_state["team_members"]:
                if task_id in self.project_state["team_members"][assignee]["tasks_assigned"]:
                    self.project_state["team_members"][assignee]["tasks_assigned"].remove(task_id)
                self.project_state["team_members"][assignee]["tasks_completed"].append(task_id)

        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} updated task {task_id} status to {status}")

    def get_project_status(self) -> Dict[str, Any]:
        """
        Get the current status of the project

        Returns:
            Dictionary containing project status
        """
        total_tasks = len(self.project_state["tasks"])
        completed_tasks = sum(1 for task in self.project_state["tasks"].values() if task["status"] == "completed")
        in_progress_tasks = sum(1 for task in self.project_state["tasks"].values() if task["status"] in ["in_progress", "reviewing"])

        progress_percentage = 0
        if total_tasks > 0:
            progress_percentage = (completed_tasks / total_tasks) * 100

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": total_tasks - completed_tasks - in_progress_tasks,
            "progress_percentage": progress_percentage,
            "team_size": len(self.project_state["team_members"]),
            "issues": len(self.project_state["issues"])
        }

    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base think method to include project management context

        Args:
            perception: Perceived information

        Returns:
            Dictionary containing thoughts and reasoning
        """
        project_status = self.get_project_status()
        perception_text = json.dumps(perception, indent=2)

        recent_messages = self.memory.get_recent_messages(5)
        recent_messages_text = "\n".join([
            f"From: {m.sender}, To: {m.recipient or 'All'}, Type: {m.message_type.value}, "
            f"Content: {json.dumps(m.content)}"
            for m in recent_messages
        ])

        project_state_text = json.dumps(self.project_state, indent=2)

        thinking_template = PromptTemplate(
            input_variables=[
                "system_prompt", "agent_name", "perception",
                "recent_messages", "project_state", "project_status"
            ],
            template="""
            {system_prompt}

            You are currently processing information as {agent_name}, the Project Manager.

            ## Project Status:
            {project_status}

            ## Recent Messages:
            {recent_messages}

            ## Current Perception:
            {perception}

            ## Project State:
            {project_state}

            ## Think:
            As the Project Manager, analyze the situation considering:
            1. What is the key information in this message?
            2. How does it relate to the project goals and current state?
            3. Does this require task assignment or status updates?
            4. Are there any issues or risks that need to be addressed?
            5. How can you keep the team aligned and productive?

            Provide your thoughts in a structured format.
            """
        )

        thinking_prompt = thinking_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            perception=perception_text,
            recent_messages=recent_messages_text,
            project_state=project_state_text,
            project_status=json.dumps(project_status, indent=2)
        )

        try:
            thoughts_response = await self.llm.ainvoke(thinking_prompt)
            thoughts_str = thoughts_response.content.strip()

            thoughts = {
                "raw": thoughts_str,
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id"),
                "project_status": project_status
            }

            logger.debug(f"ProjectManagerAgent {self.name} generated thoughts for message {perception.get('message_id')}")
            return thoughts
        except Exception as e:
            logger.error(f"Error generating thoughts for ProjectManagerAgent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "raw": f"Error generating thoughts: {str(e)}",
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id"),
                "error": str(e),
                "project_status": project_status
            }

    async def decide_action(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base decide_action method to include project management options

        Args:
            thoughts: The agent's thoughts

        Returns:
            Dictionary containing action decision
        """
        import re

        thoughts_text = thoughts.get("raw", "")
        tools_text = "\n".join([
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        ])
        project_status = thoughts.get("project_status", {})
        project_status_text = json.dumps(project_status, indent=2)

        decision_template = PromptTemplate(
            input_variables=[
                "system_prompt", "agent_name", "thoughts",
                "tools", "project_status"
            ],
            template="""
            {system_prompt}

            You are currently deciding on actions as {agent_name}, the Project Manager.

            ## Project Status:
            {project_status}

            ## Your Thoughts:
            {thoughts}

            ## Available Tools:
            {tools}

            ## Action Decision:
            Based on your thoughts, decide what action to take. Your options include:

            1. send_message: Send a message to another agent or broadcast to all
            2. use_tool: Use one of your available tools
            3. project_action: Take a project management action (assign task, update status, etc.)
            4. update_status: Update your operational status
            5. no_action: Take no action

            Provide your decision in a structured JSON format with the following schema:

            For send_message:
            {{
                "action_type": "send_message",
                "recipient": "<recipient_name or null for broadcast>",
                "message_type": "<chat|command|query|response|status>",
                "content": {{
                    "text": "<message_text>",
                    "additional_data": "<any_additional_data>"
                }},
                "thread_id": "<thread_id if continuing a conversation>",
                "in_reply_to": "<message_id if replying to a specific message>"
            }}

            For use_tool:
            {{
                "action_type": "use_tool",
                "tool_name": "<name_of_tool>",
                "args": [<positional_arguments>],
                "kwargs": {{<keyword_arguments>}}
            }}

            For project_action:
            {{
                "action_type": "project_action",
                "project_action": "<assign_task|update_task|add_team_member|set_goals>",
                "action_data": {{
                    // Data specific to the project action
                }}
            }}

            For update_status:
            {{
                "action_type": "update_status",
                "status": "<active|paused|terminated>"
            }}

            For no_action:
            {{
                "action_type": "no_action",
                "reason": "<reason_for_no_action>"
            }}

            Only output the JSON object with no additional text.
            """
        )

        decision_prompt = decision_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            thoughts=thoughts_text,
            tools=tools_text,
            project_status=project_status_text
        )

        try:
            decision_response = await self.llm.ainvoke(decision_prompt)
            decision_str = decision_response.content.strip()

            try:
                json_match = re.search(r'({.*})', decision_str, re.DOTALL)
                if json_match:
                    decision_str = json_match.group(1)

                decision = json.loads(decision_str)

                if decision.get("action_type") == "project_action":
                    decision = await self._handle_project_action(decision)

                logger.debug(f"ProjectManagerAgent {self.name} decided on action: {decision.get('action_type')}")
                return decision
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing decision JSON for ProjectManagerAgent {self.name}: {str(e)}")
                logger.error(f"Raw decision: {decision_str}")
                return {
                    "action_type": "no_action",
                    "reason": f"Error parsing decision: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error generating decision for ProjectManagerAgent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "action_type": "no_action",
                "reason": f"Error generating decision: {str(e)}"
            }

    async def _handle_project_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle project-specific actions

        Args:
            decision: The project action decision

        Returns:
            Updated action decision for the agent system
        """
        project_action = decision.get("project_action")
        action_data = decision.get("action_data", {})

        if project_action == "assign_task":
            task_id = action_data.get("task_id", str(uuid.uuid4()))
            task_description = action_data.get("description", "")
            assignee = action_data.get("assignee", "")
            deadline = action_data.get("deadline")

            self.assign_task(task_id, task_description, assignee, deadline)

            return {
                "action_type": "send_message",
                "recipient": assignee,
                "message_type": MessageType.COMMAND.value,
                "content": {
                    "text": f"You have been assigned a new task: {task_description}",
                    "task_id": task_id,
                    "deadline": deadline
                }
            }

        elif project_action == "update_task":
            task_id = action_data.get("task_id", "")
            status = action_data.get("status", "")
            update_message = action_data.get("message", "")

            self.update_task_status(task_id, status, update_message)

            return {
                "action_type": "send_message",
                "recipient": None,  # Broadcast
                "message_type": MessageType.STATUS.value,
                "content": {
                    "text": f"Task {task_id} status updated to: {status}",
                    "task_id": task_id,
                    "status": status,
                    "message": update_message
                }
            }

        elif project_action == "add_team_member":
            agent_name = action_data.get("name", "")
            agent_role = action_data.get("role", "")

            self.add_team_member(agent_name, agent_role)

            return {
                "action_type": "send_message",
                "recipient": None,  # Broadcast
                "message_type": MessageType.SYSTEM.value,
                "content": {
                    "text": f"New team member added: {agent_name} ({agent_role})",
                    "agent_name": agent_name,
                    "agent_role": agent_role
                }
            }

        elif project_action == "set_goals":
            goals = action_data.get("goals", [])

            self.set_project_goals(goals)

            return {
                "action_type": "send_message",
                "recipient": None,  # Broadcast
                "message_type": MessageType.SYSTEM.value,
                "content": {
                    "text": f"Project goals have been set: {', '.join(goals)}",
                    "goals": goals
                }
            }

        else:
            logger.warning(f"Unknown project action: {project_action}")
            return {
                "action_type": "no_action",
                "reason": f"Unknown project action: {project_action}"
            }
        

# Registry mapping role keywords to agent classes.
# Keys are matched as substrings against the lowercased role string, in
# definition order. The first match wins. To add a new role, import its
# class and add an entry here — create_agent needs no other changes.
ROLE_REGISTRY: Dict[str, type] = {
    "manager": ProjectManagerAgent,
    "project manager": ProjectManagerAgent,
    "team lead": ProjectManagerAgent,
}
