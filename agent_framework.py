"""
Agent Framework Module - Foundation for all intelligent agents in the system

This module provides the base classes and implementations for creating specialized
intelligent agents within the Agent Village ecosystem. It handles agent lifecycle,
communication, perception, cognition, and action execution.

Dependencies:
- Python 3.10+
- PyTorch 2.4.0+
- langchain 0.1.0+
- pydantic 2.3+
- SQLAlchemy 2.0+
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManager
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_village.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Represents the current operational status of an agent"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


class MessageType(Enum):
    """Defines the types of messages agents can exchange"""
    CHAT = "chat"            # Regular conversation
    COMMAND = "command"      # Command to be executed
    RESULT = "result"        # Result of an executed command
    SYSTEM = "system"        # System message
    STATUS = "status"        # Status update
    QUERY = "query"          # Information request
    RESPONSE = "response"    # Response to a query
    ERROR = "error"          # Error message


class Message(BaseModel):
    """Standardized message format for agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: datetime.datetime.now().timestamp())
    sender: str
    recipient: Optional[str] = None  # None means broadcast
    message_type: MessageType
    content: Dict[str, Any]
    thread_id: Optional[str] = None  # For tracking conversation threads
    in_reply_to: Optional[str] = None  # ID of the message this is replying to
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "thread_id": self.thread_id,
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from dictionary data"""
        # Convert string message_type back to enum
        if isinstance(data["message_type"], str):
            data["message_type"] = MessageType(data["message_type"])
        return cls(**data)


class AgentMemory:
    """
    Memory system for agents to store and retrieve information
    Handles encoding, storage, and retrieval of agent experiences
    """
    def __init__(
        self,
        agent_id: str,
        storage_path: str = "./memory_storage",
        embedding_model=None,
        max_tokens_limit: int = 4000,
    ):
        self.agent_id = agent_id
        self.storage_path = os.path.join(storage_path, f"agent_{agent_id}")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize vector database
        self.vectorstore = Chroma(
            collection_name=f"agent_{agent_id}_memory",
            embedding_function=self.embedding_model,
            persist_directory=self.storage_path
        )
        
        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Short-term memory (recent interactions)
        self.short_term_memory: List[Message] = []
        self.max_short_term_items = 50
        
        # Token tracking
        self.max_tokens_limit = max_tokens_limit
        self.token_count = 0
        
        # Working memory (current task state)
        self.working_memory: Dict[str, Any] = {}
        
        logger.info(f"Initialized memory system for agent {agent_id}")
    
    def add(self, message: Message) -> None:
        """
        Add a message to both short-term and long-term memory
        
        Args:
            message: The message to store in memory
        """
        # Add to short term memory
        self.short_term_memory.append(message)
        
        # Trim short term memory if needed
        if len(self.short_term_memory) > self.max_short_term_items:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_items:]
        
        # Add to vector store for long-term memory
        text_representation = f"Time: {datetime.datetime.fromtimestamp(message.timestamp).isoformat()}\n"
        text_representation += f"From: {message.sender}\n"
        text_representation += f"To: {message.recipient or 'All'}\n"
        text_representation += f"Type: {message.message_type.value}\n"
        text_representation += f"Content: {json.dumps(message.content)}"
        
        self.vectorstore.add_documents([
            Document(
                page_content=text_representation,
                metadata={"timestamp": message.timestamp, "message_id": message.id}
            )
        ])

        
        
        logger.debug(f"Agent {self.agent_id} stored message {message.id} in memory")
    
    def get_relevant_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on a query
        
        Args:
            query: The query to search for relevant context
            k: Number of relevant items to retrieve
            
        Returns:
            List of relevant memory items
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """
        Get the n most recent messages from short-term memory
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of the n most recent messages
        """
        return self.short_term_memory[-n:]
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """
        Store a value in working memory
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.working_memory[key] = value
    
    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from working memory
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The stored value or default
        """
        return self.working_memory.get(key, default)
    
    def clear_working_memory(self) -> None:
        """Clear all items in working memory"""
        self.working_memory = {}
    
    def save(self) -> None:
        """Persist memory to disk"""
        # Note: ChromaDB 0.4+ auto-persists when persist_directory is set.
        with open(os.path.join(self.storage_path, "short_term.json"), "w") as f:
            json.dump([m.to_dict() for m in self.short_term_memory], f)
        
        # Save working memory
        with open(os.path.join(self.storage_path, "working_memory.json"), "w") as f:
            json.dump(self.working_memory, f)
            
        logger.info(f"Agent {self.agent_id} memory saved to disk")
    
    def load(self) -> None:
        """Load memory from disk"""
        # Load short-term memory
        short_term_path = os.path.join(self.storage_path, "short_term.json")
        if os.path.exists(short_term_path):
            with open(short_term_path, "r") as f:
                data = json.load(f)
                self.short_term_memory = [Message.from_dict(m) for m in data]
        
        # Load working memory
        working_memory_path = os.path.join(self.storage_path, "working_memory.json")
        if os.path.exists(working_memory_path):
            with open(working_memory_path, "r") as f:
                self.working_memory = json.load(f)
                
        logger.info(f"Agent {self.agent_id} memory loaded from disk")


class Tool(BaseModel):
    """Tool/capability that an agent can use to interact with environment"""
    model_config = {"arbitrary_types_allowed": True}
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_permissions: List[str] = Field(default_factory=list)
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with the given arguments"""
        try:
            logger.debug(f"Executing tool {self.name} with args: {args}, kwargs: {kwargs}")
            result = self.function(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            raise


class AgentConfig(BaseModel):
    """Configuration settings for an agent"""
    name: str
    role: str
    description: str
    llm_model: str
    llm_provider: str = "anthropic"  # Supported: "anthropic", "deepseek", "openai"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    memory_path: str = "./memory_storage"
    tools: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    system_prompt: str = ""
    custom_configuration: Dict[str, Any] = Field(default_factory=dict)


class Agent(ABC):
    """
    Base class for all agents in the system
    
    This abstract class defines the interface and basic functionality
    for all intelligent agents in the Agent Village ecosystem.
    """
    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLLM,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = config.name
        self.role = config.role
        self.description = config.description
        self.status = AgentStatus.INITIALIZING
        
        # Set up LLM
        self.llm = llm
        self.llm_temperature = config.llm_temperature
        self.llm_max_tokens = config.llm_max_tokens
        
        # Set up memory
        self.memory = AgentMemory(self.id, config.memory_path)
        
        # Set up tools and permissions
        self.tools: Dict[str, Tool] = {}
        self.permissions = set(config.permissions)
        
        # Communication channels
        self.message_queue = asyncio.Queue()
        self.response_callbacks: Dict[str, Callable] = {}
        self.dispatch_callback = None  # Set by chatroom to route outgoing messages back

        # Pause control — set means running, clear means paused
        self._paused = asyncio.Event()
        self._paused.set()
        
        # System prompt
        self.system_prompt = config.system_prompt
        if not self.system_prompt:
            self.system_prompt = self._generate_default_system_prompt()
            
        # Custom configuration
        self.custom_config = config.custom_configuration
        
        # Set up callback manager
        self.callback_manager = callback_manager
        
        # Performance metrics
        self.metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "tools_used": 0,
            "errors_encountered": 0,
            "tasks_completed": 0,
            "response_time": []
        }
        
        logger.info(f"Initialized agent {self.name} ({self.id}) with role {self.role}")
        
    def _generate_default_system_prompt(self) -> str:
        """Generate a default system prompt based on agent information"""
        prompt = f"You are {self.name}, a {self.role} in the 'Crayon Box' agent village system. "
        prompt += f"\n\nDescription of your role: {self.description}"
        prompt += "\n\nYou collaborate with other AI agents to achieve collective goals. "
        prompt += "You should always act in accordance with your role and responsibilities."
        prompt += "\n\nWhen communicating with other agents, be clear, concise, and helpful."
        return prompt
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool that the agent can use
        
        Args:
            tool: The tool to add
        """
        # Check if agent has permissions to use this tool
        for permission in tool.required_permissions:
            if permission not in self.permissions:
                logger.warning(f"Agent {self.name} does not have permission {permission} required for tool {tool.name}")
                return
        
        #Check if already registered
        if tool.name in self.tools:
            return

        self.tools[tool.name] = tool
        logger.info(f"Added tool {tool.name} to agent {self.name}")
    
    async def receive_message(self, message: Message) -> None:
        """
        Receive a message and add it to the message queue
        
        Args:
            message: The message to receive
        """
        # Validate message is for this agent
        if message.recipient and message.recipient != self.name and message.recipient != "all":
            logger.warning(f"Agent {self.name} received message intended for {message.recipient}")
            return
        
        # Log message receipt
        logger.debug(f"Agent {self.name} received message: {message.id} from {message.sender}")
        
        # Store in memory
        self.memory.add(message)
        
        # Update metrics
        self.metrics["messages_received"] += 1
        
        # Add to queue
        await self.message_queue.put(message)
    
    def perceive(self, message: Message) -> Dict[str, Any]:
        """
        Process an incoming message to extract relevant information
        
        Args:
            message: The message to perceive
            
        Returns:
            Dictionary of perceived information
        """
        # Extract relevant information based on message type
        perception = {
            "message_id": message.id,
            "sender": message.sender,
            "message_type": message.message_type.value,
            "timestamp": message.timestamp,
            "content": message.content,
            "thread_id": message.thread_id,
        }
        
        # Enhance perception with contextual information
        if message.message_type == MessageType.QUERY:
            # Retrieve relevant context for queries
            query = message.content.get("query", "")
            if query:
                relevant_context = self.memory.get_relevant_context(query)
                perception["relevant_context"] = relevant_context
        
        return perception
    
    @abstractmethod
    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate thoughts based on perception
        
        Args:
            perception: Perceived information
            
        Returns:
            Dictionary containing thoughts and reasoning
        """
        pass
    
    @abstractmethod
    async def decide_action(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take based on thoughts
        
        Args:
            thoughts: The agent's thoughts
            
        Returns:
            Dictionary containing action decision
        """
        pass
    
    async def act(self, action_decision: Dict[str, Any]) -> Any:
        """
        Execute the decided action
        
        Args:
            action_decision: The action decision
            
        Returns:
            Result of the action
        """
        action_type = action_decision.get("action_type")
        
        if action_type == "send_message":
            return await self._send_message(action_decision)
        elif action_type == "use_tool":
            return await self._use_tool(action_decision)
        elif action_type == "update_status":
            return self._update_status(action_decision)
        elif action_type == "no_action":
            return None
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None
    
    async def _send_message(self, action_decision: Dict[str, Any]) -> Message:
        """
        Send a message to another agent or the environment
        
        Args:
            action_decision: Decision containing message details
            
        Returns:
            The sent message
        """
        content = action_decision.get("content", {})
        recipient = action_decision.get("recipient")
        message_type = MessageType(action_decision.get("message_type", "chat"))
        thread_id = action_decision.get("thread_id")
        in_reply_to = action_decision.get("in_reply_to")
        
        message = Message(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            thread_id=thread_id,
            in_reply_to=in_reply_to
        )
        
        # Update metrics
        self.metrics["messages_sent"] += 1
        
        logger.debug(f"Agent {self.name} sending message: {message.id} to {recipient or 'all'}")
        
        # Dispatch back through the chatroom if a callback is registered
        if self.dispatch_callback:
            asyncio.create_task(self.dispatch_callback(message))
        
        return message
    
    async def _use_tool(self, action_decision: Dict[str, Any]) -> Any:
        """
        Use a tool to perform an action
        
        Args:
            action_decision: Decision containing tool details
            
        Returns:
            Result of using the tool
        """
        tool_name = action_decision.get("tool_name")
        args = action_decision.get("args", [])
        kwargs = action_decision.get("kwargs", {})
        
        if tool_name not in self.tools:
            logger.warning(f"Agent {self.name} attempted to use unknown/illegal tool: {tool_name}")
            return {"error": f"Unknown/illegal tool: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        try:
            # Update metrics
            self.metrics["tools_used"] += 1
            
            logger.debug(f"Agent {self.name} using tool: {tool_name}")
            
            # Execute the tool
            result = tool.execute(*args, **kwargs)
            return {"result": result, "tool": tool_name}
        except Exception as e:
            logger.error(f"Error using tool {tool_name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {"error": str(e), "tool": tool_name}
    
    def _update_status(self, action_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the agent's status
        
        Args:
            action_decision: Decision containing status details
            
        Returns:
            Updated status information
        """
        new_status = action_decision.get("status")
        if new_status in [status.value for status in AgentStatus]:
            self.status = AgentStatus(new_status)
            logger.info(f"Agent {self.name} updated status to {self.status.value}")
            return {"status": self.status.value}
        else:
            logger.warning(f"Agent {self.name} attempted to set invalid status: {new_status}")
            return {"error": f"Invalid status: {new_status}", "current_status": self.status.value}
    
    async def run_perception_cycle(self) -> None:
        """
        Run a single perception-cognition-action cycle
        
        This method implements the basic cognitive loop:
        1. Get a message from the queue
        2. Perceive the message
        3. Think about the perception
        4. Decide what action to take
        5. Execute the action
        """
        try:
            # Get a message from the queue
            message = await self.message_queue.get()
            
            start_time = datetime.datetime.now()
            
            # Perceive the message
            perception = self.perceive(message)
            
            # Think about the perception
            thoughts = await self.think(perception)
            
            # Decide what action to take
            action_decision = await self.decide_action(thoughts)
            
            # Execute the action
            result = await self.act(action_decision)
            
            end_time = datetime.datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.metrics["response_time"].append(processing_time)
            
            # Mark task as done
            self.message_queue.task_done()
            
            # Return the result (will be handled by the environment)
            return result
        except Exception as e:
            logger.error(f"Error in perception cycle for agent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return None
    
    async def run(self) -> None:
        """
        Run the agent continuously
        
        Continuously processes messages and executes perception cycles
        """
        self.status = AgentStatus.ACTIVE
        logger.info(f"Agent {self.name} starting continuous operation")
        
        try:
            while self.status in (AgentStatus.ACTIVE, AgentStatus.PAUSED):
                # Block here while paused, without consuming CPU
                await self._paused.wait()

                # Re-check status after unblocking (could have been terminated while paused)
                if self.status != AgentStatus.ACTIVE:
                    break

                # Process one message
                result = await self.run_perception_cycle()
                
                # If we need to wait for more messages, add a small delay
                if self.message_queue.empty():
                    await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"Error in agent {self.name} main loop: {str(e)}")
            self.status = AgentStatus.ERROR
        finally:
            # Save agent memory before shutting down
            self.memory.save()
            logger.info(f"Agent {self.name} shutting down, memory saved")
            
    def pause(self) -> None:
        """Pause the agent's run loop. In-flight LLM calls complete before the loop suspends."""
        self._paused.clear()
        self.status = AgentStatus.PAUSED
        logger.info(f"Agent {self.name} paused")

    def resume(self) -> None:
        """Resume a paused agent."""
        self._paused.set()
        self.status = AgentStatus.ACTIVE
        logger.info(f"Agent {self.name} resumed")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate average response time
        if metrics["response_time"]:
            metrics["avg_response_time"] = sum(metrics["response_time"]) / len(metrics["response_time"])
        else:
            metrics["avg_response_time"] = 0
            
        return metrics


class LLMAgent(Agent):
    """
    Agent implementation using a Language Model for cognition
    
    This class implements the abstract think() and decide_action() methods
    using prompts to the LLM for generating thoughts and decisions.
    """
    
    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate thoughts using the LLM based on perception
        
        Args:
            perception: Perceived information
            
        Returns:
            Dictionary containing thoughts and reasoning
        """
        # Format perception for the prompt
        perception_text = json.dumps(perception, indent=2)
        
        # Get relevant context from memory
        recent_messages = self.memory.get_recent_messages(5)
        recent_messages_text = "\n".join([
            f"From: {m.sender}, To: {m.recipient or 'All'}, Type: {m.message_type.value}, "
            f"Content: {json.dumps(m.content)}"
            for m in recent_messages
        ])
        
        # Create the prompt for thinking
        thinking_template = PromptTemplate(
            input_variables=["system_prompt", "agent_name", "agent_role", "perception", "recent_messages"],
            template="""
            {system_prompt}
            
            You are currently processing information as {agent_name}, a {agent_role}.
            
            ## Recent Messages:
            {recent_messages}
            
            ## Current Perception:
            {perception}
            
            ## Think:
            Analyze the situation and generate your thoughts. Consider:
            1. What is the key information in this message?
            2. How does it relate to your goals and role?
            3. What context from memory is relevant?
            4. What are the implications?
            
            Provide your thoughts in a structured format.
            """
        )
        
        # Generate thoughts using the LLM
        thinking_prompt = thinking_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            agent_role=self.role,
            perception=perception_text,
            recent_messages=recent_messages_text
        )
        
        try:
            thoughts_response = await self.llm.ainvoke(thinking_prompt)
            thoughts_str = thoughts_response.content.strip()
            
            # Parse thoughts into a structured format
            thoughts = {
                "raw": thoughts_str,
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id")
            }
            
            logger.debug(f"Agent {self.name} generated thoughts for message {perception.get('message_id')}")
            
            return thoughts
        except Exception as e:
            logger.error(f"Error generating thoughts for agent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "raw": f"Error generating thoughts: {str(e)}",
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id"),
                "error": str(e)
            }
    
    async def decide_action(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take using the LLM based on thoughts
        
        Args:
            thoughts: The agent's thoughts
            
        Returns:
            Dictionary containing action decision
        """
        # Format thoughts for the prompt
        thoughts_text = thoughts.get("raw", "")
        
        # Get available tools
        tools_text = "\n".join([
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        ])
        
        # Create the prompt for decision making
        decision_template = PromptTemplate(
            input_variables=["system_prompt", "agent_name", "agent_role", "thoughts", "tools"],
            template="""
            {system_prompt}
            
            You are currently deciding on actions as {agent_name}, a {agent_role}.
            
            ## Your Thoughts:
            {thoughts}
            
            ## Available Tools:
            {tools}
            
            ## Action Decision:
            Based on your thoughts, decide what action to take. Your options include:
            
            1. send_message: Send a message to another agent or broadcast to all
            2. use_tool: Use one of your available tools
            3. update_status: Update your operational status
            4. no_action: Take no action
            
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
        
        # Generate decision using the LLM
        decision_prompt = decision_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            agent_role=self.role,
            thoughts=thoughts_text,
            tools=tools_text
        )
        
        try:
            decision_response = await self.llm.ainvoke(decision_prompt)
            decision_str = decision_response.content.strip()
            
            # Extract JSON from the response
            try:
                # Find JSON in the response - sometimes LLM adds explanation text
                import re
                json_match = re.search(r'({.*})', decision_str, re.DOTALL)
                if json_match:
                    decision_str = json_match.group(1)
                
                decision = json.loads(decision_str)
                logger.debug(f"Agent {self.name} decided on action: {decision.get('action_type')}")
                return decision
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing decision JSON for agent {self.name}: {str(e)}")
                logger.error(f"Raw decision: {decision_str}")
                # Fall back to no_action
                return {
                    "action_type": "no_action",
                    "reason": f"Error parsing decision: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error generating decision for agent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "action_type": "no_action",
                "reason": f"Error generating decision: {str(e)}"
            }
    