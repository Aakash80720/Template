# LangGraph Agent Template

## models/state.py (Agent)
```python
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    context: list[str] = []
    tool_outputs: dict = {}
    requires_human_review: bool = False
    iteration_count: int = 0
    final_answer: str | None = None

    class Config:
        arbitrary_types_allowed = True
```

## pipelines/agents/tools.py
```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query")

@tool(args_schema=SearchInput)
def web_search(query: str) -> str:
    """Search the web for current information."""
    # Implement with Tavily, SerpAPI, etc.
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Use a safe evaluator — never eval() directly
        import ast
        tree = ast.parse(expression, mode='eval')
        result = eval(compile(tree, '<string>', 'eval'))  # noqa: S307 — restricted to AST
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOLS = [web_search, calculator]
```

## pipelines/agents/nodes.py
```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from services.llm import get_llm
from pipelines.agents.tools import TOOLS
from models.state import AgentState
from monitoring.logger import get_logger

logger = get_logger(__name__)

llm_with_tools = get_llm().bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
Think step by step. Use tools when you need external information.
When you have enough information, provide a final answer."""

def agent_node(state: AgentState) -> dict:
    """Core reasoning node — calls LLM with tools."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state.messages
    response = llm_with_tools.invoke(messages)
    logger.info(f"Agent response: tool_calls={bool(response.tool_calls)}")
    return {"messages": [response], "iteration_count": state.iteration_count + 1}

def human_review_node(state: AgentState) -> dict:
    """HITL interrupt point — pauses for human approval."""
    # LangGraph interrupt() suspends execution here and resumes on .update()
    from langgraph.types import interrupt
    decision = interrupt({
        "question": "Do you approve this action?",
        "proposed_action": state.messages[-1].content if state.messages else "",
    })
    return {"requires_human_review": False, "messages": [HumanMessage(content=f"Human decision: {decision}")]}

def should_use_tools(state: AgentState) -> str:
    """Conditional edge: route to tools, human review, or end."""
    last_message = state.messages[-1] if state.messages else None

    if state.iteration_count > 10:  # safety limit
        return "end"
    if state.requires_human_review:
        return "human_review"
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
```

## pipelines/agents/graph.py
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models.state import AgentState
from pipelines.agents.nodes import agent_node, human_review_node, tool_node, should_use_tools
from monitoring.logger import get_logger

logger = get_logger(__name__)

def build_agent_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("human_review", human_review_node)

    # Edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "human_review": "human_review", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("human_review", "agent")

    checkpointer = checkpointer or MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

    logger.info("Agent graph compiled successfully")
    return compiled
```

## Running the Agent (entrypoint pattern)
```python
from pipelines.agents.graph import build_agent_graph
from models.state import AgentState
from langchain_core.messages import HumanMessage

agent = build_agent_graph()
config = {"configurable": {"thread_id": "session-001"}}

# First invoke
result = agent.invoke(
    AgentState(messages=[HumanMessage(content="What is the weather in Paris?")]),
    config=config,
)

# If interrupted for HITL, resume with:
# agent.update_state(config, {"messages": [HumanMessage(content="approved")]})
# result = agent.invoke(None, config=config)
```
