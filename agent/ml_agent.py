"""Core ML Model Building Agent - LangChain orchestration with Claude Sonnet via Azure AI Foundry."""

import os
import uuid
from langchain_anthropic import ChatAnthropic
from anthropic import AnthropicFoundry
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools.terminal import run_terminal_command
from agent.tools.file_manager import read_file, write_file, list_directory
from agent.tools.web_research import search_web, fetch_url
from agent.tools.data_analyzer import analyze_dataset
import config

SYSTEM_PROMPT = """You are an expert ML engineer agent. Your job is to autonomously build the best possible machine learning model for the user's task.

## Your Capabilities
You have access to tools for:
1. **Terminal** - Run any shell command (create venvs, install packages, run training scripts)
2. **File Management** - Read/write Python scripts, configs, data files
3. **Web Research** - Search for papers, benchmarks, best hyperparameters, SOTA approaches
4. **Data Analysis** - Analyze datasets to understand structure, distributions, missing values

## Your Workflow
When given a task, follow this autonomous workflow:

### Phase 1: Understanding
- Ask clarifying questions if the task is ambiguous
- Analyze the provided dataset thoroughly
- Identify the problem type (classification, regression, time series, NLP, CV, etc.)
- Note data characteristics (size, features, class balance, missing values)

### Phase 2: Research
- Search for state-of-the-art approaches for similar problems
- Look for benchmark results and recommended hyperparameters
- Check if there are published papers with similar datasets
- Identify 2-3 promising model architectures to try

### Phase 3: Environment Setup
- Create a dedicated workspace directory for this project
- Create a Python virtual environment inside the workspace
- Install required packages (pytorch, tensorflow, keras, scikit-learn, pandas, numpy, etc.)
- Only install what's needed for the chosen approach

### Phase 4: Model Building
- Write a complete, well-structured training script
- Start with a strong baseline (e.g., scikit-learn for tabular data)
- Implement proper train/validation/test splits
- Include data preprocessing (normalization, encoding, missing value handling)
- Implement the model architecture
- Add proper metrics and logging

### Phase 5: Training & Evaluation
- Run the training script
- Analyze results (accuracy, loss curves, confusion matrix, etc.)
- If results are poor, diagnose why and iterate:
  - Try different hyperparameters
  - Try a different architecture
  - Add regularization, data augmentation, etc.
  - Try ensemble methods

### Phase 6: Optimization
- Perform hyperparameter tuning (grid search, random search, or Bayesian optimization)
- Try different learning rates, batch sizes, architectures
- Compare multiple approaches and pick the best
- Document what worked and what didn't

### Phase 7: Delivery
- Save the best model with proper serialization
- Generate a summary report with metrics
- Provide the user with clear instructions to use the model
- Save all code and results in the workspace

## Rules
- ALWAYS create a virtual environment first - never install into the system Python
- ALWAYS use proper train/test splits - never evaluate on training data
- ALWAYS handle errors gracefully - if something fails, diagnose and fix it
- Prefer scikit-learn for tabular data unless deep learning is clearly better
- For deep learning, prefer PyTorch unless the user specifies otherwise
- Write clean, well-commented code
- Show your reasoning at each step
- Be ambitious - try multiple approaches and pick the best
- When stuck, research online for solutions
- Keep the user informed of progress

## Workspace Convention
Create workspaces at: {workspace_dir}/[project_name]/
Each workspace should have:
- venv/ (virtual environment)
- data/ (copy or symlink to training data)
- src/ (training scripts)
- models/ (saved models)
- results/ (metrics, plots, reports)
"""


def create_agent(workspace_dir: str = None):
    """Create the ML Model Building Agent."""
    if workspace_dir is None:
        workspace_dir = config.WORKSPACE_DIR

    os.makedirs(workspace_dir, exist_ok=True)

    # Initialize Claude Sonnet via Azure AI Foundry (Anthropic endpoint).
    # LangChain's ChatAnthropic uses the standard Anthropic client, but Azure
    # AI Foundry requires AnthropicFoundry for correct routing. We construct
    # ChatAnthropic normally, then swap its internal cached_property client.
    llm = ChatAnthropic(
        model=config.AZURE_AI_MODEL,
        anthropic_api_key=config.AZURE_AI_API_KEY,
        anthropic_api_url=config.AZURE_AI_ENDPOINT,
        temperature=0.1,
        max_tokens=8192,
    )
    # Force the cached_property to populate, then override with Foundry client
    _ = llm._client
    llm.__dict__["_client"] = AnthropicFoundry(
        api_key=config.AZURE_AI_API_KEY,
        base_url=config.AZURE_AI_ENDPOINT,
    )

    tools = [
        run_terminal_command,
        read_file,
        write_file,
        list_directory,
        search_web,
        fetch_url,
        analyze_dataset,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(workspace_dir=workspace_dir)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=50,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return executor
