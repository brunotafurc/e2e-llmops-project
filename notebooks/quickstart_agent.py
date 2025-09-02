import json
import uuid
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
from typing import Any, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

# Get an OpenAI client configured to talk to Databricks model serving endpoints
# We'll use this to query an LLM in our agent
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

# The snippet below tries to pick the first LLM API available in your Databricks workspace
# from a set of candidates. You can override and simplify it
# to just specify LLM_ENDPOINT_NAME.
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"

# Enable automatic tracing of LLM calls
mlflow.openai.autolog()


@mlflow.trace
def predict_agent(message):

def run_agent(content):
    """
    Send a user prompt to the LLM, and return a list of LLM response messages
    The LLM is allowed to call the code interpreter tool if needed, to respond to the user
    """
    result_msgs = []
    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=[{"role": "user", "content": content}],
    )
    msg = response.choices[0].message
    result_msgs.append(msg.to_dict())
    return result_msgs

class QuickstartAgent(ChatAgent):
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        prompt = "You will receive a phrase, please return the sentiment. You only have three options: positive, neutral or negative. This is the phrase: "
        message = messages[-1].content
        raw_msgs = run_agent(prompt + " " + message)
        out = []
        for m in raw_msgs:
            out.append(ChatAgentMessage(
                id=uuid.uuid4().hex,
                **m
            ))

        return ChatAgentResponse(messages=out)

AGENT = QuickstartAgent()
mlflow.models.set_model(AGENT)