
from typing import Any, Generator, Optional
import mlflow
import mlflow.deployments
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import dspy
import uuid
import os

LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
dspy.settings.configure(lm=lm)
endpoint_name= "ka-ca0f678f-endpoint" #update this

class rag_signature(dspy.Signature):
  """
  use the given tools to answer the question
  """ 
  question: str = dspy.InputField()
  response: str = dspy.OutputField() 

class DSPyChatAgent(ChatAgent):     
    def __init__(self):
      self.rag_signature = rag_signature
      self.endpoint_name = endpoint_name
      self.rag_agent = dspy.ReAct(self.rag_signature, tools=[self.sec_search],max_iters=1)
    

    def sec_search(self, databricks_question):
        """This function needs the User's question. The question is used to pull documentation about Databricks. Use the information to answer the user's question"""
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint=self.endpoint_name,
            inputs={"dataframe_split": {
                "columns": ["input"],
                "data": [[
                    [{"role": "user", "content": databricks_question}]
                ]]
            }}
        )
        return response['predictions']['output'][0]['content'][0]['text']
      
    def prepare_message_history(self, messages: list[ChatAgentMessage]):
        history_entries = []
        # Assume the last message in the input is the most recent user question.
        for i in range(0, len(messages) - 1, 2):
            history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
        return dspy.History(messages=history_entries)
      
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        latest_question = messages[-1].content
        response = self.rag_agent(question=latest_question).response
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
        )

from mlflow.models import set_model
AGENT = DSPyChatAgent()
mlflow.dspy.autolog()
# mlflow.set_experiment(experiment_id="835bf9ec05f24eb09289e8030853d968")
mlflow.set_experiment(experiment_name="/Users/austin.choi@databricks.com/ai_pioneer_h2/03_Agents_in_Code")
set_model(AGENT)
