
from typing import Any, Generator, Optional
from databricks.sdk.service.dashboards import GenieAPI
import mlflow
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
from databricks.vector_search.client import VectorSearchClient

mlflow.dspy.autolog()
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
dspy.settings.configure(lm=lm)
VECTOR_SEARCH_ENDPOINT_NAME = "ka-769ce120-vs-endpoint"
vectorSearchIndexName = "__databricks_internal_catalog_tiles_2703287350484668.769ce120_73ad70ab11e841cc.ka_769ce120_709cf6e4_index"

class rag_signature(dspy.Signature):
  """
  use the given tools to answer the question
  """ 
  question: str = dspy.InputField()
  response: str = dspy.OutputField() 

class DSPyChatAgent(ChatAgent):     
    def __init__(self):
      self.rag_signature = rag_signature
      self.vector_endpoint = VECTOR_SEARCH_ENDPOINT_NAME
      self.vector_index = vectorSearchIndexName
      self.rag_agent = dspy.ReAct(self.rag_signature, tools=[self.sec_search],max_iters=1)

    def sec_search(self, databricks_question):
        """This function needs the User's question. The question is used to pull documentation about Databricks. Use the information to answer the user's question"""

        #TODO: Set this in the init
        VECTOR_SEARCH_ENDPOINT_NAME = self.vector_endpoint
        vectorSearchIndexName = self.vector_index

        vsc = VectorSearchClient(
        )

        index = vsc.get_index(endpoint_name=self.vector_endpoint, index_name=self.vector_index)

        result = index.similarity_search(num_results=3, columns=["chunk_text"], query_text=databricks_question)

        return result['result']['data_array'][0][0]
      
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
set_model(AGENT)
