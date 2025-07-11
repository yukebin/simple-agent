from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from tools.crawler import vitalbridge_info
from config import MODEL_NAME, TEMPERATURE


class AnswerSchema(BaseModel):
    """Structured answer format"""
    is_lvzhou: bool = Field(description="Is this about VitalBridge?")
    answer: str = Field(description="The actual answer to the user's question")


prompt = ChatPromptTemplate.from_messages([
    ("system", ("You are a helpful assistant. Use the 'vitalbridge_info' tool "
                "ONLY for questions about Vitalbridge or vitalbridge.com.")),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

agent = create_openai_functions_agent(
    llm=model,
    tools=[vitalbridge_info],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[vitalbridge_info], verbose=True)

# Run example
if __name__ == "__main__":
    question = "What kind of startups does Vitalbridge invest in?"
    response = agent_executor.invoke({"input": question})
    print(response)