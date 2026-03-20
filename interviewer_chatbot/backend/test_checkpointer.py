from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from services.gemini_client import gemini_client
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages


# 1. Define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # ← add this reducer


# 2. Gemini model
llm = gemini_client


# 3. Node
def chatbot(state: ChatState):
    messages = state["messages"]

    try:
        # Use the method that supports message history
        text = llm.generate_content_list(messages)

        # If empty → fallback message
        if not text.strip():
            text = "(Gemini returned empty response)"

        ai_message = AIMessage(content=text)

    except Exception as e:
        logger.exception("Error in chatbot node")
        ai_message = AIMessage(content=f"Error calling Gemini: {str(e)}")

    return {"messages": messages + [ai_message]}


# 4. Build graph
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)
builder.set_entry_point("chatbot")
builder.add_edge("chatbot", END)

# 5. Memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# 6. Terminal chat loop
def run_chat():
    print("🤖 Chat started (type 'exit' to quit)\n")

    # same thread_id = same memory
    config = {"configurable": {"thread_id": "terminal-user"}}

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        print(f"raw message {result['messages']}")
        print("AI:", result["messages"][-1].content)
        print()


if __name__ == "__main__":
    run_chat()
