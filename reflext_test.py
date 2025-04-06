from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, List, Sequence
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

llm_qwen = Ollama(model = "glm4:9b")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate = prompt | llm_qwen


essay = ""
request = HumanMessage(
    content="写一篇关于为什么小王子与现代童年有关的文章"
)
for chunk in generate.stream({"messages": [request]}):
    print(chunk, end="")
    essay += chunk

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名老师，正在批改一篇论文。为用户提交的内容生成评论和建议。"
            " 提供详细的建议，包括长度、深度、风格等要求。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# reflection_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
#             " Provide detailed recommendations, including requests for length, depth, style, etc.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
reflect = reflection_prompt | llm_qwen

reflection = ""
print("*"*90)
for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
    print(chunk, end="")
    reflection += chunk

print("*"*90)
for i in range(5):
    print("\n" + "*"*80 +" \n")
    for chunk in generate.stream(
        {"messages": [request, AIMessage(content=essay), HumanMessage(content=reflection)]}
    ):
        print(chunk, end="")



# print("\n" + reflection_prompt)



# class State(TypedDict):
#     messages: Annotated[list, add_messages]


# async def generation_node(state: State) -> State:
#     return {"messages": [await generate.ainvoke(state["messages"])]}


# async def reflection_node(state: State) -> State:
#     # Other messages we need to adjust
#     cls_map = {"ai": HumanMessage, "human": AIMessage}
#     # First message is the original user request. We hold it the same for all nodes
#     translated = [state["messages"][0]] + [
#         cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
#     ]
#     res = await reflect.ainvoke(translated)
#     # We treat the output of this as human feedback for the generator
#     return {"messages": [HumanMessage(content=res.content)]}


# builder = StateGraph(State)
# builder.add_node("generate", generation_node)
# builder.add_node("reflect", reflection_node)
# builder.add_edge(START, "generate")


# def should_continue(state: State):
#     if len(state["messages"]) > 6:
#         # End after 3 iterations
#         return END
#     return "reflect"


# builder.add_conditional_edges("generate", should_continue)
# builder.add_edge("reflect", "generate")
# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "1"}}

# for event in graph.astream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Generate an essay on the topicality of The Little Prince and its message in modern life"
#             )
#         ],
#     },
#     config,
# ):
#     print(event)
#     print("---")