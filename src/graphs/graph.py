from langgraph.graph import StateGraph, END
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput
)
from graphs.node import (
    format_prompt_node,
    generate_image_node
)

# 创建状态图，指定工作流的入参和出参
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

# 添加节点
builder.add_node("format_prompt", format_prompt_node, metadata={"type": "agent", "llm_cfg": "config/format_prompt_llm_cfg.json"})
builder.add_node("generate_image", generate_image_node)

# 设置入口点
builder.set_entry_point("format_prompt")

# 添加边
builder.add_edge("format_prompt", "generate_image")
builder.add_edge("generate_image", END)

# 编译图
main_graph = builder.compile()
