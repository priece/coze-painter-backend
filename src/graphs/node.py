import os
import json
import base64
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from coze_coding_dev_sdk import LLMClient, ImageGenerationClient
from coze_coding_dev_sdk.s3 import S3SyncStorage

from graphs.state import (
    FormatPromptInput,
    FormatPromptOutput,
    GenerateImageInput,
    GenerateImageOutput
)


def format_prompt_node(
    state: FormatPromptInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> FormatPromptOutput:
    """
    title: 格式化提示词
    desc: 使用LLM将用户的提示词、情绪和风格格式化为结构化的文生图提示词
    integrations: 大语言模型
    """
    ctx = runtime.context

    # 从配置文件读取LLM配置
    cfg_file = os.path.join(os.getenv("COZE_WORKSPACE_PATH"), config['metadata']['llm_cfg'])
    with open(cfg_file, 'r', encoding='utf-8') as fd:
        _cfg = json.load(fd)

    llm_config = _cfg.get("config", {})
    sp = _cfg.get("sp", "")
    up = _cfg.get("up", "")

    # 使用jinja2模板渲染用户提示词
    up_tpl = Template(up)
    user_prompt_content = up_tpl.render({
        "prompt": state.prompt,
        "emotion": state.emotion,
        "style": state.style
    })

    # 调用大语言模型
    client = LLMClient(ctx=ctx)

    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=user_prompt_content)
    ]

    response = client.invoke(
        messages=messages,
        model=llm_config.get("model", "doubao-seed-1-6-251015"),
        temperature=llm_config.get("temperature", 0.7),
        top_p=llm_config.get("top_p", 0.9),
        max_tokens=llm_config.get("max_tokens", 1000)
    )

    # 安全处理响应内容
    if isinstance(response.content, str):
        formatted_prompt = response.content.strip()
    else:
        text_parts = [
            item.get("text", "") for item in response.content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        formatted_prompt = " ".join(text_parts).strip()

    return FormatPromptOutput(formatted_prompt=formatted_prompt)


def generate_image_node(
    state: GenerateImageInput,
    config: RunnableConfig,
    runtime: Runtime[Context]
) -> GenerateImageOutput:
    """
    title: 生成图片
    desc: 使用结构化的文生图提示词生成高质量图片并保存到对象存储
    integrations: 生图大模型, 对象存储
    """
    ctx = runtime.context

    # 调用图片生成模型
    client = ImageGenerationClient(ctx=ctx)

    response = client.generate(
        prompt=state.formatted_prompt,
        size="2K",
        watermark=False
    )

    # 检查响应是否成功
    if not response.success:
        error_messages = response.error_messages if hasattr(response, 'error_messages') else []
        raise Exception(f"图片生成失败: {error_messages}")

    # 获取图片URL并转存到对象存储
    if hasattr(response, 'image_urls') and response.image_urls:
        image_url = response.image_urls[0]
        
        # 初始化对象存储客户端
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        
        # 使用upload_from_url直接从URL转存到对象存储
        try:
            object_key = storage.upload_from_url(
                url=image_url,
                bucket=None,
                timeout=30
            )
        except Exception as e:
            raise Exception(f"图片上传到对象存储失败: {e}")
        
        # 生成签名URL用于访问
        try:
            signed_url = storage.generate_presigned_url(
                key=object_key,
                bucket=None,
                expire_time=3600  # 1小时有效期
            )
        except Exception as e:
            raise Exception(f"生成访问URL失败: {e}")
    else:
        raise Exception(f"未能获取到生成的图片URL")

    if not signed_url:
        raise Exception("生成的图片URL为空")

    return GenerateImageOutput(image_url=signed_url)
