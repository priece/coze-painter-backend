from typing import Optional
from pydantic import BaseModel, Field


class GlobalState(BaseModel):
    """全局状态定义"""
    prompt: str = Field(..., description="用户输入的提示词")
    emotion: str = Field(default="", description="情绪描述")
    style: str = Field(default="", description="风格描述")
    formatted_prompt: str = Field(default="", description="LLM格式化后的文生图提示词")
    image_url: str = Field(default="", description="生成的图片对象存储URL")


class GraphInput(BaseModel):
    """工作流的输入"""
    prompt: str = Field(..., description="用户输入的提示词")
    emotion: str = Field(default="", description="情绪描述")
    style: str = Field(default="", description="风格描述")


class GraphOutput(BaseModel):
    """工作流的输出"""
    image_url: str = Field(..., description="生成的图片对象存储URL")


class FormatPromptInput(BaseModel):
    """格式化提示词节点的输入"""
    prompt: str = Field(..., description="用户输入的提示词")
    emotion: str = Field(default="", description="情绪描述")
    style: str = Field(default="", description="风格描述")


class FormatPromptOutput(BaseModel):
    """格式化提示词节点的输出"""
    formatted_prompt: str = Field(..., description="格式化后的文生图提示词")


class GenerateImageInput(BaseModel):
    """图片生成节点的输入"""
    formatted_prompt: str = Field(..., description="格式化后的文生图提示词")


class GenerateImageOutput(BaseModel):
    """图片生成节点的输出"""
    image_url: str = Field(..., description="生成的图片对象存储URL")
