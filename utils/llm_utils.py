from typing import Optional
from langchain_openai import ChatOpenAI
from config.lm_config import lm_config


def get_llm_client(
    model: Optional[str] = None,
    json_mode: Optional[bool] = False
) -> ChatOpenAI:

    if not model:
        model = lm_config.llm_model

    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    # 关闭 Qwen3 推理模型的思考模式：
    # 思考会占用 output token，若 max_tokens 不够（如 1024），会出现
    # finish_reason=length，思考耗尽所有 token，导致真正答案一个字符都出不来（总长度为 0）。
    kwargs["extra_body"] = {"enable_thinking": False}

    client = ChatOpenAI(
        model=model,
        base_url=lm_config.base_url,
        api_key=lm_config.api_key,
        temperature=lm_config.llm_temperature,
        max_tokens=lm_config.llm_max_tokens,
        timeout=30,
        max_retries=2,
        **kwargs,
    )
    return client