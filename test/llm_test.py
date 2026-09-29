import os
from langchain.chat_models import init_chat_model

from config.lm_config import lm_config



model = init_chat_model(
    model=lm_config.vl_model,  # 或 "deepseek-v4-flash"
    model_provider="openai",  #DeepSeek 完全兼容 OpenAI Chat Completions 协议
    api_key=lm_config.api_key,
    base_url=lm_config.base_url,  # 注意加上 /v1
)



res = model.invoke('解释一下langchain是什么,简洁回答100字以内')

print(res)