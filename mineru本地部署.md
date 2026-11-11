# shopkeeper-wisdom
FastAPI backend for shopkeeper wisdom, merchant knowledge base with RAG.

uv add python-dotenv    # 加载 .env 环境变量文件，管理密钥、配置等敏感信息
uv add numpy            # 科学计算核心库，提供高性能数组操作、线性代数等功能
uv add langgraph        # 基于LangChain的工作流框架，用于构建多智能体/状态管理的LLM应用
uv add grandalf         # 图布局与可视化库，支持图的绘制、路径计算等
uv add fastapi          # 高性能异步Web框架，用于快速构建API接口（支持自动生成文档）
uv add minio            # MinIO对象存储的Python客户端，用于操作S3兼容的存储服务
uv add langchain-openai # LangChain对接OpenAI API的专用模块（含GPT、Embeddings等）
uv add langchain        # LLM应用开发框架，整合各类大模型、工具、数据源
uv add langchain-text-splitters # LangChain的文本分割模块，用于切分长文本适配LLM上下文
uv add pymongo          # MongoDB数据库的Python驱动，用于连接和操作MongoDB
uv add colorlog         # 彩色日志输出库，让终端日志带颜色区分，提升调试与查看效率

#陆陆续续还有一些其他依赖，需要特别说明，我们在后续的课程中再做介绍和安装

#可能有用的命令
#查看虚拟环境中的依赖关系
uv tree

#根据pyproject.toml同步虚拟环境
uv sync

#强制根据pyproject.toml重新下载环境依赖
rm uv.
uv.lock
uv sync --reinstall   

#必要时清理uv缓存，重新下载所有的依赖
#uv缓存的位置由环境变量 UV_CACHE_DIR 指定 


mineru本地部署（PowerShell 里面执行）
1. uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
2. set MINERU_MODEL_SOURCE=modelscope   配置下载源（当前 CMD 窗口临时生效）
3. set MODELSCOPE_CACHE=D:\ai_models\mineru    模型存放目录
4. mineru-models-download  下载 

提示：`mineru[core]` / `mineru[all]` **会自动装上 torch，但是默认是 CPU 版本**
        CPU 版 torch 只能跑 CPU 推理，** GTX1660 想要 GPU 加速需要配置**
uv pip install torch torchvision torchaudio --torch-backend=cu128 --no-cache 


