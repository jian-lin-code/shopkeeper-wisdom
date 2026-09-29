import asyncio
from turtledemo.penrose import start
from typing import Dict
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()

# ========== 配置CORS跨域 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境，全部允许；上线改成前端域名白名单
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 挂载静态资源目录 ==========
# 项目目录结构：
# main.py
# static/
#    index.html
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 全局存储：session_id -> asyncio.Queue
task_queues: Dict[str, asyncio.Queue] = {}


# 后台长任务：模拟工作流，不断往队列写消息
async def long_task(session_id: str):
    print("开始执行长耗时任务……")
    queue = asyncio.Queue()
    task_queues[session_id] = queue

    for i in range(1,11):
        await asyncio.sleep(1)
        msg = f"这是工作流的第{i}条执行结果"
        await queue.put(msg)

    # 放入None标记任务完成
    await queue.put(None)


# 触发任务接口
@app.get("/query/{session_id}")
async def query_by_session(session_id: str, backgroundTasks: BackgroundTasks):
    # 后台执行long_task
    backgroundTasks.add_task(long_task, session_id)
    return {"message": "任务已经开始，请耐心等待", "session_id": session_id}


# SSE消息生成器
async def event_generator(session_id: str):
    # 等待队列创建（后台任务可能还没初始化完）
    while session_id not in task_queues:
        await asyncio.sleep(0.1)
    queue = task_queues[session_id]

    while True:
        msg = await queue.get()
        if msg is None:
            # SSE结束，返回结束标记
            yield f"data: [DONE]\n\n"
            break
        # SSE标准格式：data: xxx\n\n
        yield f"data: {msg}\n\n"


# SSE流式接口
@app.get("/stream/{session_id}")
async def stream_by_session(session_id: str):
    return StreamingResponse(
        event_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=7001, reload=True)
