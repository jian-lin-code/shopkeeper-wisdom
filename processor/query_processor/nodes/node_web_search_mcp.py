import asyncio
import json

from Demos.mmapfile_demo import page_size

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
import httpx2
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from config.bailian_mcp_config import mcp_config
from tool.logger import logger
from utils.task_utils import add_running_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 添加节点运行中
        session_id = state.get("session_id","")
        is_stream = state.get("is_stream",False)
        add_running_task(session_id,self.name,is_stream)

        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        query = state.get("rewritten_query", "")
        docs = []

        try:
            result = asyncio.run(self._mcp_call(query))
            if not result:
                logger.warning("web_search 返回内容为空")
                return {"web_search_docs": docs}

            json_text = getattr(result[0], "text", "")
            result_json_obj = json.loads(json_text)
            pages = result_json_obj["pages"]
            for item in pages:
                if not isinstance(item, dict):
                    continue
                snippet = (item.get("snippet") or item.get("text") or item.get("content") or "").strip()
                url = (item.get("url") or item.get("link") or "").strip()
                title = (item.get("title") or "").strip()
                docs.append({"title": title, "url": url, "snippet": snippet})
        except Exception as e:
            # 联网搜索失败不应阻塞主链路，降级为空结果
            logger.error(f"web_search 解析结果失败: {e}", exc_info=True)

        return {"web_search_docs": docs}


    async def _mcp_call(self, query):
        api_key = mcp_config.api_key
        mcp_url = mcp_config.mcp_base_url

        # ✅ 必须创建 httpx2.AsyncClient 实例，headers放在这里
        http_client = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx2.Timeout(30)
        )

        async with streamable_http_client(mcp_url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                # print("可用工具：", [t.name for t in tools.tools])
                res = await session.call_tool(name="bailian_web_search",arguments={"query": query, "count": 3})
                return res.content

if __name__ == "__main__":
    test = NodeWebSearchMcp()
    init_state = {
        "rewritten_query":" H3C ER2100 企业级路由器怎么用呀",
    }
    response = test(init_state)

    json_response = json.dumps(response,ensure_ascii=False,indent=4)

    logger.info(f'json_response：{json_response}')

