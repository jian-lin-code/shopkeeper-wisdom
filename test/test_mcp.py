import asyncio
import httpx2
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from config.bailian_mcp_config import mcp_config

async def test():
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
            print("可用工具：", [t.name for t in tools.tools])
            res = await session.call_tool(name="bailian_web_search", arguments={"query":"mcp协议是什么？","count":3})
            for block in res.content:
                print(block.text)

if __name__ == "__main__":
    asyncio.run(test())
