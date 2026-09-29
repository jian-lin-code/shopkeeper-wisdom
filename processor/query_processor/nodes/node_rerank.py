import json
from typing import List, Dict, Any


from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents
from utils.task_utils import add_running_task

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 3 #总数最少条数

# 断崖阈值（绝对，判断高分文档）
RERANK_GAP_ABS: float = 0.5
# 断崖阈值（相对，判断低分文档）
RERANK_GAP_RATIO: float = 0.25


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

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

        """
        执行重排序
        流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
        :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
        :return: 更新后的 state，包含 reranked_docs
        """

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)

        # 2. Rerank 精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态 Top_K 截取(断崖检测)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state['reranked_docs'] = cutoff_docs

        # 5. 返回state
        return state

    # 1. 合并多源文档
    def _step_1_merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""

        final_docs = []

        # 1. 获取本地 RRF 的文档
        for rrf_doc in state.get('rrf_chunks'):
            format_rrf_doc = {
                "content": rrf_doc.get('content'),
                "title": rrf_doc.get('title'),
                "chunk_id": rrf_doc.get('chunk_id'),
                "url": None,
                "source": "local"
            }
            final_docs.append(format_rrf_doc)

        # 2. 获取 api 远程的文档
        for web_doc in state.get('web_search_docs'):
            format_web_doc = {
                "content": web_doc.get('snippet'),
                "title": web_doc.get('title'),
                "chunk_id": None,
                "url": web_doc.get('url'),
                "source": "api"
            }
            final_docs.append(format_web_doc)

        return final_docs

    # 2.Rerank精排(精排打分)
    def _step_2_rerank_merged_docs(self, state: QueryGraphState, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 Reranker 模型对文档进行精排"""

        try:
            user_query = state.get('rewritten_query')
            # 获取文档列表的conten字段组成列表
            contents = [doc.get("content") for doc in merged_multi_docs]
            # 调用Rerank模型：交叉编码器（精排阶段）
            # Query 和 Document 联合编码，精度更高
            rerank_scores = rerank_documents(user_query, contents)

            scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            # 等同如下写法
            # scored_docs = []
            # for doc, score in zip(merged_multi_docs, rerank_scores):
            #     scored_docs.append({
            #         "content": doc.get("content"),
            #         "title": doc.get("title"),
            #         "chunk_id": doc.get("chunk_id"),
            #         "url": doc.get("url"),
            #         "source": doc.get("source"),
            #         "score": float(score),
            #     })

            sorted_score_docs = sorted(
                scored_docs,
                key=lambda x: x["score"],
                reverse=True
            )

            return sorted_score_docs

        except Exception as e:
            logger.error(f"Rerank 重排序失败: {str(e)}")
            return [{**merged_multi_docs, "score": None}]

    # 3. 动态 Top_K 截取(断崖检测)
    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not ranked_docs:
            return []

        upper_bound = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower_bound = min(RERANK_MIN_TOPK, upper_bound)

        # 默认值：取满硬上限（最多10条）
        cutoff_pos = upper_bound

        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距）
        for idx in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[idx].get("score")
            next_score = ranked_docs[idx + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            abs_gap = current_score - next_score
            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                cutoff_pos = idx + 1
                logger.info(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                # logger.debug(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break

        return ranked_docs[:cutoff_pos]

if __name__ == "__main__":
    test = NodeRerank()

    init_state = {
        "rrf_chunks": [
            {
                "item_name": "H3CER2100企业级路由器",
                "chunk_id": 469230113564629588,
                "content": "# H3C ER2100 企业级路由器\n\n用户手册\n\nCopyright © 2009-2015 杭州华三通信技术有限公司及其许可者 版权所有，保留一切权利。未经本公司书面许可，任何单位和个人不得擅自摘抄、复制本书内容的部分或全部，并不得以任何形式传播。\n\nH3C、 、H3CS、H3CIE、H3CNE、Aolynk、 Aolynk 、H<sup>3</sup>Care、 Peare 、IRF、NetPilot、Netflow、SecEngine、SecPath、SecCenter、SecBlade、Comware、ITCMM、HUASAN、华三均为杭州华三通信技术有限公司的商标。对于本手册中出现的其它公司的商标、产品标识及商品名称，由各自权利人拥有。\n\n由于产品版本升级或其他原因，本手册内容有可能变更。H3C保留在没有任何通知或者提示的情况下对本手册的内容进行修改的权利。本手册仅作为使用指导，H3C尽全力在本手册中提供准确的信息，但是 H3C 并不确保手册内容完全没有错误，本手册中的所有陈述、信息和建议也不构成任何明示或暗示的担保。\n\n\n## 前 言\n\nH3C ER2100企业级路由器 用户手册将会详细地指导您如何通过 Web设置页面或命令行对设备进行本地管理。\n\n前言部分包含如下内容：\n\n• 读者对象\n\n• 本书约定\n\n资料获取方式\n\n• 技术支持\n\n资料意见反馈\n"
            },
            {
                "item_name": "H3CER2100企业级路由器",
                "chunk_id": 469230113564629681,
                "content": "## 说明\n\n此典型配置举例仅体现 H3C ER2100上的设置，且所涉及的设置均在 H3C ER2100缺省配置的基础上进行。如果您之前已经对 H3C ER2100上做过相应的设置，为了保证效果，请确保当前设置和以下设置不冲突。\n\n(1) 在管理计算机的 Web 浏览器地址栏中输入 http://192.168.1.1，回车。输入缺省的用户名、密码（缺省均为admin，区分大小写）以及验证码，单击<确定>按钮后便可进入Web设置页面\n\n![](images/cedb7d5216feeaae998d6c8c2a0748867e3b58eb22f422c8e8f554b9ab867d4e.jpg)\n\n(2) 选择“接口设置→WAN 设置→连接到因特网”，在“WAN网口”下拉框中选择“静态地址（手工配置地址）”选项。用电信提供的参数填写WAN口的上网参数，单击<应用>按钮生效\n\n(3) 选择“安全专区→ARP 安全→ARP检测”，设置IP地址搜索范围，单击<扫描>按钮开始搜索。待搜索完毕后，请确认搜索是否有遗漏（比如：查看搜索到的条目数是否与客户端的开机数一致）。如果没有遗漏，单击<全选>按钮选中所有的表项，再单击<绑定>按钮，将所有客户端主机的IP/MAC进行绑定即可；如果存在遗漏，您还可以选择“安全专区→ARP安全→ARP绑定”，手工添加ARP绑定项\n\n(4) 选择“安全专区→ARP安全→ARP防护”，选中“检测ARP攻击时，发送免费ARP报文”复选框，单击<应用>按钮生效\n\n(5) 选择“QoS 设置→流量管理→IP流量限制”。选中“启用IP流量限制”复选框和“允许每IP通道借用空闲的带宽”单选框，填写WAN口对应的带宽，单击<应用>按钮生效\n\n(6) 单击<新增>按钮，在弹出的对话框中设置IP流量限制规则：建议上行和下行流量的上限值均设置为300Kbps。同时，您也可以根据实际的网络情况对其进行适当地调整\n"
            },
            {
                "item_name": "H3CER2100企业级路由器",
                "chunk_id": 469230113564629684,
                "content": "## 说明\n\n本手册以通过 Console口登录路由器进行命令行管理为例。\n\n\n## 13.1 通过Console口搭建配置环境\n\n\n## 1. 连接管理计算机到路由器\n\n将管理计算机的串口通过配置线缆与路由器的 Console口相连。\n\n\n## 2. 配置管理计算机参数\n\n操作步骤如下（以 Windows XP系统为例）：\n\nWindows界面上单击[开始/所有程序/附件/通讯]，运行终端仿真程序。在“名称”文本框中键入新建连接的名称，比如：aaa，单击<确定>按钮\n\n(2) 在“连接时使用”下拉框中选择进行连接的串口（请确保选择的串口应与配置线缆实际连接的串口相一致），单击<确定>按钮\n\n\n## 连接到\n\n![](images/daf96fe3769caaef7cb567bec5db406d8c0fc35d8316eb730e8cc3fedd9522e1.jpg)\n\n(3) 在串口的属性对话框中设置相关参数，如右图所示，单击<确定>按钮\n\n确定\n\n每秒位数(B)：9600\n\n数据位 @):8\n\n停止位(S):\n\n数据流控制()：无\n\n还原为默认值B)\n\n确定\n\n取消\n\n(4) 选择[文件/属性/设置]，进入如右图所示的属性设置窗口。选择终端仿真类型为自动检测，单击<确定>按钮\n\n![](images/65adad785e84f0675231cb94cc9296aefad631711155b3e9efda2cc9d31aef15.jpg)\n\n(5) 回车后，即可登录路由器，且终端上显示命令行提示符<H3C>\n\n![](images/02cf96c1a1229a8d911813506d2145651c543e6f87dbd4b328a637f79ad9ad2b.jpg)\n"
            },
            {
                "chunk_id": 469230113564629598,
                "content": "## 3 <sub>登录Web设置页面</sub>\n\n![](images/df928071c4ffd926af356fb15266cfcb2004fe6ea12e4a1e9bcca8a5733ecd3d.jpg)\n\n\n## 说明\n\n本小节仅介绍如何本地登录路由器的Web设置页面。如果您想实现远程登录路由器进行管理，需要先本地登录路由器，并开启其远程管理功能，相关的介绍请参见“10.3 远程管理”。\n\n本章节主要包含以下内容：\n\n• 准备工作\n\n登录路由器Web设置页面\n\n\n## 3.1 准备工作\n\n完成硬件安装后（安装过程可参见《H3C ER2100企业级路由器快速入门》），在登录路由器的 Web设置页面前，您需要确保管理计算机和网络满足一些基本要求。\n\n\n## 3.1.1 管理计算机要求\n\n请确认管理计算机已安装了以太网卡。\n\n\n## 3.1.2 建立网络连接\n\n1. 设置管理计算机的IP地址\n\n自动获取 IP地址（推荐使用）：请将管理计算机设置成“自动获得 IP地址”和“自动获得 DNS服务器地址”（计算机系统的缺省配置），由路由器自动为管理计算机分配 IP地址。\n\n设置静态 IP地址：请将管理计算机的 IP地址与路由器的 LAN口 IP地址设置在同一网段内（LAN 口缺省的 IP 地址为：192.168.1.1，子网掩码为 255.255.255.0）\n\n操作步骤如下（以 Windows XP系统为例）：\n\n(1) 单击屏幕左下角<开始>按钮进入[开始]菜单，选择“控制面板”。双击“网络连接”图标，再双击弹出的“本地连接”图标，弹出“本地连接状态”窗口\n\n(2) 单击<属性>按钮，进入“本地连接属性”窗口\n\n(3) 选中“Internet 协议（TCP/IP）”，单击<属性>按钮，进入“Internet协议（TCP/IP）属性”窗口。选择“使用下面的IP地址”单选按钮，输入IP地址（在 192.168.1.2～192.168.1.254 中任意值）、子网掩码（255.255.255.0）及默认网关（192.168.1.1），确定后完成操作\n\n![](images/3fe39eee742158c03103218ddf1a6d31606e8e54b6e22c32551e3f01b1808540.jpg)\n",
                "item_name": "H3CER2100企业级路由器"
            },
            {
                "item_name": "H3CER2100企业级路由器",
                "chunk_id": 469230113564629595,
                "content": "## 2 <sub>产品概述</sub>\n\n本章节主要包含以下内容：\n\n产品简介\n\n• 主要特性\n\n典型组网应用\n\n\n## 2.1 产品简介\n\n感谢您选择了全新的 ER2100路由器（以下简称路由器）。\n\n它是华三通信技术有限公司（以下简称 H3C）全新推出的一款面向中小企业的高性能宽带路由器，采用专业的 64位网络处理器，同时配合 DDRII RAM进行高速转发，可以达到百兆线速转发。路由器支持丰富的软件特性，比如：ARP防攻击、流量限速、QQ/MSN应用限制、DDNS动态域名、IPSecVPN 等功能，并提供非常简便、易操作的 Web 设置页面，可以帮您快速地完成各功能特性需求的配置。\n\n![](images/9290b7a16318c10e7ae4991cc7abd37b3bb2fffc90e7cc2d8e5a25e0c70a1d08.jpg)\n\n\n## 说明\n\n本手册中所描述的功能特性规格可能随产品的升级而发生改变，恕不另行通知。详情您可以向 H3C公司市场人员或技术支援人员咨询获取。\n\n\n## 2.2 主要特性\n\n\n## 2.2.1 强大的功能特性\n\n• 高性能防火墙\n\n内置高性能防火墙，通过设置出站和入站通信策略来快速地实现访问控制。\n\n• 防攻击\n\n支持对来自因特网和内网的常见攻击进行防护。同时，内置内网异常流量防护模块，对局域网内各台主机的流量进行检查，并根据您所选择的防护等级（支持高、中、低三种）进行相应的处理，确保网络在遭受此类异常攻击时仍能正常工作。\n"
            },
            {
                "item_name": "H3CER2100企业级路由器",
                "chunk_id": 469230113564629595,
                "content": "XIXI"
            }

        ],
        "web_search_docs": [
        {
            "title": "H3C ER2100企业级路由器 用户手册-6W104",
            "url": "https://www.h3c.com/cn/d_201509/892076_30005_0.htm",
            "snippet": "H3C ER2100企业级路由器 用户手册-6W104 01-正文 1 您想了解什么? <table><tr><th>如果您想?</th><th>您可以查看</th></tr><tr><td>初识产品的大致形态、业务特性或者它在实际网络应用中的定位</td><td>“ 产品 概述 ”</td></tr><tr><td>通过搭建Web环境来管理设备,同时想进一步熟悉其设置页面</td><td>“ 登录Web设置页面 ”和“ 熟悉Web设置页面 ”</td></tr><tr><td>通过Web设置页面来设置设备WAN口、LAN口的相关参数及DHCP功能</td><td>“ 接口 设置 ”</td></tr><tr><td>通过Web设置页面来实现设备及网络环境的安全性,比如:ARP安全、接入控制、防火墙等</td><td>“ 安全专区 ”</td></tr><tr><td>通过Web设置页面来实现设备IPSec VPN功能</td><td>“ 设置IPSec VPN ”</td></tr><tr><td>通过Web设置页面来设置设备WAN口的带宽、IP流量限制、网络连接限数等</td><td>“ 设置QoS ”</td></tr><tr><td>通过Web设置页面来实现设备的高级业务功能,比如:虚拟服务器、业务控制、静态路由等</td><td>“ 高级设置 ”</td></tr><tr><td>通过Web设置页面对设备进行维护管理,比如:软件升级、用户管理、SNMP等</td><td>“ 设备管理 ”</td></tr><tr><td>通过Web设置页面对设备当前的设置状态进行查询或对系统运行情况进行监控等</td><td>“ 系统监控 ”</td></tr><tr><td>通过具体的典型组网举例来进一步理解设备的关键特性</td><td>“ 典型 组网 配置 举例 ”</td></tr><tr><td>通过命令行来简单地维护设备</td><td>“ 附录- 命令行设置 ”</td></tr><tr><td>定位或排除使用设备过程中遇到的问题</td><td>“ 附录- 故障排除 ”</td></tr><tr><td>获取设备重要的缺省出厂配置信息</td><td>“ 附录- 缺省设置 ”</td></tr></table> 2 产品概述 本章节主要包含以下内容: 产品简介 主要特性 · 典型组网应用1 产品简介 感谢您选择了全新的ER2100路由器(以下简称路由器)。 它是华三通信技术有限公司(以下简称H3C)全新推出的一款面向中小企业的高性能宽带路由器,采用专业的64位网络处理器,同时配合DDRII RAM进行高速转发,可以达到百兆线速转发。路由器支持丰富的软件特性,比如:ARP防攻击、流量限速、QQ/MSN应用限制、DDNS动态域名、IPSec VPN等功能,并提供非常简便、易操作的Web设置页面,可以帮您快速地完成各功能特性需求的配置。 本手册中所描述的功能特性规格可能随产品的升级而发生改变,恕不另行通知。详情您可以向H3C公司市场人员或技术支援人员咨询获取。 2."
        },
        {
            "title": "H3C ER2100V2企业级路由器-新华三集团-H3C",
            "url": "https://www.h3c.com/cn/Service/Document_Software/Document_Center/Routers/Catalog/ER/ER2100V2/",
            "snippet": "H3C ER2100V2企业级路由器 快速配置指南 H3C ER2100系列增强型企业级路由器 快速入门-6PW104 2017-03-31 类型: 快速配置指南 运行环境要求 H3C室内安装类设备运行环境要求-6W101 2026-01-27 类型: 运行环境要求 电源手册 H3C外部交流电源线选用手册-6W102 2024-08-02 类型: 电源手册 用户手册 H3C ER2100系列增强型企业级路由器 用户手册-6W103 2017-05-09 类型: 用户手册 典型配置 H3C ER系列路由器L2TP VPN 典型配置举例-6W101 2017-05-23 类型: 典型配置 H3C个人资料库 H3C个人资料库 2022-01-29 类型: H3C个人资料库"
        },
        {
            "title": "H3C ER2100",
            "url": "https://baike.baidu.com/item/H3C%20ER2100/10100684",
            "snippet": "H3C ER2100是新华三集团推出的企业级路由器,定位于中小企业的高性能网络需求,属于H3C智能联接解决方案体系。该产品采用MIPS 64位网络处理器,搭配64MB DDRII内存与8MBFlash内存,配有1个WAN接口、4个LAN接口以及Console控制端口。 硬件支持百兆线速转发,内置IPSec VPN功能可建立10条并发安全连接。安全模块包含状态防火墙、防DDoS攻击及ARP病毒双重防护机制(静态ARP绑定与DHCP授权ARP技术)。"
        }
    ],
        "rewritten_query": " H3C ER2100 企业级路由器怎么用呀"
    }
    response = test(init_state)

    json_response = json.dumps(response,ensure_ascii=False,indent=4)

    logger.info(f'json_response：{json_response}')