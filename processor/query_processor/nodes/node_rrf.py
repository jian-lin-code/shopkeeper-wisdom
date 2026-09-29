import json
from typing import List, Tuple, Dict, Any

from accelerate.test_utils.scripts.test_script import init_state_check

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.task_utils import add_running_task


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

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

        # 1. 获取各路搜索的结果（排除网络搜索: reranK节点做）
        embedding_search_list = [
            doc.get('entity') for doc in (state.get('embedding_chunks') or []) if isinstance(doc, dict)
        ]
        hyde_embedding_search_list = [
            doc.get('entity') for doc in (state.get('hyde_embedding_chunks') or []) if isinstance(doc, dict)
        ]

        # 2. 为不同路的搜索结果设置不同的权重
        rrf_inputs = [
            (embedding_search_list, 1.0),
            (hyde_embedding_search_list, 1.0)
        ]

        # 3. 利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 4. 获取rrf_chunks（只取文档，不要分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        # 5. 更新state
        state['rrf_chunks'] = rrf_chunks

        return state

    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用 RRF 公式计算每一个文档的总得分
        :param rrf_inputs:  列表，每个元素是(各路的搜索结果列表, 权重)的元组
        :param k:           平滑参数(RFF常数)，通常取 60
        :param max_results: 合并完之后返回的文档数，None 表示全部
        :return:            合并以及排序后的文档列表，[(元素, RRF 得分), ...] 按得分降序
        """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}  # 存放所有 chunk 的文档数据

        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input, start=1): #enumerate(可迭代对象, start=起始序号)
                chunk_id = doc.get('chunk_id')
                # RRF 公式: score += weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)

                # 使用 setdefault 保留首次遇到的文档版本(只记录第一次)
                chunk_data.setdefault(chunk_id, doc)

        # 按得分降序排序
        unsorted_results = [(chunk_data[cid], score) for cid, score in chunk_scores.items()]
        sorted_results = sorted(
            unsorted_results,
            # 排序时看每个元素的第 2 个值（也就是分数）
            key=lambda x: x[1],
            reverse=True
        )

        # 等价写法
        # def get_score(item):
        #     return item[1]  # 返回分数
        # sorted_results = sorted(unsorted_results, key=get_score, reverse=True)

        # 动态截取前 max_results 条
        return sorted_results[:max_results] if max_results else sorted_results

if __name__ == "__main__":
    init_state = {
            "embedding_chunks":[
            {
                "chunk_id": 469230113564629588,
                "distance": 0.8155478239059448,
                "entity": {
                    "item_name": "H3CER2100企业级路由器",
                    "chunk_id": 469230113564629588,
                    "content": "# H3C ER2100 企业级路由器\n\n用户手册\n\nCopyright © 2009-2015 杭州华三通信技术有限公司及其许可者 版权所有，保留一切权利。未经本公司书面许可，任何单位和个人不得擅自摘抄、复制本书内容的部分或全部，并不得以任何形式传播。\n\nH3C、 、H3CS、H3CIE、H3CNE、Aolynk、 Aolynk 、H<sup>3</sup>Care、 Peare 、IRF、NetPilot、Netflow、SecEngine、SecPath、SecCenter、SecBlade、Comware、ITCMM、HUASAN、华三均为杭州华三通信技术有限公司的商标。对于本手册中出现的其它公司的商标、产品标识及商品名称，由各自权利人拥有。\n\n由于产品版本升级或其他原因，本手册内容有可能变更。H3C保留在没有任何通知或者提示的情况下对本手册的内容进行修改的权利。本手册仅作为使用指导，H3C尽全力在本手册中提供准确的信息，但是 H3C 并不确保手册内容完全没有错误，本手册中的所有陈述、信息和建议也不构成任何明示或暗示的担保。\n\n\n## 前 言\n\nH3C ER2100企业级路由器 用户手册将会详细地指导您如何通过 Web设置页面或命令行对设备进行本地管理。\n\n前言部分包含如下内容：\n\n• 读者对象\n\n• 本书约定\n\n资料获取方式\n\n• 技术支持\n\n资料意见反馈\n"
                }
            },
            {
                "chunk_id": 469230113564629595,
                "distance": 0.8080917596817017,
                "entity": {
                    "item_name": "H3CER2100企业级路由器",
                    "chunk_id": 469230113564629595,
                    "content": "## 2 <sub>产品概述</sub>\n\n本章节主要包含以下内容：\n\n产品简介\n\n• 主要特性\n\n典型组网应用\n\n\n## 2.1 产品简介\n\n感谢您选择了全新的 ER2100路由器（以下简称路由器）。\n\n它是华三通信技术有限公司（以下简称 H3C）全新推出的一款面向中小企业的高性能宽带路由器，采用专业的 64位网络处理器，同时配合 DDRII RAM进行高速转发，可以达到百兆线速转发。路由器支持丰富的软件特性，比如：ARP防攻击、流量限速、QQ/MSN应用限制、DDNS动态域名、IPSecVPN 等功能，并提供非常简便、易操作的 Web 设置页面，可以帮您快速地完成各功能特性需求的配置。\n\n![](images/9290b7a16318c10e7ae4991cc7abd37b3bb2fffc90e7cc2d8e5a25e0c70a1d08.jpg)\n\n\n## 说明\n\n本手册中所描述的功能特性规格可能随产品的升级而发生改变，恕不另行通知。详情您可以向 H3C公司市场人员或技术支援人员咨询获取。\n\n\n## 2.2 主要特性\n\n\n## 2.2.1 强大的功能特性\n\n• 高性能防火墙\n\n内置高性能防火墙，通过设置出站和入站通信策略来快速地实现访问控制。\n\n• 防攻击\n\n支持对来自因特网和内网的常见攻击进行防护。同时，内置内网异常流量防护模块，对局域网内各台主机的流量进行检查，并根据您所选择的防护等级（支持高、中、低三种）进行相应的处理，确保网络在遭受此类异常攻击时仍能正常工作。\n"
                }
            },
            {
                "chunk_id": 469230113564629592,
                "distance": 0.8059043288230896,
                "entity": {
                    "item_name": "H3CER2100企业级路由器",
                    "chunk_id": 469230113564629592,
                    "content": "## 目 录\n\n1 您想了解什么？……1-1\n2 产品概述……2-1\n2.1 产品简介……2-1\n2.2 主要特性……2-1\n2.2.1 强大的功能特性……2-1\n2.2.2 友好的用户界面……2-2\n2.2.3 丰富的统计诊断功能和管理方式……2-2\n2.3 典型组网应用……2-2\n3 登录Web设置页面……3-1\n3.1 准备工作……3-1\n3.1.1 管理计算机要求……3-1\n3.1.2 建立网络连接……3-1\n3.1.3 取消代理服务器……3-3\n3.2 登录路由器Web设置页面……3-4\n4 熟悉Web设置页面……4-1\n4.1 Web设置页面介绍……4-1\n4.2 常用页面控件介绍……4-1\n4.3 页面列表操作介绍……4-2\n4.4 Web用户超时处理……4-3\n4.5 退出Web设置页面……4-3\n5 接口设置……5-1\n5.1 设置WAN……5-1\n5.1.1 连接到因特网……5-1\n5.1.2 设置WAN口MAC地址克隆……5-3\n5.1.3 设置WAN口的速率和双工模式……5-3\n5.2 设置LAN……5-4\n5.2.1 修改LAN口的IP地址……5-4\n5.2.2 设置LAN口MAC地址克隆……5-5\n5.2.3 设置LAN口的基本属性……5-5\n5.2.4 设置本地端口镜像……5-6\n5.3 设置DHCP……5-7\n5.3.1 DHCP简介……5-7\n5.3.2 DHCP的IP地址分配……5-8\n\n5.3.3 设置DHCP服务器 …… 5-9\n5.3.4 设置DHCP静态表 …… 5-10\n5.3.5 显示和维护DHCP客户列表 …… 5-12\n\n6 安全专区 …… 6-1\n6.1 设置ARP安全 …… 6-1\n6.1.1 ARP简介 …… 6-1\n6.1.2 设置ARP绑定 …… 6-3\n6.1.3 设置ARP检测 …… 6-5\n6.1.4 设置发送免费ARP …… 6-5\n6.2 设置接入控制 …… 6-7\n6.2.1 设置MAC过滤 …… 6-7\n6.2.2 设置网站过滤 …… 6-8\n6.2.3 设置IPMAC过滤 …… 6-10\n6.3 设置防火墙 …… 6-11\n6.3.1 设置出站通信策略 …… 6-11\n6.3.2 设置入站通信策略 …… 6-13\n6.4 设置防攻击 …… 6-15\n6.4.1 防攻击方式 …… 6-15\n6.4.2 设置IDS防范 …… 6-15\n6.4.3 设置报文源认证 …… 6-16\n6.4.4 设置异常流量防护 …… 6-17\n\n7 设置IPSec VPN …… 7-1\n7.1 IPSec VPN简介 …… 7-1\n7.1.1 IPSec简介 …… 7-1\n7.1.2 IPSec VPN常见的组网模式 …… 7-3\n7.2 IPSec VPN设置方法 …… 7-4\n7.3 通过快速向导实现IPSec VPN …… 7-4\n7.4 通过高级设置实现IPSec VPN …… 7-6\n7.4.1 设置IKE …… 7-6\n7.4.2 设置IPSec …… 7-10\n7.4.3 查看VPN状态 …… 7-13\n\n8 设置QoS …… 8-1\n8.1 设置IP流量限制 …… 8-1\n8.2 设置网络连接限数 …… 8-3\n\n9 高级设置 …… 9-1\n9.1 设置网络连接参数 …… 9-1\n\n9.2 设置虚拟服务器....9-1\n9.3 设置端口触发....9-3\n9.4 设置ALG应用....9-4\n9.5 设置静态路由....9-5\n9.6 业务控制....9-7\n    9.6.1 限制使用IM软件....9-7\n    9.6.2 设置QQ特权号码....9-8\n    9.6.3 限制使用金融软件....9-8\n9.7 应用服务....9-9\n    9.7.1 设置DDNS....9-9\n    9.7.2 设置UPnP....9-10"
                }
            },
            {
                "chunk_id": 469230113564629681,
                "distance": 0.7983441352844238,
                "entity": {
                    "item_name": "H3CER2100企业级路由器",
                    "chunk_id": 469230113564629681,
                    "content": "## 说明\n\n此典型配置举例仅体现 H3C ER2100上的设置，且所涉及的设置均在 H3C ER2100缺省配置的基础上进行。如果您之前已经对 H3C ER2100上做过相应的设置，为了保证效果，请确保当前设置和以下设置不冲突。\n\n(1) 在管理计算机的 Web 浏览器地址栏中输入 http://192.168.1.1，回车。输入缺省的用户名、密码（缺省均为admin，区分大小写）以及验证码，单击<确定>按钮后便可进入Web设置页面\n\n![](images/cedb7d5216feeaae998d6c8c2a0748867e3b58eb22f422c8e8f554b9ab867d4e.jpg)\n\n(2) 选择“接口设置→WAN 设置→连接到因特网”，在“WAN网口”下拉框中选择“静态地址（手工配置地址）”选项。用电信提供的参数填写WAN口的上网参数，单击<应用>按钮生效\n\n(3) 选择“安全专区→ARP 安全→ARP检测”，设置IP地址搜索范围，单击<扫描>按钮开始搜索。待搜索完毕后，请确认搜索是否有遗漏（比如：查看搜索到的条目数是否与客户端的开机数一致）。如果没有遗漏，单击<全选>按钮选中所有的表项，再单击<绑定>按钮，将所有客户端主机的IP/MAC进行绑定即可；如果存在遗漏，您还可以选择“安全专区→ARP安全→ARP绑定”，手工添加ARP绑定项\n\n(4) 选择“安全专区→ARP安全→ARP防护”，选中“检测ARP攻击时，发送免费ARP报文”复选框，单击<应用>按钮生效\n\n(5) 选择“QoS 设置→流量管理→IP流量限制”。选中“启用IP流量限制”复选框和“允许每IP通道借用空闲的带宽”单选框，填写WAN口对应的带宽，单击<应用>按钮生效\n\n(6) 单击<新增>按钮，在弹出的对话框中设置IP流量限制规则：建议上行和下行流量的上限值均设置为300Kbps。同时，您也可以根据实际的网络情况对其进行适当地调整\n"
                }
            },
            {
                "chunk_id": 469230113564629684,
                "distance": 0.7943429350852966,
                "entity": {
                    "item_name": "H3CER2100企业级路由器",
                    "chunk_id": 469230113564629684,
                    "content": "## 说明\n\n本手册以通过 Console口登录路由器进行命令行管理为例。\n\n\n## 13.1 通过Console口搭建配置环境\n\n\n## 1. 连接管理计算机到路由器\n\n将管理计算机的串口通过配置线缆与路由器的 Console口相连。\n\n\n## 2. 配置管理计算机参数\n\n操作步骤如下（以 Windows XP系统为例）：\n\nWindows界面上单击[开始/所有程序/附件/通讯]，运行终端仿真程序。在“名称”文本框中键入新建连接的名称，比如：aaa，单击<确定>按钮\n\n(2) 在“连接时使用”下拉框中选择进行连接的串口（请确保选择的串口应与配置线缆实际连接的串口相一致），单击<确定>按钮\n\n\n## 连接到\n\n![](images/daf96fe3769caaef7cb567bec5db406d8c0fc35d8316eb730e8cc3fedd9522e1.jpg)\n\n(3) 在串口的属性对话框中设置相关参数，如右图所示，单击<确定>按钮\n\n确定\n\n每秒位数(B)：9600\n\n数据位 @):8\n\n停止位(S):\n\n数据流控制()：无\n\n还原为默认值B)\n\n确定\n\n取消\n\n(4) 选择[文件/属性/设置]，进入如右图所示的属性设置窗口。选择终端仿真类型为自动检测，单击<确定>按钮\n\n![](images/65adad785e84f0675231cb94cc9296aefad631711155b3e9efda2cc9d31aef15.jpg)\n\n(5) 回车后，即可登录路由器，且终端上显示命令行提示符<H3C>\n\n![](images/02cf96c1a1229a8d911813506d2145651c543e6f87dbd4b328a637f79ad9ad2b.jpg)\n"
                }
            }
        ],
            "hyde_embedding_chunks":[
            {
                "chunk_id": 469230113564629598,
                "distance": 0.8482054471969604,
                "entity": {
                    "chunk_id": 469230113564629598,
                    "content": "## 3 <sub>登录Web设置页面</sub>\n\n![](images/df928071c4ffd926af356fb15266cfcb2004fe6ea12e4a1e9bcca8a5733ecd3d.jpg)\n\n\n## 说明\n\n本小节仅介绍如何本地登录路由器的Web设置页面。如果您想实现远程登录路由器进行管理，需要先本地登录路由器，并开启其远程管理功能，相关的介绍请参见“10.3 远程管理”。\n\n本章节主要包含以下内容：\n\n• 准备工作\n\n登录路由器Web设置页面\n\n\n## 3.1 准备工作\n\n完成硬件安装后（安装过程可参见《H3C ER2100企业级路由器快速入门》），在登录路由器的 Web设置页面前，您需要确保管理计算机和网络满足一些基本要求。\n\n\n## 3.1.1 管理计算机要求\n\n请确认管理计算机已安装了以太网卡。\n\n\n## 3.1.2 建立网络连接\n\n1. 设置管理计算机的IP地址\n\n自动获取 IP地址（推荐使用）：请将管理计算机设置成“自动获得 IP地址”和“自动获得 DNS服务器地址”（计算机系统的缺省配置），由路由器自动为管理计算机分配 IP地址。\n\n设置静态 IP地址：请将管理计算机的 IP地址与路由器的 LAN口 IP地址设置在同一网段内（LAN 口缺省的 IP 地址为：192.168.1.1，子网掩码为 255.255.255.0）\n\n操作步骤如下（以 Windows XP系统为例）：\n\n(1) 单击屏幕左下角<开始>按钮进入[开始]菜单，选择“控制面板”。双击“网络连接”图标，再双击弹出的“本地连接”图标，弹出“本地连接状态”窗口\n\n(2) 单击<属性>按钮，进入“本地连接属性”窗口\n\n(3) 选中“Internet 协议（TCP/IP）”，单击<属性>按钮，进入“Internet协议（TCP/IP）属性”窗口。选择“使用下面的IP地址”单选按钮，输入IP地址（在 192.168.1.2～192.168.1.254 中任意值）、子网掩码（255.255.255.0）及默认网关（192.168.1.1），确定后完成操作\n\n![](images/3fe39eee742158c03103218ddf1a6d31606e8e54b6e22c32551e3f01b1808540.jpg)\n",
                    "item_name": "H3CER2100企业级路由器"
                }
            },
            {
                "chunk_id": 469230113564629681,
                "distance": 0.8397153615951538,
                "entity": {
                    "chunk_id": 469230113564629681,
                    "content": "## 说明\n\n此典型配置举例仅体现 H3C ER2100上的设置，且所涉及的设置均在 H3C ER2100缺省配置的基础上进行。如果您之前已经对 H3C ER2100上做过相应的设置，为了保证效果，请确保当前设置和以下设置不冲突。\n\n(1) 在管理计算机的 Web 浏览器地址栏中输入 http://192.168.1.1，回车。输入缺省的用户名、密码（缺省均为admin，区分大小写）以及验证码，单击<确定>按钮后便可进入Web设置页面\n\n![](images/cedb7d5216feeaae998d6c8c2a0748867e3b58eb22f422c8e8f554b9ab867d4e.jpg)\n\n(2) 选择“接口设置→WAN 设置→连接到因特网”，在“WAN网口”下拉框中选择“静态地址（手工配置地址）”选项。用电信提供的参数填写WAN口的上网参数，单击<应用>按钮生效\n\n(3) 选择“安全专区→ARP 安全→ARP检测”，设置IP地址搜索范围，单击<扫描>按钮开始搜索。待搜索完毕后，请确认搜索是否有遗漏（比如：查看搜索到的条目数是否与客户端的开机数一致）。如果没有遗漏，单击<全选>按钮选中所有的表项，再单击<绑定>按钮，将所有客户端主机的IP/MAC进行绑定即可；如果存在遗漏，您还可以选择“安全专区→ARP安全→ARP绑定”，手工添加ARP绑定项\n\n(4) 选择“安全专区→ARP安全→ARP防护”，选中“检测ARP攻击时，发送免费ARP报文”复选框，单击<应用>按钮生效\n\n(5) 选择“QoS 设置→流量管理→IP流量限制”。选中“启用IP流量限制”复选框和“允许每IP通道借用空闲的带宽”单选框，填写WAN口对应的带宽，单击<应用>按钮生效\n\n(6) 单击<新增>按钮，在弹出的对话框中设置IP流量限制规则：建议上行和下行流量的上限值均设置为300Kbps。同时，您也可以根据实际的网络情况对其进行适当地调整\n",
                    "item_name": "H3CER2100企业级路由器"
                }
            },
            {
                "chunk_id": 469230113564629684,
                "distance": 0.8298162817955017,
                "entity": {
                    "chunk_id": 469230113564629684,
                    "content": "## 说明\n\n本手册以通过 Console口登录路由器进行命令行管理为例。\n\n\n## 13.1 通过Console口搭建配置环境\n\n\n## 1. 连接管理计算机到路由器\n\n将管理计算机的串口通过配置线缆与路由器的 Console口相连。\n\n\n## 2. 配置管理计算机参数\n\n操作步骤如下（以 Windows XP系统为例）：\n\nWindows界面上单击[开始/所有程序/附件/通讯]，运行终端仿真程序。在“名称”文本框中键入新建连接的名称，比如：aaa，单击<确定>按钮\n\n(2) 在“连接时使用”下拉框中选择进行连接的串口（请确保选择的串口应与配置线缆实际连接的串口相一致），单击<确定>按钮\n\n\n## 连接到\n\n![](images/daf96fe3769caaef7cb567bec5db406d8c0fc35d8316eb730e8cc3fedd9522e1.jpg)\n\n(3) 在串口的属性对话框中设置相关参数，如右图所示，单击<确定>按钮\n\n确定\n\n每秒位数(B)：9600\n\n数据位 @):8\n\n停止位(S):\n\n数据流控制()：无\n\n还原为默认值B)\n\n确定\n\n取消\n\n(4) 选择[文件/属性/设置]，进入如右图所示的属性设置窗口。选择终端仿真类型为自动检测，单击<确定>按钮\n\n![](images/65adad785e84f0675231cb94cc9296aefad631711155b3e9efda2cc9d31aef15.jpg)\n\n(5) 回车后，即可登录路由器，且终端上显示命令行提示符<H3C>\n\n![](images/02cf96c1a1229a8d911813506d2145651c543e6f87dbd4b328a637f79ad9ad2b.jpg)\n",
                    "item_name": "H3CER2100企业级路由器"
                }
            },
            {
                "chunk_id": 469230113564629588,
                "distance": 0.8199946284294128,
                "entity": {
                    "chunk_id": 469230113564629588,
                    "content": "# H3C ER2100 企业级路由器\n\n用户手册\n\nCopyright © 2009-2015 杭州华三通信技术有限公司及其许可者 版权所有，保留一切权利。未经本公司书面许可，任何单位和个人不得擅自摘抄、复制本书内容的部分或全部，并不得以任何形式传播。\n\nH3C、 、H3CS、H3CIE、H3CNE、Aolynk、 Aolynk 、H<sup>3</sup>Care、 Peare 、IRF、NetPilot、Netflow、SecEngine、SecPath、SecCenter、SecBlade、Comware、ITCMM、HUASAN、华三均为杭州华三通信技术有限公司的商标。对于本手册中出现的其它公司的商标、产品标识及商品名称，由各自权利人拥有。\n\n由于产品版本升级或其他原因，本手册内容有可能变更。H3C保留在没有任何通知或者提示的情况下对本手册的内容进行修改的权利。本手册仅作为使用指导，H3C尽全力在本手册中提供准确的信息，但是 H3C 并不确保手册内容完全没有错误，本手册中的所有陈述、信息和建议也不构成任何明示或暗示的担保。\n\n\n## 前 言\n\nH3C ER2100企业级路由器 用户手册将会详细地指导您如何通过 Web设置页面或命令行对设备进行本地管理。\n\n前言部分包含如下内容：\n\n• 读者对象\n\n• 本书约定\n\n资料获取方式\n\n• 技术支持\n\n资料意见反馈\n",
                    "item_name": "H3CER2100企业级路由器"
                }
            },
            {
                "chunk_id": 469230113564629660,
                "distance": 0.7127511501312256,
                "entity": {
                    "chunk_id": 469230113564629660,
                    "content": "## Λ注意\n\n• 请您在软件升级之前备份路由器当前的设置信息。如果升级过程中出现问题，您可以用其来恢复到原来的设置。\n\n• 升级过程中请勿断开路由器的电源，否则可能会造成路由器不能正常工作。\n\n• 路由器升级成功后，将会重新启动。\n\n\n## 页面向导：设备管理→基本管理→软件升级\n\n单击页面上的“H3C的技术支持网站”链接下载对应产品的最新软件版本，保存到本地主机。然后，单击<浏览>按钮，选择相应的升级软件。最后，单击<升级>按钮，即可开始升级。\n\n![](images/9bee08124b8cf4908f6d773270ea4c57fb801facd1d4f22a415c3ffa62e312d2.jpg)\n\n说明\n\n软件升级后，您可以通过查看 基本信息页面中“软件版本”来验证当前运行的版本是否正确。\n\n\n## 10.1.4 重新启动路由器\n\n\n## 页面向导：设备管理→基本管理→重启动\n\n注意\n\n• 重新启动期间，请勿断开路由器的电源。\n\n• 重新启动期间，网络通信将暂时中断。\n\n单击页面上的<重启动>按钮，确认后，路由器重新启动。\n\n\n## 10.2 用户管理\n\n\n## 10.2.1 登录管理\n",
                    "item_name": "H3CER2100企业级路由器"
                }
            }
        ]
    }
    test = NodeRrf()

    response = test(init_state)

    json_response = json.dumps(response,ensure_ascii=False,indent=4)

    logger.info(f'json_response：{json_response}')