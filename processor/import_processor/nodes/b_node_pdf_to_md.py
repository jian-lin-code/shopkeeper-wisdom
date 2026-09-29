import shutil
import time
import zipfile
from pathlib import Path
from loguru import logger
import requests
from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from utils.task_utils import add_running_task


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        task_id = state.get("task_id","")
        add_running_task(task_id,self.name)

        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # 步骤3：下载ZIP包并提取MD文件
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 步骤4：读取md的内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 步骤5：更新state状态
        state["md_path"] = md_path
        state["md_content"] = md_content

        return state

    def _step_1_validate_paths(self,state:ImportGraphState):

        # 1. 校验路径
        pdf_path = state.get("pdf_path","")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path",expected_type=str)
        file_dir = state.get("file_dir","")
        if not file_dir:
            raise StateFieldError(field_name="file_dir",expected_type=str)

        # 2. 封装路径
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3. 文档是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{pdf_path}")
        if not file_dir_obj.exists():
            raise FileProcessingError(message=f"输出文件不存在:{file_dir}")

        return pdf_path_obj,file_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj:Path):
        # 1. 从MinerU服务器获取上传链接
        base_url = mineru_config.base_url
        api_token = mineru_config.api_token
        url = f"{base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        # logger.info(f"url: {url}")
        # logger.info(f"header: {header}")
        # logger.info(f"data: {data}")

        # 获取上传url和任务的batch_id
        response = requests.post(url, headers=header, json=data)
        # 对响应结果进行校验
        # 先校验http状态
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")
        # 校验业务码
        result = response.json()
        if result.get("code") != 0:
            raise PdfConversionError(f"获取上传链接失败：返回数据：{result}")
        # 获取响应结果
        signed_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"]
        print('signed_url:',signed_url)
        print('batch_id',batch_id)

        # 2. 文件上传
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(signed_url, data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")

            self.logger.info(f"文件上传成功！")

        # 3. 批量获取任务结果
        poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()  # 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔时间
        self.logger.info(f"【任务轮询】最大超时：{timeout_seconds}s，batch_id：{batch_id}")
        # 4. 根据batch_id轮询任务状态直到成功"done"
        while True:
            # 已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            # 发起轮询请求，短超时10秒，异常则重试
            try:
                res_poll = requests.get(url=poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}，bactch_id：{batch_id}")
                time.sleep(poll_interval)
                continue

            # 处理HTTP响应错误
            if res_poll.status_code != 200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            # 解析轮询结果，校验业务状态
            poll_data = res_poll.json()
            if poll_data["code"] != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]

            # 获取结果
            result_item = extract_results[0]
            data_state = result_item["state"]

            # 状态为 done
            if data_state == "done":
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，bactch_id：{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，bactch_id：{batch_id}")

                return full_zip_url

            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        # 1、下载ZIP包
        self.logger.info(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")
        response = requests.get(zip_url)
        # 对响应结果进行校验
        if response.status_code != 200:
            raise RuntimeError(f"【ZIP下载】ZIP包下载失败：状态码：{response.status_code}，响应结果：{response}")
        # 拼接ZIP包保存路径并保存
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)
        self.logger.info(f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}")

        # 2. 如果目标文件夹已存在，先删除（确保环境干净）
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 3、创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 4、解压
        self.logger.info(f"【ZIP解压】开始解压ZIP包：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        # 5、重命名
        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())

    # 假设import_file_path是pdf文件路径这个节点跑完的state是：
    output_state = {
        "is_pdf_read_enabled": True,
        "pdf_path": r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\pdf\H3CLA2608室内无线网关用户手册-6W100-整本手册.pdf",
        "file_title": "H3CLA2608室内无线网关用户手册-6W100-整本手册",
        "file_dir": r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\output",
        "md_path": r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\output\H3CLA2608室内无线网关用户手册-6W100-整本手册\H3CLA2608室内无线网关用户手册-6W100-整本手册.md",
        "md_content": """用户手册 Copyright © 2014 杭州华三通信技术有限公司及其许可者 版权所有，保留一切权利。此处省略..."""
    }
if __name__ == '__main__':
    test = NodePDFToMD()
    init_state = {
        "pdf_path": r"D:\BaiduNetdiskDownload\掌柜智库课件0525\2.资料\04-设备手册汇总\doc\万用表RS-12的使用.pdf",
        "file_dir": r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\output"
    }
    res = test(init_state)

    logger.info(f"res: {res}")

