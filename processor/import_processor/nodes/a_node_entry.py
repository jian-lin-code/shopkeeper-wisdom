from pathlib import Path

from modelscope.server.models import output

from processor.import_processor import state
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """
    name = "node_entry"

    def process(self, state: ImportGraphState):
        task_id = state.get("task_id","")
        add_running_task(task_id,self.name)

        # 1. 从state中获取文件
        import_file_path = state.get("import_file_path")
        # import_file_path = state["import_file_path"] #这样写的话字段不存在直接报错
        # 判断路径是否为空
        if not import_file_path:
            raise StateFieldError(field_name='import_file_path', expected_type=str)

        # 2. 转换Path标准化对象
        import_file_path_obj = Path(import_file_path)
        # 判断文件是否存在
        if not import_file_path_obj.exists(): #文件是否存在 → True / False
            raise FileProcessingError(message=f"文件{import_file_path_obj.name}不存在")

        # 3. 检查文件后缀
        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(message=f"该文件的后缀格式{import_file_path_obj.suffix}不支持")

        # 4. 获取上传文件的标题，更新到state中
        state["file_title"] = import_file_path_obj.stem

        # 5. 设置pdf转md存放目录
        # 项目根目录，推荐固定以项目根为基准，不要连续多个parent
        BASE_DIR = Path(__file__).parent.parent.parent.parent
        output = BASE_DIR / "output"
        # 自动创建目录
        output.mkdir(parents=True, exist_ok=True)
        state["file_dir"] = str(output)

        # 6. 返回state
        return state

    # 假设import_file_path是pdf文件路径这个节点跑完的state是：
    output_state = {
        "is_pdf_read_enabled": True,
        "pdf_path": r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\pdf\H3CLA2608室内无线网关用户手册-6W100-整本手册.pdf",
        "file_title": "H3CLA2608室内无线网关用户手册-6W100-整本手册",
        "file_dir": r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\output"
    }
