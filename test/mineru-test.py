import subprocess
from pathlib import Path

# 执行 mineru raw 命令
BASE_DIR = Path(__file__).parent.parent #取当前脚本所在目录的上级
# print(BASE_DIR)
output_dir = BASE_DIR / "output"
cmd = [
    "mineru",
    "-p",
    r"C:\Users\xiaojianlin\Desktop\project\shopkeeper-wisdom\pdf\H3CLA2608室内无线网关用户手册-6W100-整本手册.pdf",
    "-o",
    str(output_dir),
    "--backend",
    "pipeline"
]

result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

print("stdout:", result.stdout)
print("stderr:", result.stderr) # 这里会拿到报错信息
