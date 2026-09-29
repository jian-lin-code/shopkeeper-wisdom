from config.mineru_config import mineru_config
print(mineru_config)

class Person:
    def __init__(self,name):
        self.name = name

    def __repr__(self):
        # 手动定义打印输出的字符串
        return f"Person(name={repr(self.name)})"
p = Person(name="123")
print(p)


for i in range(0,10,5):
    print(i)