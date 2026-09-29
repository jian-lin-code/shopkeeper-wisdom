from pymongo import MongoClient

# 连接URI，authSource=admin 认证库，和你mongosh登录一致
uri = "mongodb://root:123456@172.19.46.73:27017/?authSource=admin"

# 创建客户端
client = MongoClient(uri)

try:
    # 测试连接
    client.admin.command("ping")
    print("✅ MongoDB连接成功")

    # 选择数据库 test
    db = client["test"]
    # 选择集合 student
    coll = db["student"]

    # ========== CRUD示例 ==========
    # 1. 插入单条
    res = coll.insert_one({"name": "张三", "age": 20})
    print(f"插入id: {res.inserted_id}")

    # 2. 查询一条
    doc = coll.find_one({"name": "张三"})
    print("查询结果：", doc)

    # 3. 更新一条
    coll.update_one({"name": "张三"}, {"$set": {"age": 21}})

    # 4. 查询全部
    for item in coll.find():
        print(item)

    # 5. 删除一条
    # coll.delete_one({"name": "张三"})

except Exception as e:
    print("❌ 连接失败：", e)
finally:
    client.close()
