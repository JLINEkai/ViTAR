from openai import OpenAI
import requests

# def test_client():
#     # 测试参数
#     api_key = "EMPTY"
#     api_base = "10.140.1.75:18901/v1"
    
#     try:
#         # 创建客户端
#         client = OpenAI(
#             api_key=api_key,
#             base_url=api_base,
#         )
        
#         # 测试获取模型列表
#         response = requests.get(f"{api_base}")
#         models = response.json()
        
#         # 打印结果
#         print("Client created successfully!")
#         print(f"Available models: {models['data']}")
#         return True
        
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         return False

# import math
# print(math.__file__)
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "10.140.1.75:18901/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(
    prompt="San Francisco is a",
    max_tokens=16
)
print("Completion result:", completion)
if __name__ == '__main__':
    test_client() 