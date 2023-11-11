import os
import base64
import time
import json
import random
import string
from qcloud_cos import CosConfig, CosS3Client

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 sd_env.json 文件的绝对路径
env_path = os.path.join(script_dir, 'sd_env.json')

with open(env_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data)

# 配置信息
config = CosConfig(
    Region='ap-shanghai',
    SecretId=data['SecretId'],
    SecretKey=data['SecretKey'],
)
client = CosS3Client(config)


def generate_random_string(length):
    """生成指定长度的随机字符串"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))


def upload_image(base64_str):
    # 解码 Base64 图片
    decoded_image = base64.b64decode(base64_str)

    # 生成文件名，格式为: 2022-10-11_20:20-随机字符串.png
    timestamp = time.strftime('%Y-%m-%d_%H:%M')
    random_str = generate_random_string(10)
    filename = f"{timestamp}-{random_str}.png"

    # 在这里构建文件在 COS 中的路径
    cos_path = f'SD_Photo_Output/{filename}'

    # 上传图片
    try:
        response = client.put_object(
            Bucket='mikacat-ai-1302339726',
            Body=decoded_image,
            Key=cos_path,
            EnableMD5=False,
        )
        print('对象存储上传成功.')
    except Exception as e:
        print(f'对象存储上传异常: {e}')

    url = f'https://mikacat-ai-1302339726.cos.ap-shanghai.myqcloud.com/{cos_path}'
    return url


def listUpload_image(base64_list):
    """处理 Base64 编码图片列表的上传"""
    urls = []
    for base64_str in base64_list:
        url = upload_image(base64_str)
        urls.append(url)
    return urls

# 示例调用
# 假设 base64_list 是您的 Base64 图片字符串列表
# urls = listUpload_image(base64_list)
# print(urls)
