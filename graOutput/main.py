import csvfile as csvfile
import requests
import csv
import time
import csvfile

# API密钥，需替换为你自己合法有效的密钥
api_key = "X5E3ISHC41JFT9UJR38JR58SBUDEPZ3DS3"
# 目标以太坊地址，示例中已给出，你可按需替换
address = "0xddbd2b932c763ba5b1b7ae3b362eac3e8d40121a"
# API接口的基础URL
base_url = "https://api.etherscan.io/api"

# 定义CSV文件的表头，根据接口返回的交易数据字段来确定
headers = ["blockNumber", "timeStamp", "hash", "nonce", "blockHash", "transactionIndex", "from", "to", "value",
           "gas", "gasPrice", "isError", "txreceipt_status", "input", "contractAddress", "cumulativeGasUsed",
           "gasUsed", "confirmations", "methodId", "functionName"]
# 打开CSV文件，使用 'w' 模式写入，设置newline=''避免出现多余空行
with open('ethereum_transactions.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()

    # 记录请求次数
    request_count = 0
    while True:
        # 构建请求参数
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": api_key
        }
        try:
            # 发送GET请求
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "1" and data.get("message") == "OK":
                    transactions = data.get("result", [])
                    for transaction in transactions:
                        writer.writerow(transaction)
                else:
                    print(f"第{request_count + 1}次请求，接口返回错误信息: {data.get('message')}")
            else:
                print(f"第{request_count + 1}次请求失败，状态码: {response.status_code}")
        except requests.RequestException as e:
            print(f"第{request_count + 1}次请求出现异常: {str(e)}")

        request_count += 1
        # 控制请求频率，这里简单假设每秒请求不超过5次（具体按接口要求来），每次请求间隔0.2秒
        time.sleep(0.2)