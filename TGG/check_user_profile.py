import json

def check_user_profile():
    found = False
    count = 0
    
    with open('./data/seq.jsonl', 'r') as f:
        for line in f:
            if count < 100:  # 只检查前100行
                try:
                    data = json.loads(line)
                    if data[2] is not None:  # 检查用户特征字段
                        print(f'第{count+1}行找到用户画像数据: {data[2]}')
                        found = True
                        break
                    count += 1
                except:
                    continue
            else:
                break
    
    print(f'\n前100行中是否找到用户画像: {found}')
    
    if not found:
        print("所有行的用户特征字段都是 None")
    
    return found

if __name__ == "__main__":
    check_user_profile()
