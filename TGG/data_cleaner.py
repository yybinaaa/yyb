import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import hashlib


class DataCleaner:
    """
    数据清洗器，集成到推荐系统中
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.clean_file_suffix = "_clean"
        self.backup_suffix = "_backup"
    
    def get_file_hash(self, file_path):
        """计算文件MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_file_clean(self, file_path):
        """检查文件是否已经清洗过"""
        hash_file = file_path.with_suffix('.md5')
        if not hash_file.exists():
            return False
        
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        current_hash = self.get_file_hash(file_path)
        return stored_hash == current_hash
    
    def mark_file_clean(self, file_path):
        """标记文件为已清洗"""
        hash_file = file_path.with_suffix('.md5')
        current_hash = self.get_file_hash(file_path)
        with open(hash_file, 'w') as f:
            f.write(current_hash)
    
    def clean_json_file(self, input_file, output_file, max_line_length=50000):
        """
        清洗JSON文件，去除损坏的行
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            max_line_length: 最大行长度，超过此长度的行被认为是损坏的
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        print(f"开始清洗JSON文件: {input_path}")
        print(f"输出文件: {output_path}")
        
        # 检查是否已经清洗过
        if self.is_file_clean(input_path):
            print(f"文件 {input_path} 已经清洗过，跳过清洗步骤")
            return True
        
        total_lines = 0
        valid_lines = 0
        corrupted_lines = 0
        too_long_lines = 0
        
        # 统计总行数
        print("统计文件行数...")
        with open(input_path, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        
        print(f"总行数: {total_lines}")
        
        # 清洗文件
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(tqdm(infile, total=total_lines, desc="清洗进度")):
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 检查行长度
                if len(line) > max_line_length:
                    too_long_lines += 1
                    print(f"警告: 第{line_num+1}行过长 ({len(line)}字符)，跳过")
                    continue
                
                # 尝试解析JSON
                try:
                    data = json.loads(line)
                    # 验证数据结构
                    if isinstance(data, list) and len(data) > 0:
                        # 检查每个元素是否包含必要的字段
                        valid_data = True
                        for item in data:
                            if not isinstance(item, (list, tuple)) or len(item) < 6:
                                valid_data = False
                                break
                        
                        if valid_data:
                            outfile.write(line + '\n')
                            valid_lines += 1
                        else:
                            corrupted_lines += 1
                            print(f"警告: 第{line_num+1}行数据结构无效，跳过")
                    else:
                        corrupted_lines += 1
                        print(f"警告: 第{line_num+1}行不是有效的列表格式，跳过")
                        
                except json.JSONDecodeError as e:
                    corrupted_lines += 1
                    print(f"警告: 第{line_num+1}行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    corrupted_lines += 1
                    print(f"警告: 第{line_num+1}行未知错误: {e}")
                    continue
        
        print(f"\n清洗完成!")
        print(f"总行数: {total_lines}")
        print(f"有效行数: {valid_lines}")
        print(f"损坏行数: {corrupted_lines}")
        print(f"过长行数: {too_long_lines}")
        print(f"保留率: {valid_lines/total_lines*100:.2f}%")
        
        # 标记文件为已清洗
        self.mark_file_clean(input_path)
        
        return valid_lines > 0
    
    def create_offsets(self, input_file, output_offsets_file):
        """
        为清洗后的文件创建offsets文件
        
        Args:
            input_file: 清洗后的文件路径
            output_offsets_file: 输出offsets文件路径
        """
        print(f"创建offsets文件: {output_offsets_file}")
        
        offsets = []
        current_offset = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="计算offsets"):
                offsets.append(current_offset)
                current_offset += len(line.encode('utf-8')) + 1  # +1 for newline
        
        # 保存offsets
        with open(output_offsets_file, 'wb') as f:
            pickle.dump(offsets, f)
        
        print(f"offsets文件创建完成，共{len(offsets)}个用户")
    
    def backup_original_files(self, files_to_backup):
        """备份原始文件"""
        print("备份原始文件...")
        for file_path in files_to_backup:
            if Path(file_path).exists():
                backup_path = Path(file_path).parent / f"{Path(file_path).stem}{self.backup_suffix}{Path(file_path).suffix}"
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    print(f"已备份: {file_path} -> {backup_path}")
    
    def clean_and_prepare_data(self, force_clean=False):
        """
        清洗并准备数据文件
        
        Args:
            force_clean: 是否强制重新清洗
        """
        print("开始数据清洗和准备...")
        
        # 定义需要清洗的文件
        data_files = [
            "seq.jsonl",
            "predict_seq.jsonl"  # 如果存在的话
        ]
        
        files_to_backup = []
        files_to_replace = []
        
        for data_file in data_files:
            input_path = self.data_dir / data_file
            if not input_path.exists():
                print(f"文件不存在，跳过: {input_path}")
                continue
            
            files_to_backup.append(input_path)
            
            # 生成清洗后的文件路径
            clean_file = input_path.with_name(f"{input_path.stem}{self.clean_file_suffix}{input_path.suffix}")
            clean_offsets_file = clean_file.parent / f"{clean_file.stem}_offsets.pkl"
            
            # 检查是否需要清洗
            if force_clean or not self.is_file_clean(input_path):
                print(f"清洗文件: {input_path}")
                success = self.clean_json_file(input_path, clean_file)
                if success:
                    # 创建offsets文件
                    self.create_offsets(clean_file, clean_offsets_file)
                    files_to_replace.append((input_path, clean_file, clean_offsets_file))
                else:
                    print(f"清洗失败: {input_path}")
            else:
                print(f"文件已清洗，跳过: {input_path}")
        
        # 备份原始文件
        self.backup_original_files(files_to_backup)
        
        # 替换为清洗后的文件
        print("替换为清洗后的文件...")
        for original_file, clean_file, clean_offsets_file in files_to_replace:
            # 替换主文件
            if clean_file.exists():
                import shutil
                shutil.move(clean_file, original_file)
                print(f"已替换: {original_file}")
            
            # 替换offsets文件
            original_offsets_file = original_file.parent / f"{original_file.stem}_offsets.pkl"
            if clean_offsets_file.exists():
                import shutil
                shutil.move(clean_offsets_file, original_offsets_file)
                print(f"已替换: {original_offsets_file}")
        
        print("数据清洗和准备完成!")


def clean_data_integration():
    """
    集成到推荐系统的数据清洗函数
    """
    cleaner = DataCleaner()
    cleaner.clean_and_prepare_data()


if __name__ == "__main__":
    # 测试数据清洗
    cleaner = DataCleaner()
    cleaner.clean_and_prepare_data()
