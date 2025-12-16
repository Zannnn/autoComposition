import json
import sys

def analyze_json_structure(data, indent=0, parent_key=''):
    """
    递归分析JSON数据结构
    :param data: 要分析的JSON数据
    :param indent: 缩进量，用于格式化输出
    :param parent_key: 父级键名，用于构建完整键路径
    """
    if isinstance(data, dict):
        # 如果是字典类型，遍历所有键值对
        for key in data:
            current_key = f"{parent_key}.{key}" if parent_key else key
            print("  " * indent + f"- 键: {key}")
            
            # 检查值的类型
            value_type = type(data[key]).__name__
            print("  " * indent + f"  类型: {value_type}")
            
            # 如果值是字典或列表，递归分析
            if isinstance(data[key], (dict, list)):
                print("  " * indent + "  包含:")
                analyze_json_structure(data[key], indent + 1, current_key)
            print()  # 增加空行分隔不同的键
    elif isinstance(data, list):
        # 如果是列表类型，分析第一个元素的结构（假设列表元素结构一致）
        if len(data) > 0:
            print("  " * indent + "- 列表元素结构:")
            analyze_json_structure(data[0], indent + 1, parent_key)
        else:
            print("  " * indent + "- 空列表")
    else:
        # 基本数据类型，无需进一步分析
        pass

def main():
    if len(sys.argv) != 2:
        print("使用方法: python json_analyzer.py <json文件路径>")
        print("示例: python json_analyzer.py data.json")
        return
    
    file_path = sys.argv[1]
    
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                json_data = json.load(file)
                print(f"成功读取JSON文件: {file_path}")
                print("JSON结构分析如下:\n")
                analyze_json_structure(json_data)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print("请检查文件是否为有效的JSON格式")
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
