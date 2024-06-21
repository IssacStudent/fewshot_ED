# 指定Markdown文件的路径
file_path = 'a.md'

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 移除空行（空行可能包含空格、制表符和换行符）
non_empty_lines = [line for line in lines if line.strip() != '']

# 将处理后的内容写回文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(non_empty_lines)
