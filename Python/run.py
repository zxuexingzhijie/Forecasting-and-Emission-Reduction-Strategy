#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主运行脚本
用于依次执行数据预处理和可视化分析
"""

import os
import subprocess
import time

def print_section(title):
    """打印带分隔符的标题"""
    line = "=" * 80
    print(f"\n{line}\n{title.center(80)}\n{line}\n")

def run_script(script_name, description):
    """运行Python脚本并显示进度"""
    print_section(description)
    print(f"正在运行 {script_name}...")
    start_time = time.time()
    
    try:
        # 执行脚本
        process = subprocess.Popen(f"python {script_name}", shell=True)
        process.communicate()
        
        # 检查执行结果
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"\n✓ {script_name} 执行成功！耗时 {elapsed_time:.2f} 秒")
            return True
        else:
            print(f"\n✗ {script_name} 执行失败，错误代码: {process.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ 执行 {script_name} 时发生错误: {e}")
        return False

def ensure_directory_exists(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"✓ 已创建目录: {directory_path}")
        except Exception as e:
            print(f"✗ 创建目录失败: {directory_path}, 错误: {e}")
            return False
    return True

def main():
    # 检查脚本文件是否存在
    scripts = [
        ("data_preprocessing.py", "数据预处理"),
        ("data_visualization.py", "数据可视化分析")
    ]
    
    print_section("空气质量数据预处理与分析流程")
    print("此脚本将依次执行数据预处理和可视化分析")
    
    # 确保当前目录是脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"工作目录: {os.getcwd()}")
    
    # 检查输入文件
    input_file = "../附件一.csv"
    if not os.path.exists(input_file):
        print(f"\n✗ 输入文件 {input_file} 不存在！请确保文件位置正确。")
        return
    
    # 确保processed目录存在
    processed_dir = "../processed"
    figures_dir = "../processed/figures"
    
    if not ensure_directory_exists(processed_dir):
        return
    
    if not ensure_directory_exists(figures_dir):
        return
        
    print(f"\n✓ 输出目录已准备: {processed_dir}")
    
    # 依次执行脚本
    failed_scripts = []
    for script, description in scripts:
        if not os.path.exists(script):
            print(f"\n✗ 脚本文件 {script} 不存在！")
            failed_scripts.append(script)
            continue
        
        success = run_script(script, description)
        if not success:
            failed_scripts.append(script)
    
    # 总结运行结果
    print_section("运行结果总结")
    
    if failed_scripts:
        print(f"以下脚本执行失败: {', '.join(failed_scripts)}")
        print("请检查日志以了解详细错误信息。")
    else:
        print("所有脚本执行成功！")
        print("\n输出文件:")
        print(f"  - {processed_dir}/processed_data.csv: 预处理后的数据文件")
        print(f"  - {figures_dir}/: 包含所有生成的可视化图表")

if __name__ == "__main__":
    main() 