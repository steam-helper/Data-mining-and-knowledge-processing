import json
import streamlit as st

def load_data(filepath):
    """加载数据，并确保文件路径和格式正确"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.write(f"成功加载 {len(data)} 篇文章 from {filepath}")
        return data
    except FileNotFoundError:
        st.error(f"数据文件未找到: {filepath}")
        return []
    except json.JSONDecodeError:
        st.error(f"解析 JSON 文件时出错: {filepath}")
        return []
    except Exception as e:
        st.error(f"加载数据时发生错误: {e}")
        return []
