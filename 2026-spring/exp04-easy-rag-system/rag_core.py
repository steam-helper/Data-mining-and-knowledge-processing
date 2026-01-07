import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY

def generate_answer(query, context_docs, gen_model, tokenizer):
    """根据查询和上下文生成答案"""
    if not context_docs:
        return "未找到相关文档来回答您的问题。"
    if not gen_model or not tokenizer:
        st.error("生成模型或分词器不可用。")
        return "错误: 生成组件未加载。"

    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs])  # 合并检索到的文档

    prompt = f"""基于以下上下文文档，仅回答用户的问题。
如果在上下文中没有找到答案，请明确说明，不要编造信息。

上下文文档:
{context}

用户问题: {query}

答案:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"生成答案时发生错误: {e}")
        return "抱歉，我在生成答案时遇到错误。"
