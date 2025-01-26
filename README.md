# deepseek-r1-on-Inter-NPU
deepseek-ai/DeepSeek-R1-Distill-Qwen的本地部署，NPU终于有用了

## 环境配置

```powershell
pip install openvino-genai==2024.6.0
pip install openvino-dev
pip install --upgrade --upgrade-strategy eager optimum[openvino]
主要就是这几个，其他的看减少了手动pip一下

# 下载DeepSeek-R1-Distill-Qwen-14B到本地并量化
optimum-cli export openvino --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --task text-generation-with-past --weight-format int4 --group-size -1 --ratio 1.0 --sym deepseek-ai\INT4-NPU_compressed_weights 
```

# 运行示例
```python
import openvino_genai as ov_genai
import re

# 强化模板结构（增加格式约束）
def deepseek_think_prompt(question):
    return f"""<|im_start|>system
请严格按以下格式输出：
<thinking>
（在此进行多角度分析，至少包含3个不同科学原理）
</thinking>
<answer>
（用两句话总结，不超过50字）
</answer>

示例：
用户：为什么海水是咸的？
回答：
<thinking>
1. 地球化学循环：陆地岩石中的盐分通过河流冲刷进入海洋...
2. 火山活动：海底火山释放含矿物质的热液...
3. 水分蒸发：海水蒸发只带走纯水，盐分持续积累...</thinking>
<answer>
海水咸味源于陆地盐分经河流输入和海底矿物溶解，经过数十亿年积累，蒸发作用使盐浓度保持稳定。</answer><|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

# 初始化模型（保持你的原始路径）
pipe = ov_genai.LLMPipeline(r"D:\OpenVINO-model\DeepseekR1-14B\deepseek\INT4-NPU_compressed_weights", device="NPU")

# 修正后的生成配置
generation_config = ov_genai.GenerationConfig(
    max_new_tokens=512,    # 适当缩短长度
    temperature=0.6,       # 平衡格式稳定性
    top_p=0.9,
    repetition_penalty=1.1
)

user_question = "为什么后裔射日不射星期日"

# 生成响应
prompt = deepseek_think_prompt(user_question)
output = pipe.generate(prompt, generation_config=generation_config)

# 强化解析逻辑（处理未闭合标签）
def parse_response(text):
    # 清理对话标记
    cleaned = re.sub(r"<\|im_.*?\|>", "", text).strip()
    
    # 提取思考部分（允许不闭合标签）
    think_match = re.search(r"<thinking>(.*?)(?:</thinking>|$)", cleaned, re.DOTALL|re.IGNORECASE)
    think_content = think_match.group(1).strip() if think_match else ""
    
    # 提取答案部分（智能截断）
    answer_match = re.search(r"<answer>(.*?)(?:</answer>|$)", cleaned, re.DOTALL|re.IGNORECASE)
    answer_content = answer_match.group(1).strip() if answer_match else ""
    
    # 降级处理：如果没有找到结构化内容
    if not think_content or not answer_content:
        parts = re.split(r"(答案|结论|综上)", cleaned, maxsplit=1, flags=re.I)
        return {
            "thinking": parts[0].strip(),
            "answer": parts[-1].strip("：: ") if len(parts)>1 else cleaned[-150:]
        }
    
    # 清理嵌套标签
    think_content = re.sub(r"</?think.*?>", "", think_content)
    answer_content = re.sub(r"</?answer.*?>", "", answer_content)
    
    return {
        "thinking": "\n".join([line.strip() for line in think_content.split("\n") if line.strip()]),
        "answer": answer_content.split("\n")[0].strip()
    }

# 解析并打印结果
result = parse_response(output)
print("🔬 科学分析：")
print(result["thinking"])
print("\n💡 简明结论：")
print(result["answer"])
```
