# deepseek-r1-on-Inter-NPU
deepseek-ai/DeepSeek-R1-Distill-Qwen的本地部署，NPU终于有用了

# 搭建openvino部署环境
pip install openvino-genai==2024.6.0
pip install openvino-dev
pip install --upgrade --upgrade-strategy eager optimum[openvino]
主要就是这几个，其他的看减少了手动pip一下

# 下载DeepSeek-R1-Distill-Qwen-14B到本地并量化
optimum-cli export openvino --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --task text-generation-with-past --weight-format int4 --group-size -1 --ratio 1.0 --sym deepseek-ai\INT4-NPU_compressed_weights 

# 运行mian.py文件，简单问答

