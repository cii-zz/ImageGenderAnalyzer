性别识别系统使用说明

功能简介

通过OCR文字识别和人脸分析，自动从图片中识别性别信息。

快速开始

1. 准备输入文件

创建 input.xlsx 文件，包含两列：
• 第一列：用户邮箱

• 第二列：图片URL（多个用逗号分隔）

示例：
邮箱 图片URL

user1@example.com http://example.com/photo1.jpg,http://example.com/photo2.jpg

2. 运行程序

python main.py


3. 查看结果

结果保存到 output.xlsx，包含：
• 邮箱

• 识别结果（男/女/未知）

• 原始图片URL

常用参数

# 指定输入输出文件
python main.py --input_excel data.xlsx --output_excel result.xlsx

# 使用GPU加速
python main.py --gpu 0

# 只处理前100条数据
python main.py --limit 100


处理流程

1. 读取Excel中的邮箱和图片URL
2. 下载图片（每用户最多3张）
3. 优先用OCR识别图片中的"男"/"女"文字
4. 如果OCR失败，检测图片中的人脸并分析性别
5. 综合多张图片结果得出最终性别
6. 生成结果Excel文件

