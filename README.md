ImageGenderAnalyzer 性别识别系统

📋 处理流程

1. 读取输入数据 - 从Excel文件读取邮箱和图片URL
2. 下载用户图片 - 异步下载每个用户的图片（最多3张）
3. 文字识别优先 - 使用OCR识别图片中的"男"/"女"文字
4. 人脸检测备用 - 如果文字识别失败，检测图片中的人脸
5. 性别分类 - 对检测到的人脸进行性别判断
6. 结果决策 - 综合多张图片的结果得出最终性别
7. 输出结果 - 生成包含识别结果的Excel文件

🤖 使用的模型

1. 文字识别 (OCR)

• 模型: CnOcr

• 用途: 识别图片中的中文文字，特别是"男"、"女"等性别关键词

• 选择原因:

• 专门针对中文优化，识别准确率高

• 轻量级，推理速度快

• 无需额外配置，开箱即用

2. 人脸检测

• 模型: InsightFace (buffalo_l) (可考虑RetinaFace)

• 用途: 检测图片中的人脸区域

• 选择原因:

• 业界领先的人脸检测精度

• 支持多人脸检测

• 提供人脸关键点和质量评分

3. 性别分类

• 模型: rizvandwiki/gender-classification (HuggingFace)

• 用途: 对检测到的人脸进行性别分类

• 选择原因:

• 专门为性别分类任务训练

• 基于Transformer架构，准确率高

• 支持CPU/GPU推理，部署灵活

🚀 使用方法

基本使用

python main.py


指定输入输出文件

python main.py --input_excel data.xlsx --output_excel result.xlsx


使用GPU加速

python main.py --gpu 0


处理部分数据

# 从第10行开始，处理50条数据
python main.py --start 10 --limit 50


控制并发数

# 网络不好时减少并发数
python main.py --concurrency 5


⚙️ 参数说明

参数 说明 默认值

--input_excel 输入Excel文件路径 input.xlsx

--output_excel 输出Excel文件路径 output.xlsx

--gpu GPU设备ID（-1=CPU） -1

--start 起始处理行号 0

--limit 处理数据条数（0=全部） 0

--concurrency 并发下载图片数 15

📄 输入文件格式

Excel文件需要包含至少两列：
• 第一列: 用户邮箱

• 第二列: 图片URL（多个URL用逗号分隔）

示例：

邮箱 图片URL

user1@example.com http://example.com/photo1.jpg,http://example.com/photo2.jpg

user2@example.com http://example.com/image.jpg

📊 输出结果

生成的Excel文件包含：
• email: 用户邮箱

• gender: 识别结果（男/女/未知）

• original_urls: 原始图片URL

运行结束后控制台会显示处理统计信息。