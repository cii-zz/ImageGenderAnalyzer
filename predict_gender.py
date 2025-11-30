"""
人脸性别识别推理脚本 - 精简版
"""

import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenderClassifier(nn.Module):
    """基于ResNet的性别分类器"""
    def __init__(self, use_lite=True):
        super(GenderClassifier, self).__init__()
        if use_lite:
            self.resnet = models.resnet18(weights=None)
            num_features = 512
        else:
            self.resnet = models.resnet50(weights=None)
            num_features = 2048

        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.resnet(x)

def load_model(model_path='face_gender_model.pth'):
    """加载训练好的模型"""
    model = GenderClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型加载成功!")
    return model

def get_inference_transform():
    """获取推理时的图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, transform):
    """预测单张图片的性别"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        gender = 'male' if predicted.item() == 0 else 'female'
        confidence = confidence.item() * 100
        return gender, confidence
    except Exception as e:
        print(f"预测失败 {image_path}: {e}")
        return None, 0.0

def interactive_viewer_mode():
    """交互式查看器模式 - 带弹窗显示"""
    print("加载模型...")
    model = load_model('face_gender_model.pth')
    transform = get_inference_transform()

    folder_path = input("请输入要预测的文件夹路径: ").strip()
    if not os.path.exists(folder_path):
        print("文件夹不存在!")
        return

    # 获取所有图片文件并随机排序
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("文件夹中没有找到图片!")
        return

    # 随机打乱图片顺序
    random.shuffle(image_files)
    print(f"\n找到 {len(image_files)} 张图片，已随机排序")

    current_idx = 0
    window_name = "人脸性别识别系统"

    # 创建窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 800)

    # 尝试加载中文字体
    try:
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 30)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    while True:
        # 检查窗口是否被关闭
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("窗口被关闭，退出程序")
            break

        img_path = image_files[current_idx]

        # 预测性别
        gender, confidence = predict_single_image(model, img_path, transform)
        if not gender:
            gender = "未知"
            confidence = 0

        # 读取并处理图片
        try:
            # 使用PIL读取图片
            pil_img = Image.open(img_path).convert('RGB')
            img_width, img_height = pil_img.size

            # 计算缩放比例
            max_width, max_height = 800, 500
            scale = min(max_width/img_width, max_height/img_height, 1.0)
            new_width, new_height = int(img_width*scale), int(img_height*scale)

            # 缩放图片
            if scale != 1.0:
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建画布（白色背景）
            canvas_width = max(new_width, 800)
            canvas_height = new_height + 250  # 为文字留出更多空间
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

            # 将图片放在画布中央
            x_offset = (canvas_width - new_width) // 2
            canvas.paste(pil_img, (x_offset, 20))

            # 创建绘图对象
            draw = ImageDraw.Draw(canvas)

            # 绘制文字信息
            text_y = new_height + 40

            # 文件名
            file_name = os.path.basename(img_path)
            draw.text((20, text_y), f"图片: {file_name}", fill='black', font=font)
            text_y += 40

            # 性别结果
            gender_text = "男性 (Male)" if gender == 'male' else "女性 (Female)"
            gender_color = 'blue' if gender == 'male' else 'purple'
            draw.text((20, text_y), f"性别: {gender_text}", fill=gender_color, font=font)
            text_y += 40

            # 置信度
            confidence_color = 'green' if confidence > 80 else 'orange' if confidence > 60 else 'red'
            draw.text((20, text_y), f"置信度: {confidence:.1f}%", fill=confidence_color, font=font)
            text_y += 40

            # 进度信息
            progress_text = f"进度: {current_idx+1}/{len(image_files)}"
            draw.text((20, text_y), progress_text, fill='gray', font=font)
            text_y += 40

            # 操作提示 - 集成到窗口中
            hint_text = "操作: N/→下一张  P/←上一张  R重新预测  Q/ESC退出"
            text_width = draw.textlength(hint_text, font=font)
            hint_x = (canvas_width - text_width) // 2
            draw.text((hint_x, canvas_height-40), hint_text, fill='darkblue', font=font)

            # 转换为OpenCV格式
            display_img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"显示图片失败: {e}")
            # 创建错误图像
            display_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(display_img, f"加载失败: {e}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图片
        cv2.imshow(window_name, display_img)

        # 等待按键
        key = cv2.waitKey(0) & 0xFF

        # 处理按键
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            print("退出查看器")
            break
        elif key == ord('n') or key == ord('N') or key == 83:  # N or 右箭头
            current_idx = (current_idx + 1) % len(image_files)
        elif key == ord('p') or key == ord('P') or key == 81:  # P or 左箭头
            current_idx = (current_idx - 1) % len(image_files)
        elif key == ord('r') or key == ord('R'):  # 重新预测
            print("重新预测当前图片...")
            # 这里可以添加重新预测的逻辑

    # 关闭窗口
    cv2.destroyAllWindows()

def main():
    print("="*50)
    print("人脸性别识别系统")
    print("="*50)
    interactive_viewer_mode()

if __name__ == '__main__':
    main()