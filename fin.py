import argparse
import asyncio
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import aiohttp
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms, models
from cnocr import CnOcr
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class GenderClassifier(nn.Module):
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


class HiveExecutor:
    def exeSql(self, sql):
        logger.info(f"Executing Hive SQL: {sql}")
        return (["user1@example.com", "user2@example.com", "user@example", "12312"],
                ["1.jpg","2.jpg","3.jpg","4.jpg"])

    def execute_insert(self, sql):
        pass


def getHiveExecutor():
    return HiveExecutor()


def _bytes_to_cv2(image_data):
    try:
        img = PILImage.open(BytesIO(image_data)).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting bytes to CV2 image: {e}")
        return None


def _generate_rotations(image):
    rotations = [image]
    for rotate_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        rotations.append(cv2.rotate(image, rotate_code))
    return rotations


def _gender_from_text(text):
    if not text:
        return None

    male_words = ['男', 'male', 'Male', 'MALE', 'man', 'Man']
    female_words = ['女', 'female', 'Female', 'FEMALE', 'woman', 'Woman']

    has_male = any(word in text for word in male_words)
    has_female = any(word in text for word in female_words)

    if has_male and not has_female:
        return '男'
    elif has_female and not has_male:
        return '女'
    elif has_male and has_female:
        male_count = sum(text.count(word) for word in male_words)
        female_count = sum(text.count(word) for word in female_words)
        return '男' if male_count > female_count else '女'
    return None


class SimpleGenderDetector:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._init_models()
        logger.info("模型加载完成")

    def _init_models(self):
        try:
            self.device = torch.device('cpu')
            self.model = GenderClassifier(use_lite=True)
            model_path = 'face_gender_model.pth'

            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device).eval()
                logger.info("性别分类器加载成功")
            else:
                raise FileNotFoundError(f"模型文件未找到: {model_path}")

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            logger.error(f"性别分类器加载失败: {e}")
            self.model = None

        try:
            self.ocr = CnOcr(
                rec_model_name='ch_PP-OCRv5_server',
                det_model_name='ch_PP-OCRv5_det_server',
                rec_model_backend='onnx',
                det_model_backend='onnx',
                font_path='None',
                context='cpu'
            )
        except Exception as e:
            logger.error(f"OCR加载失败: {e}")
            self.ocr = None

    async def process_user(self, email, urls, session, semaphore):

        original_urls_str = ','.join(urls[:12])
        if not urls:
            return {'email': email, 'gender': '未知', 'method': -1, 'original_urls': original_urls_str}

        logger.info(f"处理用户: {email}, 图片数: {len(urls)}")

        model_results = []   # (gender, confidence)

        for url in urls[:12]:
            result = await self._process_single_image(url, session, semaphore)
            if not result:
                continue

            gender, confidence, method = result

            if method == 0:
                return {'email': email, 'gender': gender, 'method': 0, 'original_urls': original_urls_str}

            if method == 1:
                model_results.append((gender, confidence))

        if model_results:
            best_gender, best_conf = max(model_results, key=lambda x: x[1])
            return {
                'email': email,
                'gender': best_gender,
                'method': 1,
                'original_urls': original_urls_str
            }

        return {'email': email, 'gender': '未知', 'method': -1, 'original_urls': original_urls_str}

    async def _process_single_image(self, url, session, semaphore):
        try:
            image = await self._download_image(session, url, semaphore)
            if image is None:
                return None

            for rot_img in _generate_rotations(image):

                ocr_text = await self._extract_text(rot_img)
                if ocr_gender := _gender_from_text(ocr_text):
                    return ocr_gender, 1.0, 0

                gender, conf = await self._classify_face(rot_img)
                if gender:
                    return gender, conf, 1

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            pass

        return None

    async def _download_image(self, session, url, semaphore):
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content = await response.read()
                        return await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, _bytes_to_cv2, content
                        )
            except Exception as e:
                logger.error(f"Error downloading image: {e}")
                return None

    async def _extract_text(self, image):
        if not self.ocr or image is None:
            return ""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._extract_text_sync, image
            )
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    def _extract_text_sync(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb_image.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                rgb_image = cv2.resize(rgb_image, new_size)

            results = self.ocr.ocr(rgb_image)
            return ' '.join(r['text'] for r in results)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    async def _classify_face(self, image):
        if not self.model or image is None:
            return None, 0
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._classify_face_sync, image
            )
        except Exception as e:
            logger.error(f"Error classifying face: {e}")
            return None, 0

    def _classify_face_sync(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)

            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = probabilities.max(1)
                confidence = confidence.item()

            gender = '男' if predicted.item() == 0 else '女'
            return gender, confidence

        except Exception as e:
            logger.error(f"Error classifying face: {e}")
            return None, 0


async def process_all_users(detector, df, concurrency=20):
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            detector.process_user(
                str(row.iloc[0]),
                [url.strip() for url in str(row.iloc[1]).split(',') if url.strip()][:12],
                session,
                semaphore
            )
            for _, row in df.iterrows()
        ]

        batch_size = 25
        for i in range(0, len(tasks), batch_size):
            batch_results = await asyncio.gather(*tasks[i:i + batch_size], return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)

            await asyncio.sleep(0.1)

    return results


def main():
    parser = argparse.ArgumentParser(description='性别识别')
    parser.add_argument('--concurrency', default=20, type=int, help='并发数')
    parser.add_argument('--partition_date', default='2025-11-28', help='分区日期')
    args = parser.parse_args()

    try:
        hive_executor = getHiveExecutor()
        passport_list, document_list = hive_executor.exeSql(
            f"SELECT passport, document FROM your_input_table WHERE dt='{args.partition_date}'"
        )
        df = pd.DataFrame({'passport': passport_list, 'document': document_list})
        logger.info(f"从Hive表读取 {len(df)} 行数据")

    except Exception as e:
        logger.error(f"从Hive读取数据失败: {e}")
        return

    detector = SimpleGenderDetector()
    start_time = time.time()

    try:
        results = asyncio.run(process_all_users(detector, df, args.concurrency))
        df_results = pd.DataFrame(results)
        print("所有用户识别结果：")
        print(df_results[['email', 'gender', 'method']])
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return

    if results:
        values = [
            f"('{r['email'].replace(chr(39), chr(39) + chr(39))}', '{r['gender']}', {r['method']}, '{r['original_urls'].replace(chr(39), chr(39) + chr(39))}')"
            for r in results
        ]

        insert_sql = f"INSERT OVERWRITE TABLE your_output_table PARTITION(dt='{args.partition_date}') (passport, gender, method, original_urls) VALUES {', '.join(values)}"
        hive_executor.execute_insert(insert_sql)
        logger.info(f"成功写入 {len(results)} 条记录")

    processing_time = time.time() - start_time
    gender_stats = Counter(r['gender'] for r in results)
    method_stats = Counter(r['method'] for r in results)

    logger.info(f"处理完成!")
    logger.info(f"总用户数: {len(results)}")
    logger.info(f"总耗时: {processing_time:.2f}秒")
    logger.info(f"平均时间: {processing_time / len(results):.2f}秒/用户")
    logger.info(f"性别分布: {dict(gender_stats)}")
    logger.info(f"识别方式分布: {dict(method_stats)}")


if __name__ == '__main__':
    main()
