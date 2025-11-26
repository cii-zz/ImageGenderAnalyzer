import argparse
import os
import cv2
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image as PILImage
from pillow_heif import register_heif_opener
import pandas as pd
from collections import Counter
import numpy as np
import logging
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from concurrent.futures import ThreadPoolExecutor
import time

register_heif_opener()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleGenderDetector:
    def __init__(self, args):
        self.args = args
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._init_models()
        logger.info("模型加载完成")

    def _init_models(self):
        try:
            from insightface.app import FaceAnalysis
            self.face_detector = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_detector.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.6)
        except Exception as e:
            logger.error(f"人脸检测器加载失败: {e}")
            self.face_detector = None

        try:
            self.device = torch.device('cuda' if self.args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
            model_name = "rizvandwiki/gender-classification"
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"性别分类器加载失败: {e}")
            self.model = None

        try:
            from cnocr import CnOcr
            self.ocr = CnOcr(
                rec_model_name='doc-densenet_lite_136-gru',  # 识别模型
                det_model_name='ch_PP-OCRv5_det',  # 检测模型
                rec_model_backend='onnx',  # 识别模型使用ONNX后端
                det_model_backend='onnx',  # 检测模型使用ONNX后端
                context='gpu' if self.args.gpu >= 0 and torch.cuda.is_available() else 'cpu'  # 根据参数自动选择
            )
        except Exception as e:
            logger.warning(f"OCR加载失败: {e}")
            self.ocr = None

    async def process_user(self, email, urls, session, semaphore):
        if not urls:
            return {'email': email, 'gender': '未知', 'original_urls': ''}

        logger.info(f"处理用户: {email}, 图片数: {len(urls)}")

        urls = urls[:3]
        gender_results = []

        for url in urls:
            try:
                image = await self._download_image(session, url, semaphore)
                if image is None:
                    continue

                # OCR文本识别
                ocr_text = await self._extract_text(image)
                ocr_gender = self._gender_from_text(ocr_text)

                if ocr_gender is not None:
                    gender_results.append(ocr_gender)
                    logger.info(f"OCR检测到性别: {ocr_gender}")
                    continue

                # 人脸检测和性别识别
                faces = await self._detect_faces(image)
                if faces:
                    face_gender = await self._classify_faces(faces)
                    if face_gender is not None:
                        gender_results.append(face_gender)
                        logger.info(f"人脸检测到性别: {face_gender}")

            except Exception as e:
                logger.debug(f"处理图片失败: {e}")
                continue

        final_gender = self._decide_gender(gender_results)
        logger.info(f"用户 {email} 最终结果: {final_gender}")
        return {'email': email, 'gender': final_gender, 'original_urls': ','.join(urls)}

    async def _download_image(self, session, url, semaphore):
        async with semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.read()
                        return await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, self._bytes_to_cv2, content
                        )
            except Exception as e:
                logger.debug(f"下载失败: {e}")
                return None

    def _bytes_to_cv2(self, image_data):
        try:
            img = PILImage.open(BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except:
            return None

    async def _extract_text(self, image):
        if self.ocr is None or image is None:
            return ""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._extract_text_sync, image
            )
        except:
            return ""

    def _extract_text_sync(self, image):
        try:
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # 图像尺寸调整
            h, w = rgb_image.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                rgb_image = cv2.resize(rgb_image, new_size)

            # OCR识别
            results = self.ocr.ocr(rgb_image)
            text = ' '.join([r['text'] for r in results])
            return text if len(text) > 0 else ""
        except:
            return ""

    def _gender_from_text(self, text):
        if not text:
            return None

        # 扩展性别关键词，包括中英文
        male_words = ['男', 'male', 'Male', 'MALE', '先生', '男士', '男生', '男人', 'man', 'Man']
        female_words = ['女', 'female', 'Female', 'FEMALE', '女士', '女生', '女人', '小姐', 'woman', 'Woman']

        has_male = any(word in text for word in male_words)
        has_female = any(word in text for word in female_words)

        if has_male and not has_female:
            return '男'
        elif has_female and not has_male:
            return '女'
        elif has_male and has_female:
            return self._resolve_gender_conflict(text, male_words, female_words)
        else:
            return None

    def _resolve_gender_conflict(self, text, male_words, female_words):
        male_count = sum(1 for word in male_words if word in text)
        female_count = sum(1 for word in female_words if word in text)

        if male_count > female_count:
            return '男'
        elif female_count > male_count:
            return '女'
        else:
            male_first_pos = len(text)
            female_first_pos = len(text)

            for word in male_words:
                if word in text:
                    pos = text.find(word)
                    if pos < male_first_pos:
                        male_first_pos = pos

            for word in female_words:
                if word in text:
                    pos = text.find(word)
                    if pos < female_first_pos:
                        female_first_pos = pos

            if male_first_pos < female_first_pos:
                return '男'
            elif female_first_pos < male_first_pos:
                return '女'
            else:
                return None

    async def _detect_faces(self, image):
        if self.face_detector is None or image is None:
            return []
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._detect_faces_sync, image
            )
        except:
            return []

    def _detect_faces_sync(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_detector.get(rgb_image)

            cropped_faces = []
            for face in faces:
                if face.det_score >= 0.6:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    expand = int((x2 - x1) * 0.1)
                    x1, y1 = max(0, x1 - expand), max(0, y1 - expand)
                    x2, y2 = min(image.shape[1], x2 + expand), min(image.shape[0], y2 + expand)

                    face_img = image[y1:y2, x1:x2]
                    if face_img.size > 0:
                        cropped_faces.append(face_img)

            return cropped_faces[:3]
        except:
            return []

    async def _classify_faces(self, faces):
        if self.model is None or not faces:
            return None
        try:
            genders = []
            for face in faces:
                gender = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, self._classify_face_sync, face
                )
                if gender is not None:
                    genders.append(gender)

            if genders:
                gender_counter = Counter(genders)
                return gender_counter.most_common(1)[0][0]
            return None
        except:
            return None

    def _classify_face_sync(self, face):
        try:
            if len(face.shape) == 3:
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            else:
                rgb_face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

            # 预处理
            resized = cv2.resize(rgb_face, (224, 224))
            pil_image = PILImage.fromarray(resized)

            # 模型预测
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                _, pred = torch.max(probs, 1)

            # 返回预测结果
            return '女' if pred.item() % 2 == 0 else '男'

        except Exception as e:
            logger.debug(f"性别分类失败: {e}")
            return None

    def _decide_gender(self, gender_results):
        if not gender_results:
            return "未知"

        gender_counter = Counter(gender_results)
        most_common = gender_counter.most_common(1)

        if most_common:
            return most_common[0][0]
        return "未知"


async def process_all_users(detector, df, concurrency=20):
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=20)
    results = []

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _, row in df.iterrows():
            email = str(row.iloc[0])
            urls_str = str(row.iloc[1]) if len(row) > 1 else ""
            urls = [url.strip() for url in urls_str.split(',') if url.strip()]
            task = detector.process_user(email, urls, session, semaphore)
            tasks.append(task)

        batch_size = 25
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                else:
                    logger.error(f"处理失败: {result}")

            logger.info(f"进度: {min(i + batch_size, len(tasks))}/{len(tasks)}")
            await asyncio.sleep(0.1)

    return results


def main():
    parser = argparse.ArgumentParser(description='性别识别')
    parser.add_argument('--input_excel', default='input.xlsx', help='输入Excel文件')
    parser.add_argument('--output_excel', default='output.xlsx', help='输出Excel文件')
    parser.add_argument('--gpu', default=-1, type=int, help='GPU ID')
    parser.add_argument('--start', default=0, type=int, help='起始位置')
    parser.add_argument('--limit', default=0, type=int, help='处理条数')
    parser.add_argument('--concurrency', default=20, type=int, help='并发数')

    args = parser.parse_args()

    if not os.path.exists(args.input_excel):
        logger.error(f"输入文件不存在: {args.input_excel}")
        return

    try:
        df = pd.read_excel(args.input_excel)
        logger.info(f"读取 {len(df)} 行数据")
    except Exception as e:
        logger.error(f"读取Excel失败: {e}")
        return

    if df.shape[1] < 2:
        logger.error("Excel需要至少两列")
        return

    start = max(0, args.start)
    end = start + args.limit if args.limit > 0 else len(df)
    df_processed = df.iloc[start:end].copy()

    logger.info(f"处理 {len(df_processed)} 行数据")

    detector = SimpleGenderDetector(args)
    start_time = time.time()

    try:
        results = asyncio.run(process_all_users(detector, df_processed, args.concurrency))
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return

    result_df = pd.DataFrame(results)
    result_df.to_excel(args.output_excel, index=False)

    processing_time = time.time() - start_time
    gender_stats = Counter(result_df['gender'])

    print(f"\n处理完成!")
    print(f"总用户数: {len(result_df)}")
    print(f"总耗时: {processing_time:.2f}秒")
    print(f"平均时间: {processing_time/len(result_df):.2f}秒/用户")
    print(f"性别分布: {dict(gender_stats)}")
    print(f"输出文件: {args.output_excel}")


if __name__ == '__main__':
    main()