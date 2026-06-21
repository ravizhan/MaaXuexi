import json
import os
import re
import time
import traceback
from base64 import b64encode
from io import BytesIO
from queue import SimpleQueue
from random import randint

import numpy as np
import plyer
from PIL import Image, ImageDraw
from PIL import ImageFilter
from httpx import Client, Timeout
from maa.controller import AdbController
from maa.custom_recognition import CustomRecognition
from maa.define import TaskDetail
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit


class AIResolver:
    def __init__(
            self,
            api_key,
            model,
    ):
        self.session = Client(timeout=Timeout(120.0))
        self.session.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = model

    @staticmethod
    def image_encode(img: np.ndarray) -> str:
        buffered = BytesIO()
        im = Image.fromarray(img)
        im = im.filter(ImageFilter.SHARPEN)
        im.save(buffered, format="JPEG")
        encoded_image = b64encode(buffered.getvalue()).decode()
        return encoded_image

    @staticmethod
    def image_combine(imgs: list[np.ndarray], pre_scale: float = 1.0) -> np.ndarray:
        if pre_scale != 1.0:
            imgs = [np.array(Image.fromarray(img).resize(
                (round(img.shape[1] * pre_scale), round(img.shape[0] * pre_scale)),
                Image.Resampling.LANCZOS
            )) for img in imgs]
        max_h = max(img.shape[0] for img in imgs)
        total_w = sum(img.shape[1] for img in imgs)
        new_img = np.zeros((max_h, total_w, 3), dtype=np.uint8)
        x_offset = 0
        for img in imgs:
            new_img[:img.shape[0], x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1]
        return new_img

    def resolve_choice(self, imgs: list[np.ndarray]) -> list[str] | None:
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "能力与角色:你是一位答题助手。\n背景信息:你会得到一张左边为选择题右边为答案的图片\n指令:你需要仔细阅读图片中的两部分内容，其中答案为红字部分，回答包含答案的选项\n输出风格:你无需给出推理过程以及任何解释。你只需要回答正确选项对应的字母，不得回答任何多余的文字，不得添加任何的标点符号。\n输出范围:我希望你仅仅回答 ABCDE 中的一个或多个字母。"
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpg;base64," + self.image_encode(self.image_combine(imgs, 0.5))
                        },
                    }]
                }
            ],
            "temperature": 0.2
        }
        response = self.session.post(self.url, json=data)
        if DEBUG_MODE:
            print(f"[AI] 选择题 请求状态={response.status_code}")
        try:
            if response.status_code == 200:
                result = response.json()
                raw = result["choices"][0]["message"]["content"]
                if DEBUG_MODE:
                    print(f"[AI] 选择题 原始返回=\"{raw}\"")
                answer = list(raw)
                for i in answer.copy():
                    if i not in ['A', 'B', 'C', 'D', 'E', 'F']:
                        answer.remove(i)
                if len(answer) == 0:
                    raise ValueError("Invalid answer")
                if DEBUG_MODE:
                    print(f"[AI] 选择题 解析结果={answer}")
            else:
                if DEBUG_MODE:
                    print(f"[AI] 选择题 请求失败: {response.text}")
                answer = None
        except Exception as e:
            if DEBUG_MODE:
                print(f"[AI] 选择题 异常: {e}, body={response.text}")
            answer = None
        return answer

    def resolve_blank(self, imgs: list[np.ndarray], answer: bool, blank_num: int) -> str | None:
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "能力与角色:你是一位答题助手\n背景信息:你会得到一张左边为填空题右边为答案的图片\n指令:你需要仔细阅读图片中的两部分内容，其中答案为红字部分，回答空缺处应当填写的内容\n输出风格:你无需给出推理过程，也无需给出任何解释。你只需要回答空缺处应当填的内容，填充字数应当与空缺数量相同"
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpg;base64," + self.image_encode(self.image_combine(imgs, 0.5))
                        },
                    }]
                }
            ],
            "temperature": 0.2
        }
        if not answer:
            data["messages"][0]["content"] = f"能力与角色:你是一位答题助手\n背景信息:你会得到一张包含填空题的图片\n指令:你需要阅读该图片中的问题，认真理解题目和前后文，其中答案为{blank_num}个字符，思考后作出回答，确保填入答案后的全文逻辑正确，语义正确\n输出风格:你无需给出推理过程，也无需给出任何解释。你只需要回答空缺处应当填的内容，填充字数应当为{blank_num}"
            data["model"] = self.model
        response = self.session.post(self.url, json=data)
        if DEBUG_MODE:
            print(f"[AI] 填空题 请求状态={response.status_code}")
        try:
            if response.status_code == 200:
                result = response.json()
                raw = result["choices"][0]["message"]["content"]
                if DEBUG_MODE:
                    print(f"[AI] 填空题 原始返回=\"{raw}\"")
                result = raw
            else:
                if DEBUG_MODE:
                    print(f"[AI] 填空题 请求失败: {response.text}")
                result = None
        except Exception as e:
            if DEBUG_MODE:
                print(f"[AI] 填空题 异常: {e}, body={response.text}")
            result = None
        return result

    def resolve_click_blank(self, imgs: list[np.ndarray]) -> str | None:
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "能力与角色:你是一位答题助手\n背景信息:你会得到一张左边为点选填空题右边为答案的图片\n指令:你需要仔细阅读图片中的两部分内容，其中答案为红字部分，应该的选择顺序，按照点选顺序点选的文字，使用用英文逗号分隔\n输出风格:你无需给出推理过程，也无需给出任何解释。你只需要按顺序回答点选顺序，用英文逗号分隔"
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpg;base64," + self.image_encode(self.image_combine(imgs, 0.5))
                        },
                    }]
                }
            ],
            "temperature": 0.2
        }
        response = self.session.post(self.url, json=data)
        if DEBUG_MODE:
            print(f"[AI] 点选填空题 请求状态={response.status_code}")
        try:
            if response.status_code == 200:
                result = response.json()
                raw = result["choices"][0]["message"]["content"]
                if DEBUG_MODE:
                    print(f"[AI] 点选填空题 原始返回=\"{raw}\"")
                result = raw
            else:
                if DEBUG_MODE:
                    print(f"[AI] 点选填空题 请求失败: {response.text}")
                result = None
        except Exception as e:
            if DEBUG_MODE:
                print(f"[AI] 点选填空题 异常: {e}, body={response.text}")
            result = None
        return result


resource = Resource()
resource.set_cpu()
resource.post_bundle("./resource").wait()

DEBUG_MODE = False


def _dilate(mask: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """形态学膨胀：用 kh×kw 矩形核连接相邻像素"""
    ph, pw = kh // 2, kw // 2
    padded = np.pad(mask, ((ph, ph), (pw, pw)), mode='constant', constant_values=False)
    h, w = mask.shape
    result = np.zeros_like(mask)
    for di in range(kh):
        for dj in range(kw):
            result |= padded[di:di + h, dj:dj + w]
    return result


class RedTextOCR(CustomRecognition):
    """提示红字识别：红色度过滤 → 形态学膨胀分块 → 逐块OCR → 出血线合并"""

    ROI_X, ROI_Y, ROI_W, ROI_H = 0, 483, 720, 517
    LEFT_BLEED, RIGHT_BLEED, BLEED_TH = 37, 682, 5

    def analyze(self, context, argv: CustomRecognition.AnalyzeArg):
        img = argv.image
        rx, ry, rw, rh = self.ROI_X, self.ROI_Y, self.ROI_W, self.ROI_H
        hint_img = img[ry:ry + rh, rx:rx + rw]

        # 1. 红色度过滤：R - max(B, G) > 50，非红色像素置白
        redness = hint_img[:, :, 2].astype(np.int16) - np.maximum(
            hint_img[:, :, 0], hint_img[:, :, 1]).astype(np.int16)
        red_mask = redness > 50
        processed = np.full_like(hint_img, 255)
        processed[red_mask] = hint_img[red_mask]

        # 2. 估算文字行高，确定膨胀核大小
        rows_with_red = np.any(red_mask, axis=1)
        line_heights = []
        lh_start = None
        for i, has in enumerate(rows_with_red):
            if has and lh_start is None:
                lh_start = i
            elif not has and lh_start is not None:
                line_heights.append(i - lh_start)
                lh_start = None
        if lh_start is not None:
            line_heights.append(len(rows_with_red) - lh_start)
        if not line_heights:
            if DEBUG_MODE:
                print("[RedTextOCR] 无红色像素")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail='{"texts":[]}')
        text_h = int(np.median(line_heights))
        kh, kw = 1, max(5, text_h)  # 垂直不膨胀，水平按行高膨胀连接同行文字

        # 3. 形态学膨胀 + 行列扫描分块
        dilated = _dilate(red_mask, kh, kw)
        rows_with_text = np.any(dilated, axis=1)
        line_ranges = []
        start = None
        for i, has_text in enumerate(rows_with_text):
            if has_text and start is None:
                start = i
            elif not has_text and start is not None:
                if i - start > 5:
                    line_ranges.append((start, i))
                start = None
        if start is not None and len(rows_with_text) - start > 5:
            line_ranges.append((start, len(rows_with_text)))
        blocks = []
        for r_start, r_end in line_ranges:
            cols_with_text = np.any(dilated[r_start:r_end], axis=0)
            col_start = None
            for j, has_text in enumerate(cols_with_text):
                if has_text and col_start is None:
                    col_start = j
                elif not has_text and col_start is not None:
                    if j - col_start > 5:
                        blocks.append([
                            max(0, col_start - 5),
                            max(0, r_start - 5),
                            j - col_start + 10,
                            r_end - r_start + 10
                        ])
                    col_start = None
            if col_start is not None and len(cols_with_text) - col_start > 5:
                blocks.append([
                    max(0, col_start - 5),
                    max(0, r_start - 5),
                    len(cols_with_text) - col_start + 10,
                    r_end - r_start + 10
                ])
        if not blocks:
            if DEBUG_MODE:
                print("[RedTextOCR] 无文本块")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail='{"texts":[]}')

        # 4. 逐块裁剪处理后图片，调用OCR识别
        texts = []
        for idx, block in enumerate(blocks):
            bx, by, bw, bh = block
            crop = processed[max(0, by):by + bh, max(0, bx):bx + bw].copy()
            reco = context.run_recognition("扫描选项", crop)
            text = ""
            if reco and reco.best_result:
                text = reco.best_result.text.strip()
            if DEBUG_MODE:
                print(f"[RedTextOCR] block[{idx}] roi={block} => \"{text}\"")
            texts.append(text)

        # 5. 保存调试图片（处理后图片 + 绿框标注）
        if DEBUG_MODE:
            vis = Image.fromarray(processed[:, :, ::-1]).convert("RGB")
            draw = ImageDraw.Draw(vis)
            for idx, block in enumerate(blocks):
                bx, by, bw, bh = block
                draw.rectangle([bx, by, bx + bw, by + bh], outline="lime", width=2)
                label = texts[idx] if idx < len(texts) else ""
                draw.text((bx, by - 14), label, fill="lime")
            os.makedirs("debug", exist_ok=True)
            vis.save("debug/hint_processed.png")
            if DEBUG_MODE:
                print("[RedTextOCR] 已保存 debug/hint_processed.png")

        # 6. 出血线合并：当前块右边缘靠右出血线 且 下一块左边缘靠左出血线，则合并
        i = 0
        while i < len(texts) - 1:
            right_edge = blocks[i][0] + blocks[i][2]
            left_edge = blocks[i + 1][0]
            if right_edge >= self.RIGHT_BLEED - self.BLEED_TH and left_edge <= self.LEFT_BLEED + self.BLEED_TH:
                texts[i] += texts[i + 1]
                nb = blocks[i + 1]
                bx = min(blocks[i][0], nb[0])
                by = min(blocks[i][1], nb[1])
                blocks[i] = [
                    bx, by,
                    max(blocks[i][0] + blocks[i][2], nb[0] + nb[2]) - bx,
                    max(blocks[i][1] + blocks[i][3], nb[1] + nb[3]) - by
                ]
                del texts[i + 1]
                del blocks[i + 1]
            else:
                i += 1
        result = [t for t in texts if t]
        if DEBUG_MODE:
            print(f"[RedTextOCR] 最终结果: {result}")
        return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail=json.dumps({"texts": result}))


resource.register_custom_recognition("RedTextOCR", RedTextOCR())


class MaaWorker:

    def __init__(
            self,
            queue: SimpleQueue,
            api_key,
            model: str,
    ):
        user_path = "./"
        Toolkit.init_option(user_path)

        self.queue = queue
        self.tasker = Tasker()
        self.connected = False
        self.api_key = api_key
        self.ai_resolver = AIResolver(
            api_key=api_key,
            model=model,
        )
        self.stop_flag = False
        self.pause_flag = False
        self.fast_answer = False
        self.send_log("MAA初始化成功")

    def update_ai_models(
            self,
            model: str | None = None,
    ):
        self.ai_resolver = AIResolver(
            api_key=self.api_key,
            model=model or self.ai_resolver.model,
        )

    def send_log(self, msg):
        self.queue.put(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {msg}")
        time.sleep(0.05)

    def pause(self):
        plyer.notification.notify(
            title="MaaXuexi",
            message="任务暂停，需要用户操作",
            app_name="MaaXuexi",
            timeout=60
        )
        self.send_log("任务暂停")
        self.pause_flag = True
        while self.pause_flag:
            time.sleep(0.05)

    @staticmethod
    def get_device():
        adb_devices = []
        for device in Toolkit.find_adb_devices():
            # 这两个字段的数字在JS里会整数溢出，转为字符串处理
            device.input_methods = str(device.input_methods)
            device.screencap_methods = str(device.screencap_methods)
            if device not in adb_devices:
                adb_devices.append(device)
        return adb_devices

    def connect_device(self, device):
        controller = AdbController(
            adb_path=device.adb_path,
            address=device.address,
            screencap_methods=device.screencap_methods,
            input_methods=device.input_methods,
            config=device.config,
        )
        status = controller.post_connection().wait().succeeded
        if not status:
            plyer.notification.notify(
                title="MaaXuexi",
                message="设备连接失败，请检查终端日志",
                app_name="MaaXuexi",
                timeout=30
            )
            self.send_log("设备连接失败，请检查终端日志")
            return self.connected
        if self.tasker.bind(resource, controller):
            self.connected = True
            # size = subprocess.run([device.adb_path, "shell", "wm", "size"], text=True, capture_output=True).stdout
            # size = size.strip().split(": ")[1]
            # dpi = subprocess.run([device.adb_path, "shell", "wm", "density"], text=True, capture_output=True).stdout
            # dpi = dpi.strip().split(": ")[1]
            # print(size,dpi)
            self.send_log("设备连接成功")
            self.send_log("正在启动 学习强国")
            controller.post_start_app("cn.xuexi.android").wait()
        else:
            plyer.notification.notify(
                title="MaaXuexi",
                message="设备连接失败，请检查终端日志",
                app_name="MaaXuexi",
                timeout=30
            )
            self.send_log("设备连接失败，请检查终端日志")
        return self.connected

    def detect(self):
        result: TaskDetail = self.tasker.post_task("yolo_detect").wait().get()
        if result.status.failed:
            return [], []
        details = result.nodes[0].recognition.raw_detail["all"]
        boxes, labels = [], []
        for detail in details:
            boxes.append(detail["box"])
            labels.append(detail["label"])
        return list(boxes), list(labels)

    def task(self, tasks, fast_answer=False, debug=False):
        global DEBUG_MODE
        DEBUG_MODE = debug
        self.stop_flag = False
        self.fast_answer = fast_answer
        self.send_log("任务开始")
        try:
            for task in tasks:
                if self.stop_flag:
                    self.send_log("任务已终止")
                    return
                if task == "选读文章":
                    self.read_article()
                elif task == "视听学习":
                    self.watch_video()
                elif task == "每日答题":
                    self.daily_answer()
                elif task == "趣味答题":
                    self.funny_answer()
            if self.stop_flag:
                self.send_log("任务已终止")
                return
        except Exception:
            traceback.print_exc()
            plyer.notification.notify(
                title="MaaXuexi",
                message="任务出现异常，请检查终端日志",
                app_name="MaaXuexi",
                timeout=30
            )
            self.send_log("任务出现异常，请检查终端日志")
            self.send_log("请将日志反馈至 https://github.com/ravizhan/MaaXuexi/issues")
        self.send_log("所有任务完成")
        time.sleep(0.5)


    def read_article(self):
        self.send_log("开始任务：选读文章")
        read_count = 0
        reading_time = 0
        self.send_log("进入板块 综合")
        self.tasker.post_task("综合").wait()
        time.sleep(randint(4, 5))
        while reading_time < 400:
            if self.stop_flag:
                return
            # 识别文章，获取点击文章的坐标范围
            image = self.tasker.controller.post_screencap().wait().get()
            boxes, box_class = self.detect()
            # 没有文章就滑动屏幕
            if len(boxes) == 0 or ("article" not in box_class and "article_image" not in box_class):
                self.send_log(f"未识别到文章，正在滑动屏幕")
                self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                                  randint(300, 400),
                                                  randint(1000, 1500)).wait()
                continue
            boxes, box_class = zip(
                *[(box, cls) for box, cls in zip(boxes, box_class) if cls in ["article", "article_image"]])
            self.send_log(f"识别到{len(boxes)}篇文章")
            article_list = []
            for box in boxes:
                img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                article_list.append(img)

            for i in range(len(box_class)):

                if self.stop_flag:

                    return
                if not self._has_unread_text(article_list[i]):
                    continue
                read_count += 1
                self.send_log(f"正在阅读第{read_count}篇文章")
                time.sleep(0.5)
                self.tasker.controller.post_click(boxes[i][0] + 150, boxes[i][1] + 10)
                time.sleep(3)
                for _ in range(5):
                    if self.stop_flag:
                        return
                    self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                                      randint(300, 400), randint(1000, 1500)).wait()
                    t = randint(8, 10)
                    time.sleep(t)
                    reading_time += t
                time.sleep(1)
                self.tasker.post_task("返回").wait()
                time.sleep(randint(3, 5))
            self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                              randint(300, 400), randint(1000, 1500)).wait()
        self.send_log("选读文章任务完成")


    def _has_unread_text(self, img: np.ndarray) -> bool:
        """检测文章卡片左上角 50%x50% 区域是否存在未读文章的黑色标题文字"""
        h, w = img.shape[:2]
        roi = img[:h // 2, :w // 2]
        # rgb(45, 51, 56) -> bgr(56, 51, 45)，容差 +-15
        lower = np.array([41, 36, 30], dtype=np.uint8)
        upper = np.array([71, 66, 60], dtype=np.uint8)
        mask = np.all((roi >= lower) & (roi <= upper), axis=2)
        count = np.count_nonzero(mask)
        unread = count > 80
        self.send_log(f"[颜色检测] 黑色像素={count}, 阈值=80, 判定={'未读' if unread else '已读'}")
        return unread


    def watch_video(self):
        self.send_log("开始任务：视听学习")
        watch_count = 0
        waiting_time = 0
        self.tasker.post_task("电视台").wait()
        time.sleep(randint(3, 5))
        while waiting_time < 400:
            if self.stop_flag:
                return
            # 识别视频，获取点击视频的坐标范围
            image = self.tasker.controller.post_screencap().wait().get()
            boxes, box_class = self.detect()
            # 没有视频就滑动屏幕
            if len(boxes) == 0 or "video" not in box_class:
                self.send_log(f"未识别到视频，正在滑动屏幕")
                self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                                  randint(300, 400),
                                                  randint(1000, 1500)).wait()
                continue
            boxes, box_class = zip(*[(box, cls) for box, cls in zip(boxes, box_class) if cls in ["video"]])
            self.send_log(f"识别到{len(boxes)}个视频")
            video_list = []
            for box in boxes:
                img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                video_list.append(img)

            for i in range(len(box_class)):
                if self.stop_flag:
                    return
                if not self._has_unread_text(video_list[i]):
                    continue
                watch_count += 1
                self.send_log(f"正在播放第{watch_count}个视频")
                time.sleep(0.5)
                self.tasker.controller.post_click(boxes[i][0] + 150, boxes[i][1] + 10)
                time.sleep(3)
                t = randint(50, 70)
                time.sleep(t)
                waiting_time += t
                self.tasker.post_task("返回2").wait()
                time.sleep(randint(3, 5))
            self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                              randint(300, 400), randint(1000, 1500)).wait()
        self.send_log("视听学习任务完成")

    def _navigate_to_daily_answer(self):
        self.send_log("开始任务：每日答题")
        self.tasker.post_task("我的").wait()
        wait_time = 0.5 + randint(0, 500) / 1000
        time.sleep(wait_time)
        learning_score_result: TaskDetail = self.tasker.post_task("学习积分").wait().get()
        if not learning_score_result.nodes:
            self.send_log("未找到学习积分按钮")
            self.tasker.post_task("返回2").wait()
            time.sleep(1)
            self.tasker.post_task("积分").wait()
        else:
            box = learning_score_result.nodes[0].recognition.best_result.box
            self.tasker.controller.post_click(box[0] + randint(10, 30), box[1] + randint(10, 30))
        time.sleep(10)
        load_result: TaskDetail = self.tasker.post_task("加载失败").wait().get()
        while not load_result.nodes:
            if self.stop_flag:
                return
            self.send_log("积分界面加载失败，正在重试")
            self.tasker.post_task("返回").wait()
            self.tasker.post_task("积分").wait()
            time.sleep(10)
            load_result: TaskDetail = self.tasker.post_task("加载失败").wait().get()
        self.send_log("加载成功")
        self.tasker.controller.post_swipe(randint(200, 300), randint(1000, 1100), randint(500, 600), randint(100, 200),
                                          randint(1000, 1500)).wait()
        time.sleep(randint(1, 2))
        self._click_daily_answer_button()

    def _click_daily_answer_button(self):
        result: TaskDetail = self.tasker.post_task("每日答题").wait().get()
        box = result.nodes[0].recognition.best_result.box
        self.tasker.controller.post_click(box[0] + randint(10, 30), box[1] + randint(10, 30))
        self.send_log("开始答题")
        if self.stop_flag:
            return
        time.sleep(5)

    def _check_answer_correctness(self, question_index) -> bool:
        time.sleep(1.5 + randint(0, 500) / 1000)
        next_btn = self.tasker.post_task("下一题检测").wait().get()
        if next_btn.nodes:
            self.send_log(f"[正误检测] 检测到下一题按钮，判定为答错")
            return False
        self.send_log(f"[正误检测] 第{question_index + 1}题答对")
        return True

    def _handle_wrong_answer(self):
        self.tasker.post_task("返回3").wait()
        time.sleep(1)
        popup_result: TaskDetail = self.tasker.post_task("放弃答题").wait().get()
        if popup_result.nodes:
            popup_box = popup_result.nodes[0].recognition.best_result.box
            exit_result: TaskDetail = self.tasker.post_task("放弃答题退出", {"放弃答题退出": {"roi": list(popup_box)}}).wait().get()
        else:
            exit_result: TaskDetail = self.tasker.post_task("放弃答题退出").wait().get()
        if exit_result.nodes:
            exit_box = exit_result.nodes[0].recognition.best_result.box
            cx = exit_box[0] + exit_box[2] // 2
            cy = exit_box[1] + exit_box[3] // 2
            self.tasker.controller.post_click(cx, cy).wait()
        time.sleep(2)

    def daily_answer(self):
        self._navigate_to_daily_answer()
        if self.stop_flag:
            return

        retry_count = 0
        max_retries = 1

        while True:
            if self.stop_flag:
                return

            first_q = self.tasker.post_task("第一题").wait().get()
            if not first_q.nodes:
                self.send_log("未找到第一题，答题页面异常")
                plyer.notification.notify(
                    title="MaaXuexi",
                    message="未找到第一题，答题页面异常",
                    app_name="MaaXuexi",
                    timeout=60
                )
                return

            fast_mode_this_round = self.fast_answer
            all_correct = True

            for i in range(5):
                if self.stop_flag:
                    return

                proceed_next = False

                single_result = self.tasker.post_task("单选题").wait().get()
                if single_result.nodes:
                    self.send_log(f"第{i + 1}题 单选题")
                    proceed_next = self._handle_single_choice(i)
                else:
                    multi_result = self.tasker.post_task("多选题").wait().get()
                    if multi_result.nodes:
                        self.send_log(f"第{i + 1}题 多选题")
                        proceed_next = self._handle_multi_choice(i)
                    else:
                        fill_result = self.tasker.post_task("填空题").wait().get()
                        if fill_result.nodes:
                            self.send_log(f"第{i + 1}题 填空题")
                            proceed_next = self._handle_fill_blank(i)
                        else:
                            click_blank_result = self.tasker.post_task("点选填空题").wait().get()
                            if click_blank_result.nodes:
                                self.send_log(f"第{i + 1}题 点选填空题")
                                proceed_next = self._handle_click_blank(i)
                            else:
                                self.send_log(f"第{i + 1}题 无法识别题型，请求接管")
                                plyer.notification.notify(
                                    title="MaaXuexi",
                                    message="无法识别题型，请求接管",
                                    app_name="MaaXuexi",
                                    timeout=60
                                )
                                self.pause()
                                return

                if proceed_next:
                    time.sleep(1)
                    self.tasker.post_task("下一题").wait()
                    if not self._check_answer_correctness(i):
                        self.send_log("检测到答错，重新答题")
                        self._handle_wrong_answer()
                        if fast_mode_this_round:
                            self.fast_answer = False
                            self.send_log("极速模式已关闭，切换为常规答题")
                        all_correct = False
                        break

            if all_correct:
                break

            retry_count += 1
            if retry_count > max_retries:
                self.send_log("重试次数已达上限，答题失败")
                plyer.notification.notify(
                    title="MaaXuexi",
                    message="重试次数已达上限，答题失败",
                    app_name="MaaXuexi",
                    timeout=60
                )
                return

            self.send_log(f"第 {retry_count} 次重试")
            self._click_daily_answer_button()

        time.sleep(2)
        recog_result: TaskDetail = self.tasker.post_task("访问异常").wait().get()
        if not recog_result.nodes:
            plyer.notification.notify(
                title="MaaXuexi",
                message="发现验证码，请求接管",
                app_name="MaaXuexi",
                timeout=60
            )
            self.send_log("发现验证码，请求接管")
            self.pause()

            if proceed_next:
                time.sleep(0.5)
                self.tasker.post_task("下一题").wait()
                time.sleep(randint(2, 3))
        # 结束答题，大概率会弹验证码
        time.sleep(2)
        recog_result: TaskDetail = self.tasker.post_task("访问异常").wait().get()
        if not recog_result.nodes:
            plyer.notification.notify(
                title="MaaXuexi",
                message="发现验证码，请求接管",
                app_name="MaaXuexi",
                timeout=60
            )
            self.send_log("发现验证码，请求接管")
            self.pause()

    def _scroll_to_hint(self, max_swipes=2) -> list[np.ndarray]:
        screenshots = []
        scroll_px = 700
        for i in range(max_swipes + 1):
            shot = self.tasker.controller.post_screencap().wait().get()
            if i == 0:
                screenshots.append(shot)
            else:
                h = shot.shape[0]
                screenshots.append(shot[max(0, h - scroll_px):, :])
            result = self.tasker.post_task("查看提示").wait().get()
            if result.nodes:
                return screenshots
            if i < max_swipes:
                self.send_log(f"未找到提示按钮，向下滑动 ({i + 1}/{max_swipes})")
                self.tasker.controller.post_swipe(randint(300, 400), randint(1000, 1100), randint(300, 400),
                                                  randint(300, 400), randint(300, 500)).wait()
                time.sleep(0.5)
        self.send_log("多次滑动仍未找到提示按钮，直接提交AI")
        return screenshots


    def _get_red_texts(self) -> list[str]:
        """调用 RedTextOCR 自定义识别器，从提示区域提取红色文字"""
        result: TaskDetail = self.tasker.post_task("红字识别").wait().get()
        texts = []
        if result.nodes:
            raw = result.nodes[0].recognition.raw_detail
            if raw:
                if isinstance(raw, dict):
                    best = raw.get("best", {})
                    if isinstance(best, dict):
                        detail = best.get("detail", {})
                        if isinstance(detail, dict):
                            texts = detail.get("texts", [])
                    if not texts:
                        for key in ["all", "filtered"]:
                            for item in raw.get(key, []):
                                if isinstance(item, dict):
                                    d = item.get("detail", {})
                                    if isinstance(d, dict) and "texts" in d:
                                        texts = d["texts"]
                                        break
                                if texts:
                                    break
        print(f"[识别] 红色文字: {texts}")
        return texts

    def _count_blanks(self) -> int:
        """遍历所有文本框匹配结果，统计格子总数"""
        count = 0
        for i in range(10):
            result: TaskDetail = self.tasker.post_task("文本框计数", {"文本框计数": {"index": i}}).wait().get()
            if not result.nodes:
                break
            count += 1
        return count

    def _scan_click_options(self) -> dict[str, list]:
        """点选填空题选项识别：模板匹配1-4字选项框，内缩后OCR识别文字"""
        options = {}
        for n in [1, 2, 3, 4]:
            for i in range(20):
                result: TaskDetail = self.tasker.post_task(f"点选{n}字", {f"点选{n}字": {"index": i}}).wait().get()
                if not result.nodes:
                    break
                box = list(result.nodes[0].recognition.best_result.box)
                x, y, w, h = box
                roi = [max(0, x + 10), max(0, y + 10), max(1, w - 20), max(1, h - 20)]
                ocr_result: TaskDetail = self.tasker.post_task("扫描选项", {"扫描选项": {"roi": roi}}).wait().get()
                if not ocr_result.nodes:
                    continue
                text = ocr_result.nodes[0].recognition.best_result.text.strip()
                if text:
                    options[text] = box
                    if DEBUG_MODE:
                        print(f"[点选] {n}字模板 box={box} roi={roi} => \"{text}\"")
        return options

    def _fast_try_answer(self, question_type: str, options: dict, red_texts: list, blank_num: int = 0):
        if not options and question_type not in ("填空题", "点选填空题"):
            return None
        if not red_texts:
            return None
        red_texts = [t for t in red_texts if t.strip()]
        if not red_texts:
            return None
        if question_type == "多选题":
            option_letters = list(options.keys())
            if len(option_letters) <= len(red_texts):
                print(f"[极速] 多选题: 选项数{len(option_letters)} <= 红字组数{len(red_texts)}, 全选 {option_letters}")
                return option_letters
            print(f"[极速] 多选题: 选项数{len(option_letters)} > 红字组数{len(red_texts)}, 交给AI")
            return None
        if question_type == "填空题":
            answer = "".join(red_texts)
            if blank_num > 0 and len(answer) == blank_num:
                print(f"[极速] 填空题: 答案=\"{answer}\", 字数={len(answer)}, 格子={blank_num}")
                return answer
            print(f"[极速] 填空题: 字数{len(answer)} != 格子{blank_num}, 交给AI")
            return None
        if question_type == "点选填空题":
            answer = "".join(red_texts)
            if answer:
                print(f"[极速] 点选填空题: 答案=\"{answer}\"")
                return answer
            return None
        if question_type != "单选题":
            return None
        red_texts = [t for t in red_texts if t.strip()]
        if not red_texts:
            return None
        option_texts = {l: options[l][0] for l in options}
        judge_words = {"正确", "错误", "对", "错", "√", "×"}
        is_judge = all(t in judge_words for t in option_texts.values())
        if is_judge:
            combined = "".join(red_texts)
            print(f"[极速] 判断题, 红字=\"{combined}\"")
            if combined == "正确":
                for letter, text in option_texts.items():
                    if text == "正确":
                        print(f"[极速] 判断题直接匹配: {letter} ({text})")
                        return [letter]
            target_words = {"正确", "对", "√"} if len(combined) > 5 else {"错误", "错", "×"}
            for letter, text in option_texts.items():
                if text in target_words:
                    print(f"[极速] 判断题推测: {letter} ({text})")
                    return [letter]
            return None
        import re
        combined = "".join(red_texts)
        print(f"[极速] 选择题, 红字=\"{combined}\", 选项={option_texts}")
        def strip_punct(s):
            return re.sub(r'[^\w]', '', s)
        combined_clean = strip_punct(combined)
        def fuzzy_match(a: str, b: str) -> bool:
            if a == b:
                return True
            if a in b or b in a:
                return min(len(a), len(b)) / max(len(a), len(b)) >= 2 / 3
            return False
        for letter, text in option_texts.items():
            text_clean = strip_punct(text)
            if not text_clean:
                if text in combined:
                    print(f"[极速] 标点匹配: {letter} ({text})")
                    return [letter]
            else:
                if fuzzy_match(text_clean, combined_clean):
                    print(f"[极速] 文字匹配: {letter} ({text})")
                    return [letter]
        if len(red_texts) > 1:
            from itertools import permutations
            for perm in permutations(red_texts):
                perm_text = "".join(perm)
                perm_clean = strip_punct(perm_text)
                for letter, text in option_texts.items():
                    text_clean = strip_punct(text)
                    if not text_clean:
                        if text in perm_text:
                            print(f"[极速] 排列标点匹配: {letter} ({text}), 排列=\"{perm_text}\"")
                            return [letter]
                    else:
                        if fuzzy_match(text_clean, perm_clean):
                            print(f"[极速] 排列文字匹配: {letter} ({text}), 排列=\"{perm_text}\"")
                            return [letter]
        self.send_log("[极速答题] 极速答题失败，原因：无法匹配，请求AI解答")
        return None

    def _get_options(self) -> dict[str, tuple[str, list]]:
        options = {}
        found = {}
        for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
            find_result: TaskDetail = self.tasker.post_task(f"查找{letter}").wait().get()
            if find_result.nodes:
                box = list(find_result.nodes[0].recognition.best_result.box)
                found[letter] = box
        if not found:
            print("[识别] 选项: 未找到任何选项")
            return options
        print(f"[识别] 找到选项: {list(found.keys())}")
        for letter, box in found.items():
            roi = [box[0] + box[2], box[1], 720 - box[0] - box[2], box[3]]
            ocr_result: TaskDetail = self.tasker.post_task("扫描选项", {"扫描选项": {"roi": roi}}).wait().get()
            text = ""
            if ocr_result.nodes:
                text = ocr_result.nodes[0].recognition.best_result.text.strip()
                for noise in ["銀園", "銀", "電", "機"]:
                    if text.endswith(noise):
                        text = text[:-len(noise)].strip()
            print(f"[识别] {letter} => \"{text}\"")
            if text:
                options[letter] = (text, box)
        print(f"[识别] 选项: {options}")
        return options

    def _prepare(self, question_type: str) -> tuple:
        options = {}
        blank_num = 0
        if self.fast_answer and question_type in ("单选题", "多选题"):
            options = self._get_options()
        if question_type == "填空题":
            blank_num = self._count_blanks()
        hint_found = self.tasker.post_task("查找提示").wait().get()
        if not hint_found.nodes:
            self.send_log("未找到提示按钮，滑动")
            self.tasker.controller.post_swipe(
                randint(300, 400), randint(1000, 1100),
                randint(300, 400), randint(300, 400), randint(300, 500)
            ).wait()
            time.sleep(0.5)
            if self.fast_answer and question_type in ("单选题", "多选题"):
                new_options = self._get_options()
                for k, v in new_options.items():
                    options[k] = v
        scroll_shots = self._scroll_to_hint()
        time.sleep(1)
        red_texts = []
        if self.fast_answer:
            red_texts = self._get_red_texts()
        return scroll_shots, options, red_texts, blank_num

    def _determine_answer(self, question_type: str, options: dict, red_texts: list, scroll_shots: list, blank_num: int = 0):
        fast_answer = self._fast_try_answer(question_type, options, red_texts, blank_num)
        if fast_answer is not None:
            self.tasker.post_task("关闭提示").wait()
            time.sleep(1)
            return fast_answer, True
        if question_type == "填空题" and red_texts:
                answer = "".join(red_texts)
                if len(answer) == blank_num:
                    self.tasker.post_task("关闭提示").wait()
                    time.sleep(1)
                    return answer, True
        img_list = [scroll_shots[-1], self.tasker.controller.post_screencap().wait().get()]
        self.tasker.post_task("关闭提示").wait()
        time.sleep(1)
        if question_type in ("单选题", "多选题"):
            if DEBUG_MODE:
                print("[AI] 请求选择题解答")
            answer = self.ai_resolver.resolve_choice(img_list)
        elif question_type == "填空题":
            if DEBUG_MODE:
                print("[AI] 请求填空题解答")
            answer = self.ai_resolver.resolve_blank(img_list, False, blank_num)
        elif question_type == "点选填空题":
            if DEBUG_MODE:
                print("[AI] 请求点选填空题解答")
            answer = self.ai_resolver.resolve_click_blank(img_list)
        else:
            answer = None
        if answer is None:
            plyer.notification.notify(
                title="MaaXuexi",
                message="AI解答失败，请求接管",
                app_name="MaaXuexi",
                timeout=60
            )
            self.send_log("AI解答失败, 请求接管")
            self.pause()
            return None, False
        return answer, False

    def _submit_answer(self, question_type: str, answer, from_fast: bool) -> bool:
        if question_type in ("单选题", "多选题"):
            if from_fast:
                self.send_log(f"极速模式解答成功，答案为{''.join(answer)}")
            else:
                self.send_log(f"AI解答成功，答案为{''.join(answer)}")
            failed = []
            for choice in answer:
                result: TaskDetail = self.tasker.post_task(f"选{choice}").wait().get()
                if result.nodes:
                    time.sleep(0.2)
                else:
                    failed.append(choice)
            if failed:
                print(f"[选择题] 选项 {failed} 可能被遮挡，下滑重试")
                self.tasker.controller.post_swipe(
                    randint(300, 400), randint(600, 700),
                    randint(300, 400), randint(300, 400), randint(200, 300)
                ).wait()
                time.sleep(0.3)
                for choice in failed:
                    self.tasker.post_task(f"选{choice}").wait()
                    time.sleep(0.2)
            return True
        if question_type == "填空题":
            if from_fast:
                self.send_log(f"极速模式解答成功: {answer}")
            self.send_log(f"正在输入 {answer}")
            self.tasker.post_task("文本框点击").wait()
            time.sleep(0.5)
            self.tasker.controller.post_input_text(answer).wait()
            self.send_log("输入完成")
            return True
        if question_type == "点选填空题":
            if from_fast:
                return self._fast_click_blanks(answer)
            answers = answer if isinstance(answer, list) else [a.strip() for a in answer.replace("，", ",").split(",") if a.strip()]
            self.send_log(f"AI识别到 {len(answers)} 个答案: {answers}")
            return self._click_text_answers(answers)
        return False

    def _fast_click_blanks(self, answer: str) -> bool:
        """极速点选：识别选项位置，按顺序匹配答案点选，颜色变动验证"""
        text_positions = self._scan_click_options()
        if not text_positions:
            self.send_log("极速点选: 未识别到选项, 交给AI")
            return False
        self.send_log(f"极速点选: 答案=\"{answer}\", 选项={list(text_positions.keys())}")
        clicked_any = False
        remaining = answer
        for text, box in text_positions.items():
            if text not in remaining:
                continue
            cx = box[0] + box[2] // 2
            cy = box[1] + box[3] // 2
            img_before = self.tasker.controller.post_screencap().wait().get()
            color_before = img_before[cy, cx].tolist()
            print(f"[点选填空题] 点击 \"{text}\" 位置({cx},{cy})")
            self.tasker.controller.post_click(cx, cy).wait()
            time.sleep(0.5)
            img_after = self.tasker.controller.post_screencap().wait().get()
            color_after = img_after[cy, cx].tolist()
            diff = sum(abs(int(a) - int(b)) for a, b in zip(color_before, color_after))
            if diff > 50:
                clicked_any = True
                remaining = remaining.replace(text, "", 1)
                print(f"[点选填空题] \"{text}\" 选中 (差值={diff})")
            else:
                self.send_log(f"\"{text}\" 未选中 (差值={diff}), 请求接管")
                plyer.notification.notify(title="MaaXuexi", message=f"选项 \"{text}\" 选中失败", app_name="MaaXuexi", timeout=60)
                self.pause()
                return True
        if not clicked_any:
            self.send_log("极速点选: 未匹配任何选项, 交给AI")
            return False
        self.send_log("极速点选完成")
        return True

    def _click_text_answers(self, answers: list[str]) -> bool:
        """点选填空题普通流程：AI返回选项文本，识别位置后逐个点选"""
        text_positions = self._scan_click_options()
        if not text_positions:
            self.send_log("未识别到选项, 请求接管")
            plyer.notification.notify(title="MaaXuexi", message="未识别到选项文本", app_name="MaaXuexi", timeout=60)
            self.pause()
            return True
        self.send_log(f"选项: {list(text_positions.keys())}, 答案: {answers}")
        clicked_any = False
        for ans in answers:
            target_box = None
            matched_text = None
            for text, box in text_positions.items():
                if ans == text or ans in text or text in ans:
                    target_box = box
                    matched_text = text
                    break
            if target_box is None:
                self.send_log(f"未找到 \"{ans}\", 申请支援")
                plyer.notification.notify(title="MaaXuexi", message=f"未找到选项 \"{ans}\"", app_name="MaaXuexi", timeout=60)
                self.pause()
                return True
            cx = target_box[0] + target_box[2] // 2
            cy = target_box[1] + target_box[3] // 2
            img_before = self.tasker.controller.post_screencap().wait().get()
            color_before = img_before[cy, cx].tolist()
            print(f"[点选填空题] 点击 \"{matched_text}\" 位置({cx},{cy})")
            self.tasker.controller.post_click(cx, cy).wait()
            time.sleep(0.5)
            img_after = self.tasker.controller.post_screencap().wait().get()
            color_after = img_after[cy, cx].tolist()
            diff = sum(abs(int(a) - int(b)) for a, b in zip(color_before, color_after))
            if diff > 50:
                clicked_any = True
                print(f"[点选填空题] \"{matched_text}\" 选中 (差值={diff})")
            else:
                self.send_log(f"\"{matched_text}\" 未选中 (差值={diff}), 请求接管")
                plyer.notification.notify(title="MaaXuexi", message=f"选项 \"{matched_text}\" 选中失败", app_name="MaaXuexi", timeout=60)
                self.pause()
                return True
        self.send_log("点选完成")
        return True

    def _handle_fill_blank(self, index) -> bool:
        if self.stop_flag:
            return False
        video_result: TaskDetail = self.tasker.post_task("填空题视频").wait().get()
        if not video_result.nodes:
            self.send_log("发现视频，正在请求AI解答")
            image = self.tasker.controller.post_screencap().wait().get()
            if DEBUG_MODE:
                print("[AI] 请求填空题解答(视频)")
            answer = self.ai_resolver.resolve_blank([image], False)
            if answer is None:
                plyer.notification.notify(
                    title="MaaXuexi",
                    message="AI解答失败，请求接管",
                    app_name="MaaXuexi",
                    timeout=60
                )
                self.send_log("AI解答失败, 请求接管")
                self.pause()
                return False
            return self._submit_answer("填空题", answer, False)
        scroll_shots, options, red_texts, blank_num = self._prepare("填空题")
        answer, from_fast = self._determine_answer("填空题", options, red_texts, scroll_shots, blank_num)
        if answer is None or answer == "":
            return False
        return self._submit_answer("填空题", answer, from_fast)

    def _handle_single_choice(self, index, question_type="单选题") -> bool:
        if self.stop_flag:
            return False
        scroll_shots, options, red_texts, blank_num = self._prepare(question_type)
        answer, from_fast = self._determine_answer(question_type, options, red_texts, scroll_shots, blank_num)
        if answer is None:
            return False
        return self._submit_answer(question_type, answer, from_fast)

    def _handle_multi_choice(self, index) -> bool:
        return self._handle_single_choice(index, "多选题")

    def _handle_click_blank(self, index) -> bool:
        if self.stop_flag:
            return False
        scroll_shots, options, red_texts, blank_num = self._prepare("点选填空题")
        answer, from_fast = self._determine_answer("点选填空题", options, red_texts, scroll_shots, blank_num)
        if answer is None:
            return False
        return self._submit_answer("点选填空题", answer, from_fast)

    def funny_answer(self):
        self.send_log("开始任务：趣味答题")
        pass