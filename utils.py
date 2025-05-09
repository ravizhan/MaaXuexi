import json
import time
import traceback
from base64 import b64encode
from io import BytesIO
from queue import SimpleQueue
from random import randint

import numpy as np
import plyer
from PIL import Image
from PIL import ImageFilter
from httpx import Client
from maa.controller import AdbController
from maa.custom_recognition import CustomRecognition
from maa.define import TaskDetail
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit


class AIResolver:
    def __init__(self, api_key):
        self.session = Client()
        self.session.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = "https://api.siliconflow.cn/v1/chat/completions"

    @staticmethod
    def image_encode(img: np.ndarray) -> str:
        buffered = BytesIO()
        im = Image.fromarray(img)
        new_size = list(map(lambda x: round(x * 0.5), list(im.size)))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        im = im.filter(ImageFilter.SHARPEN)
        im.save(buffered, format="JPEG")
        encoded_image = b64encode(buffered.getvalue()).decode()
        return encoded_image

    @staticmethod
    def image_combine(imgs: list[np.ndarray]) -> np.ndarray:
        # 创建一个新的空白图片
        new_img = np.zeros((1280, len(imgs) * 720, 3), dtype=np.uint8)
        # 将每张图片粘贴到新图片上
        x_offset = 0
        for img in imgs:
            new_img[:img.shape[0], x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1]
        return new_img

    def resolve_choice(self, imgs: list[np.ndarray]) -> list[str] | None:
        data = {
            "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
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
                            "url": "data:image/jpg;base64," + self.image_encode(self.image_combine(imgs))
                        },
                    }]
                }
            ],
            "temperature": 0.2
        }
        response = self.session.post(self.url, json=data)
        try:
            if response.status_code == 200:
                result = response.json()
                answer = list(result["choices"][0]["message"]["content"])
                for i in answer.copy():
                    if i not in ['A', 'B', 'C', 'D', 'E']:
                        answer.remove(i)
                if len(answer) == 0:
                    raise ValueError("Invalid answer")
            else:
                print(response.json())
                answer = None
        except:
            print(response.json())
            answer = None
        return answer

    def resolve_blank(self, imgs: list[np.ndarray], answer: bool, blank_num: int) -> str | None:
        data = {
            "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
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
                            "url": "data:image/jpg;base64," + self.image_encode(self.image_combine(imgs))
                        },
                    }]
                }
            ],
            "temperature": 0.2
        }
        if not answer:
            data["messages"][0]["content"] = f"能力与角色:你是一位答题助手\n背景信息:你会得到一张包含填空题的图片\n指令:你需要阅读该图片中的问题，认真理解题目和前后文，其中答案为{blank_num}个字符，思考后作出回答，确保填入答案后的全文逻辑正确，语义正确\n输出风格:你无需给出推理过程，也无需给出任何解释。你只需要回答空缺处应当填的内容，填充字数应当为{blank_num}"
            data["model"] = "Qwen/Qwen2.5-VL-32B-Instruct"
        response = self.session.post(self.url, json=data)
        try:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
            else:
                answer = None
        except:
            answer = None
        return answer


resource = Resource()
resource.set_cpu()
resource.post_bundle("./resource").wait()


class MaaWorker:
    def __init__(self, queue: SimpleQueue, api_key):
        user_path = "./"
        Toolkit.init_option(user_path)

        self.queue = queue
        self.tasker = Tasker()
        self.connected = False
        self.ai_resolver = AIResolver(api_key=api_key)
        self.stop_flag = False
        self.pause_flag = False
        self.send_log("MAA初始化成功")

    def send_log(self, msg):
        self.queue.put(f"{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {msg}")
        time.sleep(0.05)

    def pause(self):
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

    def similarity_match(self, img1_path: str, img2_path: str) -> bool:
        pipeline = {
            "similarity": {
                "recognition": "custom",
                "custom_recognition": "SimilarityReco",
                "custom_recognition_param": {"origin": img1_path, "pic": "../../" + img2_path}
            }
        }
        result: TaskDetail = self.tasker.post_task("similarity", pipeline).wait().get()
        return result.nodes[0].recognition.best_result.detail == "failed"

    def task(self, tasks):
        self.stop_flag = False
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
        finished_article = []
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
                Image.fromarray(article_list[i][:, :, ::-1]).save("current.jpg", "JPEG")
                if all(self.similarity_match("current.jpg", img2) for img2 in finished_article):
                    self.send_log(f"read_{len(finished_article)}")
                    Image.fromarray(article_list[i][:, :, ::-1]).save(f"read_{len(finished_article)}.jpg", "JPEG")
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
                    finished_article.append(f"read_{len(finished_article)}.jpg")
            self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                              randint(300, 400), randint(1000, 1500)).wait()
        self.send_log("选读文章任务完成")

    def watch_video(self):
        self.send_log("开始任务：视听学习")
        finished_video = []
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
                Image.fromarray(video_list[i][:, :, ::-1]).save("current.jpg", "JPEG")
                time.sleep(0.5)
                if all(self.similarity_match("current.jpg", img2) for img2 in finished_video):
                    if self.stop_flag:
                        return
                    self.send_log(f"video_{len(finished_video)}")
                    Image.fromarray(video_list[i][:, :, ::-1]).save(f"video_{len(video_list)}.jpg", "JPEG")
                    self.tasker.controller.post_click(boxes[i][0] + 150, boxes[i][1] + 10)
                    time.sleep(3)
                    t = randint(50, 70)
                    time.sleep(t)
                    waiting_time += t
                    self.tasker.post_task("返回2").wait()
                    time.sleep(randint(3, 5))
                    finished_video.append(f"video_{len(video_list)}.jpg")
            self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                              randint(300, 400), randint(1000, 1500)).wait()
        self.send_log("视听学习任务完成")

    def daily_answer(self):
        self.send_log("开始任务：每日答题")
        self.tasker.post_task("积分").wait()
        # 等待界面加载完毕
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
        # 滑动到每日答题按钮
        self.tasker.controller.post_swipe(randint(200, 300), randint(1000, 1100), randint(500, 600), randint(100, 200),
                                          randint(1000, 1500)).wait()
        time.sleep(randint(1, 2))
        # 点击每日答题按钮
        result: TaskDetail = self.tasker.post_task("每日答题").wait().get()
        box = result.nodes[0].recognition.best_result.box
        self.tasker.controller.post_click(box[0] + randint(10, 30), box[1] + randint(10, 30))
        self.send_log("开始答题")
        if self.stop_flag:
            return
        # 等待界面加载完毕
        time.sleep(5)
        # 开始答题
        for i in range(5):
            if self.stop_flag:
                return
            # 判断是不是填空题
            recog_result: TaskDetail = self.tasker.post_task("填空题").wait().get()  # 单选题和填空题相似度竟然有0.75，离谱
            if not recog_result.nodes:
                self.send_log(f"第{i + 1}题 填空题")
                recog_result: TaskDetail = self.tasker.post_task("填空题视频").wait().get()
                # 判断有没有视频，有的话调用AI解答
                if not recog_result.nodes:
                    self.send_log("发现视频，正在请求AI解答")
                    # 截图
                    image = self.tasker.controller.post_screencap().wait().get()
                    # AI解答
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
                        continue
                else:
                    # 正常填空题
                    self.send_log("查看提示")
                    click_result: TaskDetail = self.tasker.post_task("查看提示").wait().get()
                    if not click_result.nodes:
                        self.tasker.controller.post_swipe(randint(590, 600), randint(1200, 1210), randint(620, 630),
                                                          randint(1000, 1010), randint(300, 400)).wait()
                    time.sleep(1)
                    find_result: TaskDetail = self.tasker.post_task("find_red").wait().get()
                    red_border = find_result.nodes[0].recognition.best_result.box
                    rec_result: TaskDetail = self.tasker.post_task("rec_answer",
                                                                   {"rec_answer": {"roi": red_border}}).wait().get()
                    answer = rec_result.nodes[0].recognition.best_result.text
                    self.tasker.post_task("关闭提示").wait()
                time.sleep(1)
                self.send_log(f"正在输入 {answer}")
                self.tasker.post_task("文本框点击").wait()
                time.sleep(0.5)
                self.tasker.controller.post_input_text(answer).wait()
                self.send_log("输入完成")
            else:
                self.send_log(f"第{i + 1}题 选择题")
                img_list = []
                # 问题截图
                img_list.append(self.tasker.controller.post_screencap().wait().get())
                # 答案截图
                click_result: TaskDetail = self.tasker.post_task("查看提示").wait().get()
                if not click_result.nodes:
                    self.tasker.controller.post_swipe(randint(590, 600), randint(1200, 1210), randint(620, 630),
                                                      randint(1100, 1110), randint(200, 300)).wait()
                    self.tasker.post_task("查看提示").wait()
                    img_list.append(self.tasker.controller.post_screencap().wait().get())
                time.sleep(1)
                img_list.append(self.tasker.controller.post_screencap().wait().get())
                self.tasker.post_task("关闭提示").wait()
                time.sleep(1)
                # AI解答
                answer = self.ai_resolver.resolve_choice(img_list)
                if answer is None:
                    plyer.notification.notify(
                        title="MaaXuexi",
                        message="AI解答失败，请求接管",
                        app_name="MAA",
                        timeout=60
                    )
                    self.send_log("AI解答失败, 请求接管")
                    self.pause()
                    continue
                self.send_log(f"AI解答成功，答案为{''.join(answer)}")
                for i in answer:
                    if i == "A":
                        self.tasker.post_task("选A").wait()
                    elif i == "B":
                        self.tasker.post_task("选B").wait()
                    elif i == "C":
                        self.tasker.post_task("选C").wait()
                    elif i == "D":
                        self.tasker.post_task("选D").wait()
                    elif i == "E":
                        self.tasker.post_task("选E").wait()
                    time.sleep(0.2)
            time.sleep(0.5)
            # 下一题
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

    def funny_answer(self):
        self.send_log("开始任务：趣味答题")
        pass


@resource.custom_recognition("SimilarityReco")
class SimilarityReco(CustomRecognition):
    def analyze(
            self,
            context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        img1 = json.loads(argv.custom_recognition_param)["origin"]
        img1 = np.asarray(Image.open(img1))
        img2 = json.loads(argv.custom_recognition_param)["pic"]
        reco_detail = context.run_recognition(
            "test_template",
            img1,
            {
                "test_template":
                    {
                        "recognition": "FeatureMatch",
                        "template": img2,
                        "count": 200,
                        "pre_delay": 0,
                        "post_delay": 0
                    }
            },
        )
        if reco_detail is None:
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 0, 0), detail="failed"
            )
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 0, 0), detail="success"
        )
