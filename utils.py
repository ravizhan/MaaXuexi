import base64
import time
from queue import SimpleQueue
from random import randint

import cv2
import httpx
import numpy as np
import onnxruntime
import plyer
from maa.controller import AdbController
from maa.define import TaskDetail
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit


def match_sift_flann(image1, image2):
    """
    使用 SIFT 和 FLANN 匹配器匹配两幅图像的特征点.

    :param image1: 第一幅图像 (ndarray)
    :param image2: 第二幅图像 (ndarray)
    :return: 匹配结果图像和相似度分数
    """
    # 确保图像为灰度图
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 创建 SIFT 检测器
    sift = cv2.SIFT.create()

    # 检测特征点和计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # FLANN 参数设置
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # 创建 FLANN 匹配器
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行特征匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 进行 Lowe's ratio test 来筛选好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 可视化匹配结果
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 计算相似度
    similarity = len(good_matches) / min(len(keypoints1), len(keypoints2))
    # print("Similarity score: ", similarity)
    return matched_image, similarity

def letterbox(img, new_shape, color=(114, 114, 114)):
    """
    将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
    """
    shape = img.shape[:2]  # 当前图像的宽高
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
    # 缩放后的未填充尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    # 计算需要的填充
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
    dw /= 2  # padding 均分
    dh /= 2
    # 缩放图像
    if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 为图像添加边框以达到目标尺寸
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


class ONNXModel:
    def __init__(self):
        self.session = onnxruntime.InferenceSession("resource/model/detect/yolo.onnx")
        self.model_input = self.session.get_inputs()
        self.classes = {0: 'article', 1: 'article_image', 2: 'article_image_big', 3: 'video', 4: 'video_big'}

    def detect(self, img: np.ndarray):
        img_height, img_width, _ = img.shape
        img, ratio, (dw, dh) = letterbox(img, (736, 736))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        t = time.time()
        output = self.session.run(None, {self.model_input[0].name: image_data})
        print("Inference time: ", (time.time() - t)*1000, "ms")
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= 0.7:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= dw  # 移除填充
                y -= dh
                x /= ratio[0]  # 缩放回原图
                y /= ratio[1]
                w /= ratio[0]
                h /= ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(self.classes[class_id.astype(int)])
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.7, 0.7)
        new_boxes = [boxes[i] for i in indices]
        new_class_ids = [class_ids[i] for i in indices]
        if not new_boxes:
            return [], []
        # 将list1和list2合并，并按照list1子列表的第二项排序
        combined = sorted(zip(new_boxes, new_class_ids), key=lambda x: x[0][1])
        # 解压排序后的结果
        new_boxes, new_class_ids = zip(*combined)
        return list(new_boxes), list(new_class_ids)


class AIResolver:
    def __init__(self, api_key):
        self.session = httpx.Client()
        self.session.headers = {"Authorization": f"Bearer {api_key}"}

    def resolve_choice(self, img1: np.ndarray, img2: np.ndarray) -> list[str] | None:
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        data = {
            "model": "doubao-1-5-vision-pro-32k-250115",
            "messages": [
                {
                    "role": "system",
                    "content": "能力与角色:你是一位答题助手。\n背景信息:你会得到一张带有选择题的图片和一张带有答案的图片\n指令:你需要阅读分别阅读两张图片的内容，其中答案为红字部分，回答包含答案的选项\n输出风格:你无需给出推理过程以及任何解释。你只需要回答正确选项对应的ABCD，不得回答任何多余的文字，不得添加任何的标点符号。\n输出范围:我希望你仅仅回答 ABCD 中的一个或多个字母。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpg;base64,"+base64.b64encode(cv2.imencode('.jpg', img1)[1].tobytes()).decode(),
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpg;base64," + base64.b64encode(cv2.imencode('.jpg', img2)[1].tobytes()).decode(),
                        }
                    ],
                }
            ],
            "temperature": 0.2
        }
        response = self.session.post(url, json=data)
        try:
            if response.status_code == 200:
                result = response.json()
                print(result)
                answer = list(result["choices"][0]["message"]["content"])
                for i in answer.copy():
                    if i not in ['A', 'B', 'C', 'D']:
                        answer.remove(i)
                if len(answer) == 0:
                    raise ValueError("Invalid answer")
            else:
                answer = None
        except:
            answer = None
        return answer

    def resolve_blank(self, img: np.ndarray) -> str | None:
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        data = {
            "model": self.endpoint,
            "messages": [
                {
                    "role": "system",
                    "content": "能力与角色:你是一位答题助手\n背景信息:你会得到一张带有问题的图片\n指令:你需要阅读该图片中的问题，认真理解题目，确认填空的数量，思考后作出回答\n输出风格:你无需给出推理过程，也无需给出任何解释。你只需要回答空缺处应当填的内容，填充字数应当与空缺数量相同"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpg;base64,"+base64.b64encode(cv2.imencode('.jpg', img)[1].tobytes()).decode(),
                        }
                    ],
                }
            ],
            "temperature": 0.2
        }
        response = self.session.post(url, json=data)
        try:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
            else:
                answer = None
        except:
            answer = None
        return answer


class MaaWorker:
    def __init__(self, queue: SimpleQueue, api_key):
        user_path = "./"
        Toolkit.init_option(user_path)

        self.queue = queue
        self.resource = Resource()
        self.resource.set_cpu()
        self.resource.post_bundle("./resource").wait()
        self.tasker = Tasker()
        self.connected = False
        self.ai_resolver = AIResolver(api_key=api_key)
        self.model = ONNXModel()

        self.send_log("MAA初始化成功")

    def send_log(self,msg):
        self.queue.put(f"{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {msg}")

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
            self.send_log("设备连接失败，请检查终端日志")
            return self.connected
        if self.tasker.bind(self.resource, controller):
            self.connected = True
            # size = subprocess.run([device.adb_path, "shell", "wm", "size"], text=True, capture_output=True).stdout
            # size = size.strip().split(": ")[1]
            # dpi = subprocess.run([device.adb_path, "shell", "wm", "density"], text=True, capture_output=True).stdout
            # dpi = dpi.strip().split(": ")[1]
            # print(size,dpi)
            self.send_log("设备连接成功")
        else:
            self.send_log("设备连接失败，请检查终端日志")
        return self.connected

    def task(self, tasks):
        self.send_log("任务开始")
        try:
            for task in tasks:
                if task == "选读文章":
                    self.read_article()
                elif task == "视听学习":
                    self.watch_video()
                elif task == "每日答题":
                    self.daily_answer()
                elif task == "趣味答题":
                    self.funny_answer()
        except:
            self.send_log("任务出现异常，请检查终端日志")
            self.send_log("请将日志反馈至 https://github.com/ravizhan/MaaXuexi/issues")
        self.send_log("所有任务完成")
        time.sleep(0.5)

    def read_article(self):
        self.send_log("开始任务：选读文章")
        finished_article = []
        reading_time = 0
        while reading_time < 400:
            # 识别文章，获取点击文章的坐标范围
            image = self.tasker.controller.post_screencap().wait().get()
            boxes, box_class = self.model.detect(image)
            cv2.imwrite(f"./img_origin/{int(time.time())}.jpg", image)
            # 没有文章就滑动屏幕
            if len(boxes) == 0 or ("article" not in box_class and "article_image" not in box_class):
                self.send_log(f"未识别到文章，正在滑动屏幕")
                self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                             randint(300, 400),
                                             randint(1000, 1500)).wait()
                continue
            boxes, box_class = zip(*[(box, cls) for box, cls in zip(boxes, box_class) if cls in ["article", "article_image"]])
            self.send_log(f"识别到{len(boxes)}篇文章")
            article_list = []
            for box in boxes:
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                article_list.append(img)
            cv2.imwrite("result.jpg", image)
            for i in range(len(box_class)):
                if all(match_sift_flann(article_list[i], img2)[1] <= 0.7 for img2 in finished_article):
                    cv2.imwrite(f"read_{len(finished_article)}.jpg", article_list[i])
                    self.send_log(f"read_{len(finished_article)}")
                    self.tasker.controller.post_click(boxes[i][0] + 150, boxes[i][1] + 10)
                    time.sleep(3)
                    for _ in range(5):
                        self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),randint(300, 400), randint(1000, 1500)).wait()
                        t = randint(8, 10)
                        time.sleep(t)
                        reading_time += t
                    time.sleep(1)
                    self.tasker.post_task("返回").wait()
                    time.sleep(randint(3, 5))
                    finished_article.append(article_list[i])
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
            # 识别视频，获取点击视频的坐标范围
            image = self.tasker.controller.post_screencap().wait().get()
            boxes, box_class = self.model.detect(image)
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
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                video_list.append(img)
            cv2.imwrite("result.jpg", image)
            for i in range(len(box_class)):
                if all(match_sift_flann(video_list[i], img2)[1] <= 0.7 for img2 in finished_video):
                    cv2.imwrite(f"video_{len(finished_video)}.jpg", video_list[i])
                    self.send_log(f"video_{len(finished_video)}")
                    self.tasker.controller.post_click(boxes[i][0] + 150, boxes[i][1] + 10)
                    time.sleep(3)
                    t = randint(50, 70)
                    time.sleep(t)
                    waiting_time += t
                    self.tasker.post_task("返回2").wait()
                    time.sleep(randint(3, 5))
                    finished_video.append(video_list[i])
            self.tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),randint(300, 400), randint(1000, 1500)).wait()
        self.send_log("视听学习任务完成")

    def daily_answer(self):
        self.send_log("开始任务：每日答题")
        self.tasker.post_task("积分").wait()
        # 等待界面加载完毕
        time.sleep(10)
        load_result: TaskDetail = self.tasker.post_task("加载失败").wait().get()
        while not load_result.nodes:
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
        self.tasker.post_task("每日答题").wait()
        self.send_log("开始答题")
        # 等待界面加载完毕
        time.sleep(5)
        # 开始答题
        for i in range(5):
            # 判断是不是填空题
            recog_result: TaskDetail = self.tasker.post_task("填空题").wait().get()  # 单选题和填空题相似度竟然有0.75，离谱
            if not recog_result.nodes:
                self.send_log(f"第{i}题 填空题")
                recog_result: TaskDetail = self.tasker.post_task("填空题视频").wait().get()
                # 判断有没有视频，有的话调用AI解答
                if not recog_result.nodes:
                    self.send_log("发现视频，正在请求AI解答")
                    # 截图
                    image = self.tasker.controller.post_screencap().wait().get()
                    # AI解答
                    answer = self.ai_resolver.resolve_blank(image)
                    if answer is None:
                        plyer.notification.notify(
                            title="MaaXuexi",
                            message="AI解答失败，请求接管",
                            app_name="MAA",
                            timeout=0
                        )
                        self.send_log("AI解答失败, 请求接管")
                        #TODO 网页弹窗，pipe传递
                        input("完成该题后, 按任意键继续")
                        continue
                else:
                    self.send_log("查看提示")
                    self.tasker.post_task("查看提示").wait()
                    time.sleep(1)
                    find_result: TaskDetail = self.tasker.post_task("find_red").wait().get()
                    red_border = find_result.nodes[0].recognition.best_result.box
                    # self.send_log(red_border)
                    rec_result: TaskDetail = self.tasker.post_task("rec_answer",{"rec_answer": {"roi": red_border}}).wait().get()
                    answer = rec_result.nodes[0].recognition.best_result.text
                    self.tasker.post_task("关闭提示").wait()
                time.sleep(1)
                self.send_log(f"正在输入 {answer}")
                self.tasker.post_task("文本框点击").wait()
                time.sleep(0.5)
                self.tasker.controller.post_input_text(answer).wait()
                self.send_log("输入完成")
            else:
                self.send_log(f"第{i}题 选择题")
                # 问题截图
                img1 = self.tasker.controller.post_screencap().wait().get()
                # 答案截图
                self.tasker.post_task("查看提示").wait()
                time.sleep(1)
                img2 = self.tasker.controller.post_screencap().wait().get()
                self.tasker.post_task("关闭提示").wait()
                img2 = img2[500:1280, 0:720]
                time.sleep(1)
                # AI解答
                answer = self.ai_resolver.resolve_choice(img1, img2)
                if answer is None:
                    plyer.notification.notify(
                        title="MaaXuexi",
                        message="AI解答失败，请求接管",
                        app_name="MAA",
                        timeout=0
                    )
                    self.send_log("AI解答失败, 请求接管")
                    # TODO 网页弹窗，pipe传递
                    input("完成该题后, 按任意键继续")
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
                app_name="MAA",
                timeout=0
            )
            self.send_log("发现验证码，请求接管")
            # TODO 网页弹窗，pipe传递
            input("按任意键继续")


    def funny_answer(self):
        self.send_log("开始任务：趣味答题")
        pass
