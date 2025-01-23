import base64
import time
import numpy as np
import onnxruntime
import cv2
import httpx


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
    print("Similarity score: ", similarity)
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
        self.session = onnxruntime.InferenceSession("resource/model/yolo/yolo.onnx")
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
    def __init__(self, api_key, endpoint):
        self.endpoint = endpoint
        self.session = httpx.Client()
        self.session.headers = {"Authorization": f"Bearer {api_key}"}

    def resolve_choice(self, img: np.ndarray) -> list[str] | None:
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        data = {
            "model": self.endpoint,
            "messages": [
                {
                    "role": "system",
                    "content": "能力与角色:你是一位答题助手。\n背景信息:你会得到一张带有问题的图片\n指令:你需要阅读该图片中的问题，认真理解题目和选项的内容，思考后作出回答\n输出风格:你无需给出推理过程，也无需给出任何解释。你只需要回答正确的选项对应的字母，不得回答任何多余的文字。\n输出范围:我希望你仅仅回答 ABCD 中的一个或多个字符。"
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
                answer = list(result["choices"][0]["message"]["content"])
                for i in answer:
                    if i not in ['A', 'B', 'C', 'D']:
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
