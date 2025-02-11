import time
from random import randint

import cv2
from maa.define import TaskDetail
from maa.resource import Resource
from maa.controller import AdbController
from maa.tasker import Tasker
from maa.toolkit import Toolkit
from utils import ONNXModel, match_sift_flann, AIResolver
import plyer

def main():
    user_path = "./"
    Toolkit.init_option(user_path)

    resource = Resource()
    resource.set_cpu()
    res_job = resource.post_bundle("./resource")
    res_job.wait()

    adb_devices = Toolkit.find_adb_devices()
    if not adb_devices:
        print("No ADB device found.")
        exit()

    # for demo, we just use the first device
    device = adb_devices[0]
    controller = AdbController(
        adb_path=device.adb_path,
        address=device.address,
        screencap_methods=device.screencap_methods,
        input_methods=device.input_methods,
        config=device.config,
    )
    controller.post_connection().wait()

    tasker = Tasker()
    tasker.bind(resource, controller)

    if not tasker.inited:
        print("Failed to init MAA.")
        exit()

    ai_resolver = AIResolver(api_key="", endpoint="")
    model = ONNXModel()

    finished_article = []
    reading_time = 0
    finished_video = []
    waiting_time = 0
    while reading_time < 400:
        # 识别文章，获取点击文章的坐标范围
        image = tasker.controller.post_screencap().wait().get()
        boxes, box_class = model.detect(image)
        boxes, box_class = zip(*[(box, cls) for box, cls in zip(boxes, box_class) if cls in ["article", "article_image"]])
        # cv2.imwrite(f"./img_origin/{int(time.time())}.jpg", image)
        # 没有文章就滑动屏幕
        if len(boxes) == 0:
            tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600), randint(300, 400),randint(1000, 1500)).wait()
            continue
        article_list = []
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            article_list.append(img)
        cv2.imwrite("result.jpg", image)
        a = 0
        for i in range(len(box_class)):
            if all(match_sift_flann(article_list[i], img2)[1] <= 0.7 for img2 in finished_article):
                cv2.imwrite(f"read_{len(finished_article)}.jpg", article_list[i])
                print(f"read_{len(finished_article)}")
                tasker.controller.post_click(boxes[i][0]+150, boxes[i][1]+10)
                time.sleep(3)
                for _ in range(5):
                    tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),randint(300, 400), randint(1000, 1500)).wait()
                    t = randint(8, 10)
                    time.sleep(t)
                    reading_time += t
                time.sleep(1)
                tasker.post_task("返回").wait()
                time.sleep(randint(3, 5))
                finished_article.append(article_list[i])
        tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),randint(300, 400),randint(1000, 1500)).wait()
    tasker.post_task("电视台").wait()
    time.sleep(randint(3, 5))
    while waiting_time < 400:
        # 识别视频，获取点击视频的坐标范围
        image = tasker.controller.post_screencap().wait().get()
        boxes, box_class = model.detect(image)
        boxes, box_class = zip(*[(box, cls) for box, cls in zip(boxes, box_class) if cls in ["video"]])
        # 没有视频就滑动屏幕
        if len(boxes) == 0:
            tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600), randint(300, 400),randint(1000, 1500)).wait()
            continue
        video_list = []
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            video_list.append(img)
        cv2.imwrite("result.jpg", image)
        a = 0
        for i in range(len(box_class)):
            if all(match_sift_flann(video_list[i], img2)[1] <= 0.7 for img2 in finished_video):
                cv2.imwrite(f"video_{len(finished_video)}.jpg", video_list[i])
                print(f"video_{len(finished_video)}")
                tasker.controller.post_click(boxes[i][0] + 150, boxes[i][1] + 10)
                time.sleep(3)
                t = randint(50, 70)
                time.sleep(t)
                waiting_time += t
                tasker.post_task("返回2").wait()
                time.sleep(randint(3, 5))
                finished_video.append(video_list[i])
        tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),randint(300, 400), randint(1000, 1500)).wait()
    tasker.post_task("积分").wait()
    # 等待界面加载完毕
    time.sleep(10)
    laod_result: TaskDetail = tasker.post_task("加载失败").wait().get()
    while not laod_result.nodes:
        tasker.post_task("返回").wait()
        tasker.post_task("积分").wait()
        time.sleep(10)
        laod_result: TaskDetail = tasker.post_task("加载失败").wait().get()
    print("加载成功")
    # 滑动到每日答题按钮
    tasker.controller.post_swipe(randint(200, 300), randint(1000, 1100), randint(500, 600), randint(100, 200),randint(1000, 1500)).wait()
    time.sleep(randint(1, 2))
    # 点击每日答题按钮
    tasker.post_task("每日答题").wait()
    print("开始答题")
    # 等待界面加载完毕
    time.sleep(5)
    # 开始答题
    for i in range(5):
        # 判断是不是填空题
        recog_result: TaskDetail = tasker.post_task("填空题").wait().get() # 单选题和填空题相似度竟然有0.75，离谱
        if not recog_result.nodes:
            print("填空题")
            recog_result: TaskDetail = tasker.post_task("填空题视频").wait().get()
            # 判断有没有视频，有的话调用AI解答
            if not recog_result.nodes:
                print("发现视频，正在请求AI解答")
                # 截图
                image = tasker.controller.post_screencap().wait().get()
                # AI解答
                answer = ai_resolver.resolve_blank(image)
                if answer is None:
                    plyer.notification.notify(
                        title="MaaXuexi",
                        message="AI解答失败，请求接管",
                        app_name="MAA",
                        timeout=0
                    )
                    print("AI解答失败, 请求接管")
                    input("完成该题后, 按任意键继续")
                    continue
            else:
                print("查看提示")
                tasker.post_task("查看提示").wait()
                time.sleep(1)
                find_result: TaskDetail = tasker.post_task("find_red").wait().get()
                red_border = find_result.nodes[0].recognition.best_result.box
                # print(red_border)
                rec_result: TaskDetail = tasker.post_task("rec_answer", {"rec_answer":{"roi": red_border}}).wait().get()
                answer = rec_result.nodes[0].recognition.best_result.text
                tasker.post_task("关闭提示").wait()
            time.sleep(1)
            print(answer)
            print("正在输入")
            tasker.post_task("文本框点击").wait()
            time.sleep(0.5)
            tasker.controller.post_input_text(answer).wait()
            print("输入完成")
        else:
            print("选择题")
            # 问题截图
            img1 = tasker.controller.post_screencap().wait().get()
            # 答案截图
            tasker.post_task("查看提示").wait()
            time.sleep(1)
            img2 = tasker.controller.post_screencap().wait().get()
            tasker.post_task("关闭提示").wait()
            img2 = img2[500:1280, 0:720]
            time.sleep(1)
            # AI解答
            answer = ai_resolver.resolve_choice(img1,img2)
            if answer is None:
                plyer.notification.notify(
                    title="MaaXuexi",
                    message="AI解答失败，请求接管",
                    app_name="MAA",
                    timeout=0
                )
                print("AI解答失败, 请求接管")
                input("完成该题后, 按任意键继续")
                continue
            print(answer)
            for i in answer:
                if i == "A":
                    tasker.post_task("选A").wait()
                elif i == "B":
                    tasker.post_task("选B").wait()
                elif i == "C":
                    tasker.post_task("选C").wait()
                elif i == "D":
                    tasker.post_task("选D").wait()
                time.sleep(0.5)
            time.sleep(0.5)
        # 下一题
        tasker.post_task("下一题").wait()
        time.sleep(randint(2, 3))
    # 结束答题，大概率会弹验证码
    time.sleep(2)
    recog_result: TaskDetail = tasker.post_task("访问异常").wait().get()
    if not recog_result.nodes:
        plyer.notification.notify(
            title="MaaXuexi",
            message="发现验证码，请求接管",
            app_name="MAA",
            timeout=0
        )
        print("发现验证码，请求接管")
        input("按任意键继续")


if __name__ == "__main__":
    main()
