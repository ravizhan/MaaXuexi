import time
from random import randint

import cv2
from maa.resource import Resource
from maa.controller import AdbController
from maa.tasker import Tasker
from maa.toolkit import Toolkit
from utils import ONNXModel, match_sift_flann

def main():
    user_path = "./"
    Toolkit.init_option(user_path)

    resource = Resource()
    resource.set_cpu()
    res_job = resource.post_path("./resource")
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

    model = ONNXModel()

    finished_article = []
    finished_video = []
    # 看八篇文章，确保刷满时长
    while len(finished_article) < 8:
        # 识别文章，获取点击文章的坐标范围
        image = tasker.controller.post_screencap().wait().get()
        article_boxes, box_class = model.detect(image)
        # 没有文章就滑动屏幕
        if len(article_boxes) == 0:
            tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600), randint(300, 400),
                                         randint(1000, 1500)).wait()
            continue
        video_image_list = []
        for box in article_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            video_image_list.append(img)
        cv2.imwrite("result.jpg", image)
        a = 0
        for i in range(len(box_class)):
            if box_class[i] in ["article","article_image"]:
                excute = True
                for img2 in finished_article:
                    result_img, score = match_sift_flann(video_image_list[i], img2)
                    # cv2.imwrite(f"similar_{reading_count}_{a}.jpg", result_img)
                    a += 1
                    if score > 0.7:
                        excute = False
                        break
                if excute:
                    cv2.imwrite(f"read_{len(finished_article)}.jpg", video_image_list[i])
                    print(f"read_{len(finished_article)}")
                    tasker.controller.post_click(article_boxes[i][0]+150, article_boxes[i][1]+10)
                    time.sleep(3)
                    for _ in range(5):
                        tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                                     randint(300, 400), randint(1000, 1500)).wait()
                        time.sleep(randint(8, 10))
                    time.sleep(1)
                    tasker.post_pipeline("返回").wait()
                    time.sleep(randint(3, 5))
                    finished_article.append(video_image_list[i])
        tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                     randint(300, 400),randint(1000, 1500)).wait()
    tasker.post_pipeline("电视台").wait()
    time.sleep(randint(3, 5))
    # 看八个视频，确保刷满时长
    while len(finished_video) < 8:
        # 识别视频，获取点击视频的坐标范围
        image = tasker.controller.post_screencap().wait().get()
        article_boxes, box_class = model.detect(image)
        # 没有视频就滑动屏幕
        if len(article_boxes) == 0:
            tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600), randint(300, 400),
                                         randint(1000, 1500)).wait()
            continue
        video_image_list = []
        for box in article_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            video_image_list.append(img)
        cv2.imwrite("result.jpg", image)
        a = 0
        for i in range(len(box_class)):
            if box_class[i] in ["video"]:
                excute = True
                for img2 in finished_video:
                    result_img, score = match_sift_flann(video_image_list[i], img2)
                    # cv2.imwrite(f"similar_{reading_count}_{a}.jpg", result_img)
                    a += 1
                    if score > 0.7:
                        excute = False
                        break
                if excute:
                    cv2.imwrite(f"video_{len(finished_video)}.jpg", video_image_list[i])
                    print(f"video_{len(finished_video)}")
                    tasker.controller.post_click(article_boxes[i][0] + 150, article_boxes[i][1] + 10)
                    time.sleep(3)
                    time.sleep(randint(50, 70))
                    tasker.post_pipeline("返回2").wait()
                    time.sleep(randint(3, 5))
                    finished_video.append(video_image_list[i])
        tasker.controller.post_swipe(randint(200, 300), randint(900, 1000), randint(500, 600),
                                     randint(300, 400), randint(1000, 1500)).wait()



if __name__ == "__main__":
    main()
