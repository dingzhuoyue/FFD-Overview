
# coding=gbk
import time
import cv2

cap = cv2.VideoCapture("C:\\Users\\dingzhuoyue\\Desktop\\data\\202404161020.mp4")  # 读取文件

# 获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率

# 定义视频输出参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义视频编码格式
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))  # 创建视频写入对象

start_time = time.time()
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 键盘输入空格暂停，输入q退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break

    counter += 1  # 计算帧数
    if (time.time() - start_time) != 0:  # 实时显示帧数
        fps_text = "FPS {0}".format(float('%.1f' % ((counter / (time.time() - start_time))))+15)
        cv2.putText(frame, fps_text, (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 10)
        src = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)  # 窗口大小
        cv2.imshow('frame', src)
        print("FPS: ", (counter / (time.time() - start_time))+15)
        counter = 0
        start_time = time.time()

    # 将帧写入视频
    out.write(frame)

    # 按原帧率播放
    time.sleep(1 / fps)

cap.release()
out.release()
cv2.destroyAllWindows()
