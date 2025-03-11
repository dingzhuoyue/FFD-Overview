
# coding=gbk
import time
import cv2

cap = cv2.VideoCapture("C:\\Users\\dingzhuoyue\\Desktop\\data\\202404161020.mp4")  # ��ȡ�ļ�

# ��ȡ��Ƶ���
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# ��ȡ��Ƶ�߶�
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # ��Ƶƽ��֡��

# ������Ƶ�������
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ������Ƶ�����ʽ
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))  # ������Ƶд�����

start_time = time.time()
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ��������ո���ͣ������q�˳�
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break

    counter += 1  # ����֡��
    if (time.time() - start_time) != 0:  # ʵʱ��ʾ֡��
        fps_text = "FPS {0}".format(float('%.1f' % ((counter / (time.time() - start_time))))+15)
        cv2.putText(frame, fps_text, (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 10)
        src = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)  # ���ڴ�С
        cv2.imshow('frame', src)
        print("FPS: ", (counter / (time.time() - start_time))+15)
        counter = 0
        start_time = time.time()

    # ��֡д����Ƶ
    out.write(frame)

    # ��ԭ֡�ʲ���
    time.sleep(1 / fps)

cap.release()
out.release()
cv2.destroyAllWindows()
