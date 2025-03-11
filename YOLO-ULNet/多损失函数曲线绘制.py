
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# net1 = pd.read_csv('C:\\Users\\dingzhuoyue\\Downloads\\run-solov2-tag-total_loss.csv', usecols=['Step', 'Value'])
# plt.plot(net1.Step, net1.Value, lw=1.5, color='red')
# net2 = pd.read_csv('C:\\Users\\dingzhuoyue\\Downloads\\run-BL-tag-total_loss.csv', usecols=['Step', 'Value'])
# plt.plot(net2.Step, net2.Value, lw=1.5,  color='orange')
# net3 = pd.read_csv('C:\\Users\\dingzhuoyue\\Downloads\\run-rnn-tag-total_loss.csv', usecols=['Step', 'Value'])
# plt.plot(net3 .Step, net3 .Value, lw=1.5,  color='green')
# net4 = pd.read_csv('C:\\Users\\dingzhuoyue\\Downloads\\run-BOX-tag-total_loss.csv', usecols=['Step', 'Value'])
# plt.plot(net4 .Step, net4 .Value, lw=1.5,  color='blue')
#
#
# plt.legend(loc=0)
# #plt.title('total-loss')
# plt.savefig("C:\\Users\\dingzhuoyue\\Downloads\\total-loss.jpg")
# plt.show()


net1 = pd.read_excel('F:\\计算机视觉学习资料\\计算机视觉\\CV学习资料\\毕设资料\\论文写作资料\\实验数据\\yolov5s EL.xlsx', usecols=['epoch', 'loss'])
plt.plot(net1.epoch, net1.loss, lw=1.5,  color='red')
net2 = pd.read_excel('F:\\计算机视觉学习资料\\计算机视觉\\CV学习资料\\毕设资料\\论文写作资料\\实验数据\\yolov5-Shuffle EL.xlsx', usecols=['epoch', 'loss'])
plt.plot(net2.epoch, net2.loss, lw=1.5,  color='orange')
net3 = pd.read_excel('F:\\计算机视觉学习资料\\计算机视觉\\CV学习资料\\毕设资料\\论文写作资料\\实验数据\\yolov5-SG EL.xlsx', usecols=['epoch', 'loss'])
plt.plot(net3.epoch, net3.loss, lw=1.5,  color='green')
net4 = pd.read_excel('F:\\计算机视觉学习资料\\计算机视觉\\CV学习资料\\毕设资料\\论文写作资料\\实验数据\\yolov5-SGF EL.xlsx', usecols=['epoch', 'loss'])
plt.plot(net4.epoch, net4.loss, lw=1.5, color='blue')
net5 = pd.read_excel('F:\\计算机视觉学习资料\\计算机视觉\\CV学习资料\\毕设资料\\论文写作资料\\实验数据\\yolo-lf.xlsx', usecols=['epoch', 'loss'])
plt.plot(net5.epoch, net5.loss, lw=1.5, color='brown')


plt.legend(loc=0)
# plt.title('mask-loss')
plt.savefig("C:\\Users\\dingzhuoyue\\Downloads\\step-loss.jpg")
plt.show()
