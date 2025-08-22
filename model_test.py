import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet # 假设 LeNet 类在 model.py 中定义

def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                             download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1, # 通常测试时 batch_size 可以设置为 1
                                      shuffle=False, # 测试集通常不需要打乱
                                      num_workers=0) # Windows 上 num_workers 设为 0 更稳定
    return test_dataloader # 修正：返回数据加载器

def test_model_process(model, test_dataloader):
    # 自动检测并使用GPU，如果可用的话
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = model.to(device) # 将模型移动到指定设备
    model.eval() # 将模型设置为评估模式，只调用一次

    test_corrects = 0.0
    test_num = 0
    with torch.no_grad(): # 在此上下文管理器中，禁用梯度计算，节省内存和计算
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device) # 将数据移动到指定设备
            test_data_y = test_data_y.to(device) # 将标签移动到指定设备

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y).item() # 修正：使用 .item() 获取标量值
            test_num += test_data_x.size(0)

    test_acc = test_corrects / test_num # 计算准确率
    print("测试准确率: " + str(test_acc)) # 修正：将浮点数转换为字符串再拼接

if __name__=="__main__":
    model = LeNet()
    # 加载模型权重。如果模型是在GPU上训练的，但在CPU上测试，需要指定map_location
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

    # 调用修正后的 test_data_process 获取数据加载器
    test_dataloader = test_data_process()

    # 调用测试函数
    test_model_process(model, test_dataloader)

    device = 'cpu'
    model = model.to(device)

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值",result,"这是分割线","真实值",label)