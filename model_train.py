import copy
import time
import pandas as pd
from torchvision.datasets import FashionMNIST
import torch
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet # 假设 LeNet 类在 model.py 中定义
import torch.nn as nn
from tqdm import tqdm


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=8,
                                       shuffle=True,
                                       num_workers=2)  # 在Windows上如果遇到问题，可以尝试设置为0

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=8,
                                     shuffle=True,
                                     num_workers=2)  # 在Windows上如果遇到问题，可以尝试设置为0

    return train_dataloader, val_dataloader


# def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
#     # 改进：自动检测并使用GPU
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     model = model.to(device)  # 将模型移动到指定设备
#     best_model_wts = copy.deepcopy(model.state_dict())
#
#     best_acc = 0.0
#     train_loss_all = []
#     val_loss_all = []
#     train_acc_all = []
#     val_acc_all = []
#     since = time.time()
#
#     for epoch in range(num_epochs):
#         print("Epoch {}/{}".format(epoch, num_epochs - 1))
#         print("-" * 10)
#
#         train_loss = 0.0
#         train_corrects = 0.0
#         val_loss = 0.0
#         val_corrects = 0.0
#         train_num = 0
#         val_num = 0
#
#         # 训练阶段
#         model.train()  # 设置为训练模式
#         for step, (b_x, b_y) in enumerate(train_dataloader):
#             b_x = b_x.to(device)  # 将数据移动到指定设备
#             b_y = b_y.to(device)  # 将标签移动到指定设备
#
#             output = model(b_x)
#             pre_lab = torch.argmax(output, dim=1)
#             loss = criterion(output, b_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * b_x.size(0)
#             train_corrects += torch.sum(pre_lab == b_y).item()  # .item() 获取标量值
#             train_num += b_x.size(0)
#
#         # 验证阶段
#         model.eval()  # 设置为评估模式
#         with torch.no_grad():  # 在验证阶段禁用梯度计算，节省内存和计算
#             for step, (b_x, b_y) in enumerate(val_dataloader):
#                 b_x = b_x.to(device)  # 将数据移动到指定设备
#                 b_y = b_y.to(device)  # 将标签移动到指定设备
#
#                 output = model(b_x)
#                 pre_lab = torch.argmax(output, dim=1)
#                 loss = criterion(output, b_y)
#
#                 val_loss += loss.item() * b_x.size(0)
#                 val_corrects += torch.sum(pre_lab == b_y).item()  # .item() 获取标量值
#                 val_num += b_x.size(0)
#
#         train_loss_all.append(train_loss / train_num)
#         train_acc_all.append(train_corrects / train_num)  # 修正：直接除以数量
#         val_loss_all.append(val_loss / val_num)
#         val_acc_all.append(val_corrects / val_num)  # 修正：直接除以数量
#
#         print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
#         print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))
#
#         if val_acc_all[-1] > best_acc:
#             best_acc = val_acc_all[-1]
#             best_model_wts = copy.deepcopy(model.state_dict())
#
#         time_use = time.time() - since
#         print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))
#
#     model.load_state_dict(best_model_wts)
#     # 改进：保存模型的state_dict，并使用相对路径
#     torch.save(best_model_wts, './best_model.pth')
#     print("Best model saved to ./best_model.pth")
#
#     train_process = pd.DataFrame(data={"epoch": range(num_epochs),
#                                        "train_loss_all": train_loss_all,
#                                        "val_loss_all": val_loss_all,
#                                        "train_acc_all": train_acc_all,
#                                        "val_acc_all": val_acc_all, })
#
#     return train_process

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        train_num = 0
        val_num = 0

        # 训练阶段
        model.train()
        # 使用 tqdm 包装 train_dataloader
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        for step, (b_x, b_y) in enumerate(train_bar):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y).item()
            train_num += b_x.size(0)

            # 可以在进度条上显示实时信息（可选）
            train_bar.set_postfix(loss=loss.item(), acc=train_corrects / train_num)

        # 验证阶段
        model.eval()
        # 使用 tqdm 包装 val_dataloader
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_bar):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y).item()
                val_num += b_x.size(0)

                # 可以在进度条上显示实时信息（可选）
                val_bar.set_postfix(loss=loss.item(), acc=val_corrects / val_num)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss_all[-1]:.4f}, Train Acc: {train_acc_all[-1]:.4f}")
        print(f"  Val Loss: {val_loss_all[-1]:.4f}, Val Acc: {val_acc_all[-1]:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f"Total time elapsed: {time_use // 60:.0f}m {time_use % 60:.0f}s")
        print("-" * 20)

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, './best_model.pth')
    print("\nBest model saved to ./best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all, })

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


# 修正：将 "__name__" 改为 "__main__"
if __name__ == "__main__":
    # 改进：将实例变量名改为 model
    model = AlexNet()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(model, train_data, val_data, num_epochs=20)
    matplot_acc_loss(train_process)