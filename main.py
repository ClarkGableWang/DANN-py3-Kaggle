import os
import random
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import npy_to_tensor, train_test_split, load_dataset
from test import test
# Different model
from model import CNNModel
from ConvnextModel import convnext_tiny


torch.manual_seed(99)
data_root = [r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude1hptrain100.npy',
             r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude2hptrain100.npy',
             r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude3hptrain100.npy']

gadf_data_root = [r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude1hpGADFtrain100.npy',
                  r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude2hpGADFtrain100.npy',
                  r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude3hpGADFtrain100.npy']

gadf112_data_root = [r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude1hp112GADFtrain100.npy',
                     r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude2hp112GADFtrain100.npy',
                     r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude3hp112GADFtrain100.npy']

label_root = [r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude1hplabel100.npy',
              r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude2hplabel100.npy',
              r'D:\E\Anaconda\JupyterNotebookLearning\ensorFlow\cwru_train_data\hpdata\cwrude3hplabel100.npy']
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
data_path_root = os.path.join(father_path, "dataset")
model_root = os.path.join(father_path, "models")
# source_dataset_root = data_root[2]
# target_dataset_root = data_root[1]
source_dataset_root = gadf112_data_root[2]
target_dataset_root = gadf112_data_root[1]
label_source = label_root[2]
label_target = label_root[1]


cuda = True
# cudnn.benchmark = True
lr = 1e-3
batch_size = 16
# image_size = 32
n_epoch = 50

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
data_source = np.load(source_dataset_root)
data_target = np.load(target_dataset_root)
label_source = np.load(label_source)
label_target = np.load(label_target)

data_source_train, label_source_train, data_source_test, label_source_test = train_test_split(data_source, label_source)
data_target_train, label_target_train, data_target_test, label_target_test = train_test_split(data_target, label_target)
dataset_source_train = load_dataset(data_source_train, label_source_train)
dataset_target_train = load_dataset(data_target_train, label_target_train)
dataset_source_test = npy_to_tensor(data_source_test, label_source_test)
dataset_target_test = npy_to_tensor(data_target_test, label_target_test)

# Create dataloader
dataloader_source_train = torch.utils.data.DataLoader(
    dataset=dataset_source_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
dataloader_target_train = torch.utils.data.DataLoader(
    dataset=dataset_target_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
dataloader_source_test = torch.utils.data.DataLoader(
    dataset=dataset_source_test,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
dataloader_target_test = torch.utils.data.DataLoader(
    dataset=dataset_target_test,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)

# load model
my_net = convnext_tiny(num_classes=10)

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr)

# Loss function
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training

best_acc_s = 0.0
best_acc_t = 0.0
# total_error = 0.0
# lowest_error = 1e4

for epoch in range(n_epoch):
    len_dataloader = min(len(dataloader_source_train), len(dataloader_target_train))
    data_source_iter = iter(dataloader_source_train)
    data_target_iter = iter(dataloader_target_train)
    print('*'*20, 'Training', '*'*20)
    my_net.train()
    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # alpha from 0 to 1 slowly
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = next(data_source_iter)
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()

        class_output, domain_output = my_net(s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = next(data_target_iter)
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = my_net(t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        # total_error = err.item()
        err.backward()
        optimizer.step()

        # total_error = err_s_domain + err_t_domain
        # if total_error < lowest_error:
        #     torch.save(my_net, '{0}/current.pth'.format(model_root))
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                         % (epoch, i + 1, len_dataloader, err_s_label.detach().cpu().numpy(),
                            err_s_domain.detach().cpu().numpy(), err_t_domain.detach().cpu().item()))
        sys.stdout.flush()
    # save lowest error model
    # if total_error < lowest_error:
    #     lowest_error = total_error
    #     torch.save(my_net, '{0}/current.pth'.format(model_root))

    # source test acc
    print('\n') 
    print('*'*20, 'Testing', '*'*20)
    acc_s = test(dataloader_source_test, net=my_net)
    print('Accuracy of the source dataset: %.4f'%(acc_s))

    acc_t = test(dataloader_target_test, net=my_net)
    print('Accuracy of the source dataset: %.4f' % (acc_t))

    if acc_s > best_acc_s:
        best_acc_s = acc_s
    if acc_t > best_acc_t:
        best_acc_t = acc_t

print('=' * 20, 'Summary', '=' * 20)
print('Accuracy of the %s dataset: %.4f' % ('source', best_acc_s))
print('Accuracy of the %s dataset: %.4f' % ('target', best_acc_t))

#     len_dataloader_source = len(dataloader_source_test)
#     dataset_iter = iter(dataloader_source_test)
#     n_total_source = 0
#     n_correct = 0
#     my_net.eval()
#     for i in range(len_dataloader_source):
#         data_test = next(dataset_iter)
#         t_img, t_lab = data_test
#         t_img, t_lab = t_img.cuda(), t_lab.cuda()
#
#         batch_size = len(t_lab)
#         class_output, _ = my_net(input_data=t_img, alpha=alpha)
#         pred = class_output.detach().max(1, keepdim=True)[1]
#         n_correct += pred.eq(t_lab.detach().view_as(pred)).cpu().sum()
#         n_total_source += batch_size
#     acc_s = n_correct.data.numpy() * 1.0 / n_total_source
#     sys.stdout.write('\r Accuracy of the source dataset: %f ' %(acc_s))
#     sys.stdout.flush()
#
#     # target test acc
#     print('\n')
#     len_dataloader = len(dataloader_target_test)
#     dataset_iter = iter(dataloader_target_test)
#     n_total_target = 0
#     n_correct = 0
#     for i in range(len_dataloader):
#         data_test = next(dataset_iter)
#         t_img, t_lab = data_test
#         t_img, t_lab = t_img.cuda(), t_lab.cuda()
#
#         batch_size = len(t_lab)
#         class_output, _ = my_net(input_data=t_img, alpha=alpha)
#         pred = class_output.detach().max(1, keepdim=True)[1]
#         n_correct += pred.eq(t_lab.detach().view_as(pred)).cpu().sum()
#         n_total_target += batch_size
#     acc_t = n_correct.data.numpy() * 1.0 / n_total_target
#     print('Accuracy of the %s dataset: %f\n' % ('target', acc_t))
#
#     if acc_s > best_acc_s:
#         best_acc_s = acc_s
#     if acc_t > best_acc_t:
#         best_acc_t = acc_t
#
#     # torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))
#
# print('='*20, 'Summary', '='*20)
# print('Accuracy of the %s dataset: %f' % ('source', best_acc_s))
# print('Accuracy of the %s dataset: %f' % ('target', best_acc_t))
# print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')