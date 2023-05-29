import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
data_path_root = os.path.join(father_path, "dataset")

cuda = True

def test(dataloader):
    
    """ test """
    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    
    n_total = 0
    n_correct = 0

    for i in range(len_dataloader):
        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size


    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
