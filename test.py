import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")


def test(dataset_name, net):
    """

    :param dataset_name: data source test or data target test
    :return: accu
    """
    model_root = os.path.join(father_path, "models")

    cuda = True
    cudnn.benchmark = True
    alpha = 0

    """ test """
    # load model
    my_net = net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataset_name)
    data_target_iter = iter(dataset_name)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
