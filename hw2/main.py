import preprocessor as pp
import network as CNN
import torch
import torchvision
from torch.utils import data
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np



def prepare_dataset(X,y):
    res = []
    for i in range(len(X)):
        df = X[i]
        label = y[i:i+11]
        df = torch.tensor(df)
        label = torch.tensor(label[5])
        # label = label.unsqueeze(0)
        df_f = torch.reshape(df, (1,df.shape[0], df.shape[1]))
        res.append([df_f, label])
    return res


def two_array_plot(arr1, arr2, x_label, y_label, arr1_label, arr2_label, num_of_epochs):
    """
    Plot 2 data sets

    Args:
    arr1 : list of 1st plot values
    arr2 : list of 2nd plot values
    x_label (str) : x axis label
    y_label (str) : y axis label
    arr1_label (str) : 1st plot's label
    arr2_label (str) : 2nd plot's label
    num_of_epochs (int) : number of epochs
    """
    x = np.array(arr1)
    y = np.array(arr2)


    epochs = [i for i in range(num_of_epochs)]
    plt.plot(epochs,x, label=arr1_label)
    plt.plot(epochs,y, label=arr2_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # plt.xlim(0,num_of_epochs-1)
    # if feature == "Accuracy":
    #     plt.ylim(0,1)
    #
    # if feature == "Loss":
    #     lim = max(max(x), max(y))
    #     plt.ylim(0, lim+1)

    plt.show()



def main(protein, split_size, epoches):
    # with open('proteins_data_set.txt') as db:
    #     lines = db.readlines()
    X = pp.generate_dataset(protein)
    y = pp.generate_labels(protein)
    set = prepare_dataset(X,y)
    training_set = set[:split_size]
    test_set = set[split_size:]
    batch_size = 1
    training_generator = torch.utils.data.DataLoader(training_set,  batch_size=1, shuffle=True, num_workers=0 )
    test_generator = torch.utils.data.DataLoader(test_set,  batch_size=1, shuffle=True, num_workers=0 )
    train_loss, train_acc, test_loss, test_acc = CNN.training_testing_data(training_generator,test_generator,epoches)
    two_array_plot(train_loss,test_loss,"Epochs","Loss", "Train", "Test",epoches)
    two_array_plot(train_acc, test_acc, "Epochs", "Accuracy", "Train", "Test", epoches)





protein = "kvfgrcelaaamkrhglaNyrGYSlgNwvcaakfesnfntqatnrntdgstdygilqinsrwwcndgrtpgsrnlcnipcsallssditasvncakkivsdGNgmnawvawrnrcKGTDVQawIRgcrl"

print(main(protein, 60, 80))

