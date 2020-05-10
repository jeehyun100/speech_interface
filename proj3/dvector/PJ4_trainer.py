
import torch
import torch.nn as nn
import torch.optim as optim #optimizer
from tqdm import tqdm #for print
import numpy as np
from torch.utils.data import DataLoader
from PJ4_dataset import * #교수님 코드
from PJ4_model import *
import os
from tensorboardX import SummaryWriter



# Define a train function.
def train(epoch, model, criterion, optimizer, dataloader, device):
    #summary(model, (indim * context_len))
    model.train()
    bar = tqdm(enumerate(dataloader))

    samples = 0
    total_loss, total_acc = 0, 0
    for batch_idx, (data, label) in bar:
        data = data.type(torch.FloatTensor).to(device)
        label = label.to(device)

        # Pass the input data through the defined network architecture.
        pred = model(data, extract=True)

        # Compute a loss function.
        loss = criterion(pred, label)
        total_loss += loss.item() * len(label)

        acc = torch.sum(torch.eq(torch.argmax(pred, -1), label)).item()
        samples += len(label)
        total_acc += acc

        # Perform backpropagation to update network parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.set_description('Epoch:{:3d} [{}/{} {:.2f}%] CE Loss: {:.3f} ACC: {:.2f}'.format(
            epoch, batch_idx, len(dataloader), 100. * (batch_idx / len(dataloader)),
                                               total_loss / samples, (total_acc / samples) * 100.))

    # Define a test function.


def test(model, criterion, dataloader, device):
    model.eval()
    bar = tqdm(dataloader)

    samples = 0
    total_loss, total_acc = 0, 0
    for batch_idx, (data, label) in enumerate(bar):
        data = data.type(torch.FloatTensor).to(device)
        label = label.to(device)

        # Pass the input data through the defined network architecture.
        pred = model(data, extract=True)

        # Compute a loss function.
        loss = criterion(pred, label)
        total_loss += loss.item() * len(label)

        samples += len(label)
        acc = torch.sum(torch.eq(torch.argmax(pred, -1), label)).item()
        total_acc += acc

    return total_loss / samples, (total_acc / samples) * 100.


# Define a collate (stacking) function for the data loader.
def collate_fn(samples):
    data, label = [], []
    min_length = min([len(d[0]) for d in samples]) - 1

    for d, l in samples:
        st = np.random.randint(len(d) - min_length)
        data.append(torch.tensor(d[st:st + min_length]).unsqueeze(0))
        label.append(torch.tensor(l))
    data = torch.cat(data, 0)
    label = torch.LongTensor(label)

    return data, label


# Obtain the dataset and dataloader of train and test data.
def get_dataloader(train_path, test_path, data_path, feature_type='mel', n_coeff=64):
    train_dataset = SpeakerDataset(train_path, data_path,
                                   feature_type=feature_type, n_coeff=n_coeff)
    test_dataset = SpeakerDataset(test_path, data_path,
                                  feature_type=feature_type, n_coeff=n_coeff)

    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=0,
                             pin_memory=True)

    return train_loader, test_loader


def main(model_path=None):
    # 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
    writer = SummaryWriter('./runs/dvector_experiment_1')

    ########################################### Settings ##############################################
    # Set the configuration for training.
    epochs = 200  # number of epochs
    lr = 0.01  # initial learning rate
    n_spk = 30  # number of speakers in the dataset
    log_path = 'dvector.log'  # log file
    feature_type = 'mfcc'  # input feature type
    n_coeff = 13  # feature dimension
    indim = n_coeff * 3  # input dimension (MFCC, delta, delta-delta)
    context_len = 10  # number of context window # 몇개의 프레임을합쳐서 입력을 줄것인가?
    outdim = 256  # d-vector output dimension

    model_dir = "./model_aug2/"
    os.makedirs(model_dir, exist_ok=True)

    # Check if we can use a GPU device.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # # Define the directory path of training and test datasets.
    # train_path = './wsj_train.txt'
    # test_path = './wsj_test.txt'
    # # Define the directory path of training and test datasets.
    train_path = './sic_train.txt'
    test_path = './sic_test.txt'
    data_path = '..'

    ############################### Dataset and dataloader #############################################
    train_loader, test_loader = get_dataloader(train_path, test_path, data_path,
                                               feature_type=feature_type, n_coeff=n_coeff)

    #################################      Get model       #############################################

    # Define a network model.
    model = Dvector(n_spk, indim * context_len, outdim).to(device)

    print(model)

    # Set the optimizer with Adam.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Set the training criterion.
    softmax_criterion = nn.CrossEntropyLoss(reduction='mean')

    #################################### Load pre-trained model ########################################
    start = 0
    # Load the pre-trained model.
    print('Directory of the pre-trained model: {}'.format(model_path))
    if model_path:
        check = torch.load(model_path)
        start = check['epoch']
        model.load_state_dict(check['model'])
        optimizer.load_state_dict(check['optimizer'])
        print('## Successfully load the model at {} epochs!'.format(start))

        ####################################      Train and Test     ######################################
    prev_loss = 10000
    prev_acc = 0
    ct_dec = 0

    for epoch in range(start, epochs + 1):

        # Train the network.
        train(epoch, model, softmax_criterion, optimizer, train_loader, device)

        # Test the network.
        opt_loss, opt_acc = test(model, softmax_criterion, test_loader, device)

        # Print out the results.
        print("Epoch: {} Test Loss: {:.3f} Test ACC: {:.2f}".format(epoch,opt_loss,opt_acc ))
        #if epoch % 5 == 4:    # 매 1000 미니배치마다...

            # ...학습 중 손실(running loss)을 기록하고
        writer.add_scalar('training loss',
                        opt_loss,
                          (epoch+1))
        # Save the optimal model.
        if opt_loss < prev_loss:
            prev_loss = opt_loss
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       model_dir+'model_opt_loss'+"{:0.3f}".format(opt_loss)+'.pth')
            ct_edec = 0
        else:
            ct_dec += 1

            # Decrease the learning rate by 2 when the test loss decreases 3 times in a row.
            if ct_dec == 30: # 10
                optim_state = optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] /= 2
                optimizer.load_state_dict(optim_state)
                print('lr is divided by 2.')
                ct_dec = 0
        #print("prev_acc {0}".format(prev_acc))
        if opt_acc > prev_acc:
            print("save better acc {0}".format(opt_acc))
            prev_acc = opt_acc
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       model_dir+'model_opt_acc'+"{:0.3f}".format(opt_acc)+'.pth')
            #ct_edec = 0
        if opt_acc > 93.0:
            print("save better acc {0}".format(opt_acc))
            prev_acc = opt_acc
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       model_dir+'model_opt_acc'+"{:0.3f}".format(opt_acc)+'.pth')


#main(model_path='./model/model_opt.pth')
main()

