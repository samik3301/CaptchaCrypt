import torch

class CaptchaModel(torch.nn.Module):
    def __init__(self, vocabulary_size):
        super(CaptchaModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            1,
            32,
            kernel_size = (3, 3),
            padding = (1,1)
        )
        self.batchnorm1 = torch.nn.BatchNorm2d(32)
        # self.drop1 = torch.nn.Dropout(0.1)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size = (2, 2))

        self.conv2 = torch.nn.Conv2d(
            32,
            64,
            kernel_size = (3, 3),
            padding = (1, 1)
        )
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        # self.drop2 = torch.nn.Dropout(0.1)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size = (2, 2))

        ''' Extra stuff that made loss worse
        self.conv3 = torch.nn.Conv2d(
            64,
            128,
            kernel_size = (3, 3),
            padding = (1,1)
        )
        self.batchnorm3 = torch.nn.BatchNorm2d(128)
        self.drop3 = torch.nn.Dropout(0.1)

        self.conv4 = torch.nn.Conv2d(
            128,
            256,
            kernel_size = (3, 3),
            padding = (1,1)
        )
        self.batchnorm4 = torch.nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout(0.1)
        '''


        self.dense1 = torch.nn.Linear(64 * 12, 100)
        self.dropout = torch.nn.Dropout(0.3)
        self.gru = torch.nn.GRU(100, 50, num_layers = 2, bidirectional = True, dropout = 0.25)

        self.output = torch.nn.Linear(100, vocabulary_size + 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batchnorm1(x)
        # x = self.drop1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.batchnorm2(x)
        # x = self.drop2(x)
        x = self.max_pool2(x)
        
        '''
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.batchnorm3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.batchnorm4(x)
        x = self.drop4(x)
        '''
        
        # bs x 64 x 65 x 12
        # print(x.size())
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        # print(x.size())
        # bs x 65 x 780
        x = self.dense1(x)
        x = self.dropout(x)
        x, _ = self.gru(x)

        x = self.output(x)
        return x

class CaptchaLoss(torch.nn.Module):
    def __init__(self):
        super(CaptchaLoss, self).__init__()
    
    def forward(self, preds :torch.Tensor, targets :torch.Tensor):
        batch_size = preds.size(0)
        x = preds.permute(1, 0, 2) # Output of (bs x 65 x vocabsize) from the model is arranged as (65 x bs x vocabsize) for alignment for ctc loss calculation
        x = torch.nn.functional.log_softmax(x, 2) # logsoftmax across the vocabsize dim
        pred_lengths = torch.full(
            size=(batch_size, ),
            fill_value = x.size(0), # 65
            dtype= torch.int32
        )
        target_lengths = torch.full(
            size=(batch_size, ),
            fill_value = targets.size(1), # 6
            dtype= torch.int32
        )
        loss = torch.nn.CTCLoss(blank=0)(
            x,
            targets,
            pred_lengths,
            target_lengths
        )
        return loss