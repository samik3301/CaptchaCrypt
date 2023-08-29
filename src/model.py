import torch 
from torch import nn 
from torch.nn import functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel,self).__init__()
        self.conv_1 = nn.Conv2d(
            3,
            128,
            kernel_size=(3,3),
            padding = (1,1)
        )
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=(2,2)
        )
        self.conv_2 = nn.Conv2d(
            128,
            64,
            kernel_size=(3,3),
            padding = (1,1)
        )
        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=(2,2)
        )
        #adding another layer here 
        self.linear_1 = nn.Linear(1152,64) #input size becomes 1152 and number of features = 64
        self.drop_1 = nn.Dropout(0.2) #adding a dropout

        #can add either LSTM or GRU model
        self.gru = nn.GRU(64,32, bidirectional = True,
            num_layers = 2,
            dropout = 0.25
        )

        self.output = nn.Linear(64, num_chars+1)



    def forward(self,images,targets=None):
        #giving us some batch size, channels, height and width 
        bs, c, h, w = images.size()
        #print(bs,c,h,w)
        x = F.relu(self.conv_1(images))
        #print(x.size()) #first convolution , the size remainds the same
        x = self.max_pool_1(x)
        #print(x.size()) #reduces the size in hald
        x = F.relu(self.conv_2(x))
        #print(x.size()) #2nd convolution the size remains the same
        x = self.max_pool_2(x)
        #print(x.size()) #2nd max pooling , the size reduces in half
        # 1 64 18 75 - bs, num of filter, height, width
        # 1 75 64 18
        x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x=x.view(bs,x.size(1),-1) #-1 indicates multiplying other parameters and keeping the size same
        #print(x.size()) #outputs torch.Size([1, 75, 1152]) so batch size is 1, number of features are 75 and the rest is 1152 
        x = self.linear_1(x)
        x = self.drop_1(x)
        #print(x.size()) #outputs torch.Size([1, 75, 64])
        x,_ = self.gru(x)
        #print(x.size()) #returns torch.Size([1, 75, 64]) returns 64 because it is bidirectional 
        x = self.output(x)
        #print(x.size()) #returns torch.Size([1, 75, 20]) 20 as 19 unique classes/characters
        #basically having 75 different timestamps and at each timestamp its returning us a vector of size 20

        #now have to return the loss if we have targets
        #here using CTC loss function because it makes sense when it comes to connections and sequences - basically our use case

        x = x.permute(1,0,2)  #very important for formatting of the CTC Loss function as the t
        #print(x.size())

        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2) #implemented in the pytorch using log_softmax 
            input_lengths = torch.full(
                size=(bs, ),
                fill_value=log_softmax_values.size(0),
                dtype= torch.int32
            )
            #print(input_lengths)  # gives torch[75]
            target_lengths = torch.full(
                size=(bs, ),
                fill_value=targets.size(1),
                dtype= torch.int32
            )
            #print(target_lengths) # torch[5] output can have different sizes but input will have only 1 size tensor of 75 features
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values,
                targets,
                input_lengths,
                target_lengths
            )

            return x,loss
        return x, None


if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand(1,3,75,300)
    target = torch.randint(1,20,(1,5))
    #target = torch.randint(1,20,(5,5))
    x, loss = cm(img,target)

    

