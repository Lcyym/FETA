import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as M


class VGG16_IMAGE_16(nn.Module):
    def __init__(self, class_num=100):
        super(VGG16_IMAGE_16, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),
            
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512*25, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 512*25)
        x = self.classifier(x)
        return x


class VGG16_IMAGE_12(nn.Module):
    def __init__(self, class_num=100):
        super(VGG16_IMAGE_12, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),                              # 36
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),                              # 18
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),                              # 9
            
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),                    # 5
            
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*25, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*25)
        x = self.classifier(x)
        return x


class VGG16_IMAGE_8(nn.Module):
    def __init__(self, class_num=100):
        super(VGG16_IMAGE_8, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),                                  # 36
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),                                  # 9
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),                                  # 5

        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 256*5*5)
        x = self.classifier(x)
        return x


class VGG16_IMAGE_4(nn.Module):
    def __init__(self, class_num=100):
        super(VGG16_IMAGE_4, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),              # 36  

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4,4),              # 18
            
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*9*9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 64*9*9)
        x = self.classifier(x)
        return x

class EMNIST_CNN1(nn.Module):
    def __init__(self):
        super(EMNIST_CNN1,self).__init__()

        self.conv1 = nn.Sequential(      
            nn.Conv2d(1,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(       
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,7*7*64)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

class EMNIST_CNN2(nn.Module):
    def __init__(self):
        super(EMNIST_CNN1,self).__init__()

        self.conv1 = nn.Sequential(      
            nn.Conv2d(1,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(       
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        output = out_conv2.view(-1,7*7*64)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

class AlexNet1(nn.Module):
    def __init__(self,class_num=10):
        super(AlexNet1, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.avgpool1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(1024, class_num),
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.avgpool1(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier3(x)
        return F.log_softmax(x, dim=1)


class AlexNet2(nn.Module):
    def __init__(self,class_num=10):
        super(AlexNet2, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.avgpool2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )


        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(1024, class_num),
        )



    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool2(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return F.log_softmax(x, dim=1)


class AlexNet3(nn.Module):
    def __init__(self,class_num=10):
        super(AlexNet3, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.avgpool3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(1024, class_num),
        )


    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.avgpool3(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return F.log_softmax(x, dim=1)


class AlexNet(nn.Module):
    def __init__(self,class_num=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, class_num),
        )

        # self.apply(_weights_init)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def get_model(num=2):
    if num == 0:
        return AlexNet1()
    elif num == 1:
        return AlexNet2()
    else:
        return AlexNet3()
