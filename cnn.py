import torch.nn as nn

class CNN(nn.Module): #합성곱 계층
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(  #네트위크 모델들을 정의해주고 layer보다 가독성이 뛰어나게 코드를 작성
            nn.Conv2d(1, 64, kernel_size=5), #이미지 특징 추출
            nn.PReLU(), #영보다 작은 기울기를 학습시키기 위한 방식
            nn.ZeroPad2d(2), #패딩작업
            nn.MaxPool2d(kernel_size=5, stride=2) #출력 값에서 일부만 취하여 사이즈가 작은 이미지를 만든다 최대 풀링 평균 풀링 확률적 풀링
        )

        self.layer2 = nn.Sequential(   #네트위크 모델들을 정의해주고 layer보다 가독성이 뛰어나게 코드를 작성   
            nn.ZeroPad2d(padding=1),   
            nn.Conv2d(64, 64, kernel_size=3),   
            nn.PReLU(),
            nn.ZeroPad2d(padding=1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.PReLU()
        )

        self.layer5 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.PReLU(),
            nn.ZeroPad2d(1),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Linear(3200, 1024)  
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2) #일부 파라미터를 학습에 반영하지 않고 모델을 일반화하는 방법
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 7)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x): #순방향메서드는 모델이 데이터를 입력받아 학습을 진행하는 일련의 과정을 정의합니다

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.prelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.prelu(x)
        x = self.dropout(x)

        y = self.fc3(x)
        y = self.log_softmax(y)
        return y
