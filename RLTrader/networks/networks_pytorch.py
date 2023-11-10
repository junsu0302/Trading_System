import threading
import abc
import numpy as np

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network:
  lock = threading.Lock() #* 스레드 간의 충돌 방지

  def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse')
    self.input_dim = input_dim #* 입력 데이터의 차원
    self.output_dim = output_dim #* 출력 데이터의 차원
    self.lr = lr #* 신경망 학습 속도
    self.shared_network = shared_network #* 신경망의 상단부(여러 신경망 공유 가능)
    self.activation = activation #* 신경망의 출력 레이어 활성화 함수
    self.loss = loss #* 신경망의 손실 함수

    #? 신경망 입력 데이터의 형태 설정
    inp = None #* 신경망 입력 데이터의 형태
    if hasattr(self, 'num_steps'):
      inp = (self.num_steps, input_dim) #* CNN, LSTM
    else:
      inp = (self.input_dim,) #* DNN

    #? 공유 신경망 사용
    self.head = None
    if self.shared_network is None:
      self.head = self.get_network_head(inp, self.output_dim)
    else:
      self.head = self.shared_network

    #? 신경망 모델 구성 (활성화 함수, 최적화 기법, 손실 함수 정의)
    self.model = torch.nn.Sequential(self.head) #* 모델 생성

    #* 모델의 활성화 함수 선택
    if self.activation == 'linear':
      pass
    elif self.activation == 'relu':
      self.model.add_module('activation', torch.nn.ReLU())
    elif self.activation == 'leaky_relu':
      self.model.add_module('activation', torch.nn.LeakyReLU())
    elif self.activation == 'sigmoid':
      self.model.add_module('activation', torch.nn.Sigmoid())
    elif self.activation == 'tanh':
      self.model.add_module('activation', torch.nn.Tanh())
    elif self.activation == 'softmax':
      self.model.add_module('activation', torch.nn.Softmax(dim=1))
    
    #* 모델의 가중치 초기화
    self.model.apply(Network.init_weights)
    self.model.to(divice)

    #* 모델의 최적화 기법 선택
    self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)

    #* 모델의 손실 함수 선택
    self.criterion = None
    if loss == 'mse':
      self.criterion = torch.nn.MSELoss()
    elif loss == 'binary_crossentropy':
      self.criterion = torch.nn.BCELoss()

  def predict(self, sample):
    #TODO: 예측 함수
    with self.lock:
      self.model.eval() #* 모델을 평가 모드로 전환하여 학습에만 사용되는 계층 비활성화
      with torch.no_grad():
        x = torch.from_numpy(sample).float().to(device)
        pred = self.model(x).detach().cpu().numpy()
        pred = pred.flatten()
      return pred

  def train_on_batch(self, x, y):
    #TODO: 학습 함수
    loss = 0.
    with self.lock:
      self.model.train() #* 모델을 학습 모드로 전환
      x = torch.from_numpy(x).float().to(device)
      y = torch.from_numpy(y).float().to(device)
      y_pred = self.model(x)
      _loss = self.criterion(y_pred, y)
      self.optimizer.zero_grad()
      _loss.backward()
      self.optimizer.step()
      loss += _loss.item()
    return loss

  #? 공유 신경망 생성 및 가중치 초기화
  @classmethod
  def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
    #TODO: 공유 신경망 반환
    if net == 'dnn':
      return DNN.get_network_head((input_dim,), output_dim)
    elif net == 'lstm':
      return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
    elif net == 'cnn':
      return CNN.get_network_head((num_steps, input_dim), output_dim)

  @abc.abstractmethod
  def get_network_head(inp, output_dim):
    pass

  @staticmethod
  def init_weights(m):
    #TODO: 신경망에 포함된 모든 계층들의 가중치 초기화 (정규분포 초기화)
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
      torch.nn.init.normal_(m.weight, std=0.01)
    elif isinstance(m, torch.nn.LSTM):
      for weights in m.all_weights:
        for weight in weights:
          torch.nn.init.normal_(weight, std=0.01)

  #? 모델 저장 및 불러오기
  def save_model(self, model_path):
    #TODO: 모델 저장
    if model_path is not None and self.model is not None:
      torch.save(self.model, model_path)

  def load_model(self, model_path):
    #TODO: 모델 불러오기
    if model_path is not None:
      self.model = torch.load(model_path)

class DNN(Network):
  @staticmethod
  def get_network_head(inp, output_dim):
    return torch.nn.Sequential(
      torch.nn.BatchNorm1d(inp[0]),
      torch.nn.Linear(inp[0], 256),
      torch.nn.BatchNorm1d(256),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(256, 128),
      torch.nn.BatchNorm1d(128),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(128, 64),
      torch.nn.BatchNorm1d(64),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(64, 32),
      torch.nn.BatchNorm1d(32),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(32, output_dim),
    )
  
  def train_on_batch(self, x, y):
    x = np.array(x).reshape((-1, self.input_dim))
    return super().train_on_batch(x, y)

  def predict(self, sample):
    sample = np.array(sample).reshape((1, self.input_dim))
    return super().predict(sample)

class LSTMNetwork(Network):
  def __init__(self, *args, num_steps=1, **kwargs):
    self.num_steps = num_steps
    super().__init__(*args, **kwargs)

  @staticmethod
  def get_network_head(inp, output_dim):
    return torch.nn.Sequential(
      torch.nn.BatchNorm1d(inp[0]),
      LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
      torch.nn.BatchNorm1d(128),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(128, 64),
      torch.nn.BatchNorm1d(64),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(64, 32),
      torch.nn.BatchNorm1d(32),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(32, output_dim),
    )

  def train_on_batch(self, x, y):
    x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
    return super().train_on_batch(x, y)

  def predict(self, sample):
    sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
    return super().predict(sample)

class LSTMModule(torch.nn.LSTM):
  def __init__(self, *args, use_last_only=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.use_last_only = use_last_only

  def forward(self, x):
    output, (h_n, _) = super().forward(x)
    if self.use_last_only:
      return h_n[-1]
    return output

class CNN(Network):
  def __init__(self, *args, num_steps=1, **kwargs):
    self.num_steps = num_steps
    super().__init__(*args, **kwargs)

  @staticmethod
  def get_network_head(inp, output_dim)
  kernel_size = 2
  return torch.nn.Sequential(
    torch.nn.BatchNorm1d(inp[0]),
    torch.nn.Conv1d(inp[0], 1, kernel_size),
    torch.nn.BatchNorm1d(1),
    torch.nn.Flatten(),
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(inp[1] - (kernel_size - 1), 128),
    torch.nn.BatchNorm1d(128),
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(128, 64),
    torch.nn.BatchNorm1d(64),
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(64, 32),
    torch.nn.BatchNorm1d(32),
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(32, output_dim),
  )

  def train_on_batch(self, x, y):
    x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
    return super().train_on_batch(x, y)

  def predict(self, sample):
    sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
    return super().predict(sample)