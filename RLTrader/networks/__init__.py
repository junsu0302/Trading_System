import os

if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
  print('Enabling PyTorch...')
  from RLTrader.networks.networks_pytorch import Network, DNN, LSTMNetwork, CNN
else:
  print('Enabling TensorFlow...')
  from RLTrader.networks.networks_keras import Network, DNN, LSTMNetwork, CNN

#? 해당 패키지가 다음의 클래스를 갖고 있음을 명시
__all__ = ['Network', 'DNN', 'LSTMNetwork', 'CNN']