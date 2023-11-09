import numpy as np
from RLTrader import utils

class Agent:
  STATE_DIM = 3

  TRADING_CHARGE = 0.00015 #* 거래 수수료 (0.015%)
  TRADING_TAX = 0.002 #* 거래세 (0.2%)

  #? 행동
  ACTION_BUY = 0 #* 매수
  ACTION_SELL = 1 #* 매도
  ACTION_HOLD = 2 #* 관망
  ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD] #* 신경망에서 활용할 행동
  NUM_ACTIONS = len(ACTIONS) #* 신경망에서 고려할 출력값의 개수

  def __init__(self, environment, initial_balance, min_trading_price, max_trading_pirce):
    #? 주식 거래를 위한 환경 설정
    self.environment = environment #* 현재 주식 가격을 가져오기 위해 환경 참조
    self.initial_balance = initial_balance #* 초기 자본금
    self.min_trading_price = min_trading_price #* 최소 단일 매매 금액
    self.max_trading_pirce = max_trading_pirce #* 최대 단일 매매 금액

    #? Agent 클래스 속성
    self.balance = initial_balance #* 현재 현금 잔고
    self.num_stocks = 0 #* 보유 주식 수
    self.portfolio_value = 0 #* 포트폴리오 가치(PV) : balence + num_stocks * 현재 주식 가격
    self.num_buy = 0 #* 매수 횟수
    self.num_sell = 0 #* 매도 횟수
    self.num_hold = 0 #* 관망 횟수

    #? Agent 클래스의 상태
    self.ratio_hold = 0 #* 주식 보유 비율
    self.profitloss = 0 #* 손익률
    self.avg_buy_price = 0 #* 주당 매수 단가

  def reset(self):
    self.balance = self.initial_balance
    self.num_stocks = 0 
    self.portfolio_value = self.initial_balance
    self.num_buy = 0
    self.num_sell = 0 
    self.num_hold = 0 
    self.ratio_hold = 0 
    self.profitloss = 0 
    self.avg_buy_price = 0
  
  def set_balance(self, balance):
    #TODO: 에이전트의 초기 자본금 설정
    self.initial_balance = balance

  def get_states(self):
    #TODO: 에이전트의 상태 반환
    self.ratio_hold = self.num_stocks * self.environment.get_price() / self.portfolio_value
    return (
      self.ratio_hold,
      self.profitloss,
      if self.avg_buy_price > 0:
        (self.environment.get_price() / self.avg_buy_price) - 1
      else :
        0
    )
  
  def decide_action(self, pred_value, pred_policy, epsilon):
    #TODO: 에이전트의 행동 결정 (무작위 or 신경망)
    confidence = 0.

    pred = pred_policy
    if pred is None:
      pred = pred_value

    if pred is None:
      #* 예측 값이 없는 경우 탐험
      epsilon = 1
    else:
      #* 값이 모두 같은 겅우 탐험
      maxpred = np.max(pred)
      if (pred == maxpred).all():
        epsilon = 1

    #? epsilron의 확률로 무작위하게 행동 결정
    if np.random.rand() < epsilon:
      exploration = True
      action = np.random.randint(self.NUM_ACTIONS)
    else:
      exploration = False
      action = np.argmax(pred)

    #? 신경망을 통해 행동 결정
    confidence = .5
    if pred_policy is not None:
      confidence = pred[action]
    elif pred_value is not None:
      confidence = utils.sigmoid(pred[action])

    return action, confidence, exploration

  def validate_action(self, action):
    #TODO: 행동에 대한 유효성 검사
    if action == Agent.ACTION_BUY:
      #* 적어도 1주를 살 수 있는지 확인
      if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
        return False
    elif action == Agent.ACTION_SELL:
      #* 주식 잔고가 있는지 확인
      if self.num_stocks <= 0:
        return False
    return True

  def decide_trading_unit(self, confidence):
    #TODO: 결정한 행동의 신뢰도에 따라서 매수 or 매도 단위 조정
    if np.isnan(confidence):
      return self.min_trading_price
    added_trading_price = max(min(
      int(confidence * (self.max_trading_pirce - self.min_trading_price)),
      self.max_trading_pirce - self.min_trading_price), 0)
    trading_price = self.min_trading_price + added_trading_price
    return max(int(trading_price / self.environment.get_price()), 1)

  def act(self, action, confidence):
    #TODO: 투자 행동 수행
    if not self.validate_action(action):
      action = Agent.ACTION_HOLD

    #? 환경에서 현재 가격 얻기
    curr_price = self.environment.get_price()

    #TODO: 매수
    if action == Agent.ACTION_BUY:
      #? 매수할 단위 파악
      trading_unit = self.decide_trading_unit(confidence)
      balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit

      #? 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
      if balance < 0:
        trading_unit = min(
          int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
          int(self.max_trading_pirce / curr_price)
        )

      #? 수수료를 적용하여 총 매수 금액 산정
      invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
      if invest_amount > 0:
        self.avg_buy_price = #* 주당 매수 단가 갱신
          self.avg_buy_price * self.num_stocks + curr_price * trading_unit) 
          / (self.num_stocks + trading_unit)
        self.balance -= invest_amount #* 보유 현금 갱신
        self.num_stocks += trading_unit #* 보유 주식 수 갱신
        self.num_buy += 1 #* 매수 횟수 증가

    #TODO: 매도
    elif action == Agent.ACTION_SELL:
      #? 매도할 단위 파악
      trading_unit = self.decide_trading_unit(confidence)

      #? 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
      trading_unit = min(trading_unit, self.num_stocks)

      #? 수수료를 적용하여 총 매수 금액 산정
      invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
      if invest_amount > 0:
        self.avg_buy_price = #* 주당 매수 단가 갱신
          (self.avg_buy_price * self.num_stocks - curr_price * trading_unit)
          / (self.num_stocks - trading_unit)
        self.balance += invest_amount #* 보우 현금 갱신
        self.num_stocks -= trading_unit #* 보유 주식 수 갱신
        self.num_sell += 1 #* 매도 횟수 증가

    #TODO: 관망
    elif action == Agent.ACTION_HOLD:
      self.num_hold += 1 #* 관망 횟수 증가

    #? 포트폴리오 가치 갱신
    self.portfolio_value = self.balance + curr_price * self.num_stocks
    self.profitloss = self.portfolio_value / self.initial_balance - 1
    return self.profitloss