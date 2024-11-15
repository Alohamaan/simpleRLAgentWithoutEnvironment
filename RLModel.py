import random
import torch
from torch import nn
import torch.nn.init as init
import tqdm

"""ВАЖНО! все пустые методы - прсосто объявлены, чтобы была хоть какая-то структура среды, просто для удобства, 
   иначе писать агента и систему наград для него очень неудобно, так это делать более комфортно"""

class SomeEnv:

    def __init__(self):
        super(SomeEnv, self).__init__()

        self.actions = {'buy' : self.buy, 'sell' : self.sell, 'hold' : self.hold}

        self.reward = 0
        self.done = False
        self.state = None
        self.dealTypes = ['short', 'long']
        self.openedDeals = {} # тут будут сохраняться открытые сделки в виде пары "(имя актива, тип сделки" : количество

        return

    def reset(self):
        """Resets the environment."""

        #do some reset stuff and change self.state

        pass

    def buy(self, typeOfDeal, amount):
        """Buys a given amount of an active."""

        pass

    def sell(self, activeName, amount):
        """Sells a given amount of an active."""

        pass

    def hold(self, activeName, time):
        """hold a given active for a given time."""

        pass

    def getCurrentState(self):

        return self.state

    def step(self, action = None, dealType = 'short', activeName = None, amount = 0):
        """Performs action and returns state, reward, done."""

        if action == None:
            action = random.choice(self.actions.keys())

        self.actions[action]()
        self.state, self.reward, self.done = self.evaluateAction(action, dealType)

        return self.state, self.reward, self.done

    def evaluateAction(self, action, dealType, activeName = None, amount = 0):
        openedDealsKeys = set(self.openedDeals.keys())
        reward = 0
        done = False
        previousOpenedDeals = {x[1] for x in openedDealsKeys if x[0] == activeName}

        if ((action == 'buy' and dealType == 'short' and 'long' not in previousOpenedDeals and amount == self.openedDeals[(activeName,'sell')])
                or (dealType == 'long' and action == 'buy' and 'short' not in previousOpenedDeals)):

            #закрываем short или открываем long

            if (activeName, dealType) not in openedDealsKeys:
                self.openedDeals[(activeName, dealType)] = amount
                reward = 150 if dealType == 'short' else 100
            else:
                self.openedDeals[(activeName, dealType)] += amount



        elif ((action == 'sell' and dealType == 'short')
                or (dealType == 'long' and action == 'sell' and 'short' not in previousOpenedDeals)):
            #закрываем long или открываем short

            try:
                #это можно сделать через условие, но так отработает быстрее
                del self.openedDeals[(activeName, dealType)]
                reward = 150 if dealType == 'long' else 100
            except KeyError:
                print('Нет такого актива!')
                done = True
                reward = -150

        elif action == 'hold':
            reward = 100

        else:
            #так как отработали все варианты, за которые надо награждать
            done = True
            reward = -100

        self.reset()
        state = self.getCurrentState()

        return state, reward, done

class Agent(nn.Module):

    def __init__(self, inputDim: int, learningRate = 0.01, Epsilon = 0.3) -> None:
        super(Agent, self).__init__()

        self.input = nn.Linear(inputDim, 1024)
        self.hidden = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, 1)

        init.uniform_(self.input.weight, -1, 1)
        init.uniform_(self.hidden.weight, -1, 1)
        init.uniform_(self.output.weight, -1, 1)

        self.actions = []
        self.states = []
        self.next_states = []
        self.reward = []
        self.rewardPerEpoch = []
        self.done = []
        self.losses = []
        self.memory = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        self.loss = nn.MSELoss()
        self.Epsilon = Epsilon
        self.activation = nn.LeakyReLU()

        return

    def forward(self, x) -> torch.Tensor:

        x = torch.sigmoid(self.input(x))
        x = torch.sigmoid(self.hidden(x))
        x = self.activation(self.output(x))

        return x

    def remember(self, state, action, reward, next_state, isDone) -> None:

        self.states.append(state)
        self.actions.append(action)
        self.reward.append(reward)
        self.next_states.append(next_state)
        self.done.append(1 - isDone)
        self.memory += 1

        return

    def sampleBatch(self) -> tuple:

        states = torch.FloatTensor(self.states).cuda()
        self.states.clear()

        actions = torch.IntTensor(self.actions).cuda()
        self.actions.clear()

        rewards = torch.IntTensor(self.reward)
        self.reward.clear()

        next_states = torch.FloatTensor(self.next_states).cuda()
        self.next_states.clear()

        dones = torch.IntTensor(self.done)
        self.done.clear()

        self.memory = 0

        return states, actions, rewards, next_states, dones

    def doSomething(self, environment, episodes=1000, maxSteps=1000) -> None:
        for i in range(episodes):
            state = environment.reset()

            for j in range(maxSteps):
                if random.random() < self.Epsilon:
                    action = random.choice(range(2))
                else:
                    answer = self.forward((torch.FloatTensor(state) * torch.FloatTensor([3])).cuda())
                    action = torch.argmax(answer).item()

                next_state, reward, done = environment.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    return

        return

    def trainAgent(self) -> None:
        if self.memory == 0:
            return
        states, actions, rewards, next_states, done = self.samplebatch()
        memory = self.memory

        currentAnswer = self.forward(states)
        nextAnswer = self.forward(next_states)

        currentPredictedValue = currentAnswer[range(memory), actions]
        futurePredictedValue = torch.max(nextAnswer, dim=1)[0]
        predictTarget = rewards + futurePredictedValue * done

        loss = self.loss(predictTarget, currentPredictedValue)
        self.rewardPerEpoch.append(torch.sum(rewards.cpu()).item())
        self.optimizer.zero_grad()
        loss.backward()

        if self.input.weight.grad.norm() < 0.0001:
            self.input.weight.grad.data += torch.FloatTensor([0.001]).cuda()
        self.optimizer.step()

        return

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = SomeEnv()
    agent = Agent(learningRate=0.01, Epsilon=0.2, inputDim=12).to(DEVICE)
    epoch = 10000

    for i in tqdm.tqdm(range(epoch)):
        agent.doSomething(env, episodes=1000, maxSteps=1000)
        agent.trainAgent()
