from abc import ABC, abstractmethod

class Agent(ABC):
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def action(self):
        pass
    
    @abstractmethod
    def _calculate_loss(self):
        pass
