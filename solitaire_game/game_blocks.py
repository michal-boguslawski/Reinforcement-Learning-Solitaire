import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from .utils import dict_figures, dict_colors, adjust_to_print, card_relationship, Card
        

class Foundation:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.foundation = [[] for _ in range(4)]
        
    def is_end(self) -> False:
        for pile in self.foundation:
            if len(pile) == 0:
                return False
            elif pile[-1].figure < 12:
                return False
        
        return True
        
    @staticmethod
    def __get_card_name(pile: List[Card]) -> str:
        if len(pile) == 0:
            return "*"
        else:
            return pile[-1].__str__()
        
    def __str__(self):
        output_string = "|"
        for pile in self.foundation:
            output_string += f"{self.__get_card_name(pile):{adjust_to_print}}|"
        output_string += "\n"
        return output_string
    
    def push(self, card):
        if len(self.foundation[card.color]) == 0 and card.figure > 0:
            result = False
        elif card.figure == 0:
            self.foundation[card.color].append(card)
            result = True
        elif card_relationship(card_from=card, card_to=self.foundation[card.color][-1], type="fl"):
            self.foundation[card.color].append(card)
            result = True
        else:
            result = False
        
        return result


class Waste:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.waste = []
        
    @staticmethod
    def __get_card_name(pile: List[Card]) -> str:
        if len(pile) == 0:
            return "*"
        else:
            return pile[-1].__str__()
        
    def __str__(self):
        return self.__get_card_name(self.waste)
    
    def __len__(self):
        return len(self.waste)
    
    def clear(self) -> List[Card]:
        cards = []
        while len(self.waste) > 0:
            cards.append(self.waste.pop())
        return cards
    
    def append(self, card) -> bool:
        self.waste.append(card)
        return True
    
    def get(self) -> Card | None:
        if len(self.waste) == 0:
            return None
        return self.waste[-1]
    
    def remove_face(self) -> bool:
        if len(self.waste) == 0:
            return False
        else:
            self.waste.pop()
        return True


class Stock:
    def __init__(self):
        self.reset()
        self.stock = []
    
    def reset(self, cards: None | List[Tuple[npt.NDArray, npt.NDArray]] = None):
        self.stock = []
        if cards:
            for card in cards:
                self.stock.append(Card(*card))
                
    def __len__(self) -> str:
        return len(self.stock)
    
    def extend(self, cards: List[Card]) -> bool:
        self.stock.extend(cards)
        return True
    
    def get(self) -> Card:
        return self.stock.pop()
    
    def utilize(self, waste: Waste):
        if len(self) == 0 and len(waste) > 0:
            cards = waste.clear()
            result = self.extend(cards)
        
        elif len(self) == 0 and len(waste) == 0:
            result = False
        
        else:
            card = self.get()
            result = waste.append(card)
        return result


class Tableau:
    def __init__(self):
        self.reset()
        
    def reset(self, cards: None | List[Tuple[npt.NDArray, npt.NDArray]] = None):
        self.piles = [[] for _ in range(7)]
        self.faceup = np.arange(7)
        if cards:
            for pile in range(7):
                for _ in range(pile + 1):
                    self.piles[pile].append(Card(*cards.pop()))
                    
    @staticmethod
    def __get_card_name(pile: List[Card], index: int, faceup: int, default_value: str = " ") -> str:
        if index < len(pile):
            if index < faceup:
                return "*"
            else:
                return pile[index].__str__()
        else:
            return default_value
                    
    def __str__(self) -> str:
        length = max([len(pile) for pile in self.piles])
        output_string = ""
        for i in range(length):
            help_string = "|"
            for pile, faceup in zip(self.piles, self.faceup):
                help_string += f"{self.__get_card_name(pile=pile, index=i, faceup=faceup):{adjust_to_print}}|"
            output_string += help_string + "\n"
        return output_string
    
    def get_pile(self, pile_index: int) -> List[Card] | None:
        pile = self.piles[pile_index]
        if len(pile) == 0:
            return None
        
        faceup_index = self.faceup[pile_index]
        return pile[faceup_index:]
    
    def remove(self, pile_index: int, cards: List[Card]) -> bool:
        for card in cards:
            self.piles[pile_index].remove(card)
        self.faceup[pile_index] = max(min(self.faceup[pile_index], len(self.piles[pile_index]) - 1), 0)
        return True
    
    def move(self, cards: List[Card], pile_index_to: int) -> bool:
        pile = self.piles[pile_index_to]
        result = False
        cards_to_remove = []
        for i, card in enumerate(cards):
            if len(pile) == 0 and card.figure == 12:
                result = True
                break
            else:
                last_card = pile[-1]
                if card_relationship(card_from=card, card_to=last_card, type="cn,fb"):
                    result = True
                    break
        if result:
            while len(cards[i:]) > 0:
                card = cards.pop(i)
                pile.append(card)
                cards_to_remove.append(card)
        return result, cards_to_remove
