import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from helper_functions import dict_figures, dict_colors


adjust_number = 17

class Card:
    def __init__(self, figure: int, color: int):
        self.figure = figure
        self.color = color
        
    def __str__(self):
        return f"{dict_figures[self.figure]} of {dict_colors[self.color]}"
    
class Game:
    """ 
    0 - 6   -> tableau
    7       -> Stock
    8       -> Waste
    9       -> Foundation
    """
    def __init__(self, verbose: bool = False):
        self.tableau = Tableau()
        self.foundation = Foundation()
        self.stock = Stock()
        self.waste = Waste()
        self.verbose = verbose
        
    def reset(self, seed: int | None = None):
        deck = np.arange(52)
        if seed:
            np.random.seed(seed)
        np.random.shuffle(deck)
        colors = deck // 13
        figures = deck % 13
        cards = list(zip(*[figures, colors]))
        self.tableau.reset(cards=cards[:28])
        self.stock.reset(cards=cards[28:])
        self.foundation.reset()
        self.waste.reset()
        
    def __str__(self) -> str:
        output_string = ""
        output_string += "Stock size".ljust(adjust_number + 1) + "|" + "Waste".ljust(adjust_number + 1)
        output_string += "".ljust(adjust_number + 1) + "Foundation\n"
        output_string += f"|{len(self.stock):<{adjust_number}}|{self.waste.__str__():{adjust_number}}|"
        output_string += "".ljust(adjust_number) + self.foundation.__str__()
        output_string += "Tableau:\n"
        output_string += self.tableau.__str__()
        return output_string
    
    def play(self):
        while not self.foundation.is_end():
            print(self)
            try:
                pile_from = int(input("From pile: "))
                pile_to = int(input("To pile: "))
                result = self.move(pile_from=pile_from, pile_to=pile_to)
            except ValueError:
                print("Wrong move")
    
    def move(self, pile_from: int, pile_to: int) -> bool:
        result = False
        if pile_from < 7 and pile_to < 7:
            # from Tableau to Tableau
            cards = self.tableau.get_pile(pile_index=pile_from)
            if cards:
                result, cards_to_remove = self.tableau.move(cards=cards, pile_index_to=pile_to)
                self.tableau.remove(pile_index=pile_from, cards=cards_to_remove)
                
        elif pile_from < 7 and pile_to == 9:
            # from Tableau to Foundation
            pile = self.tableau.get_pile(pile_from)
            if pile:
                if self.foundation.push(pile[-1]):
                    result = self.tableau.remove(pile_index=pile_from, cards=[pile[-1]])
        
        elif pile_from == 7 and pile_to == 8:
            # from Stock to Waste
            if len(self.stock) == 0 and len(self.waste) > 0:
                cards = self.waste.clear()
                result = self.stock.extend(cards)
            
            elif len(self.stock) == 0 and len(self.waste) == 0:
                result = False
            
            else:
                card = self.stock.get()
                result = self.waste.append(card)
        
        elif pile_from == 8 and pile_to < 7:
            # from Waste to Tableau
            card = self.waste.get()
            if card:
                move_result, _ = self.tableau.move(cards=[card], pile_index_to=pile_to)
                if move_result:
                    result = self.waste.remove_face()
        
        elif pile_from == 8 and pile_to == 9:
            # from Waste to Foundation
            card = self.waste.get()
            if card:
                if self.foundation.push(card):
                    result = self.waste.remove_face()
        
        elif pile_from == 9 and pile_to < 7:
            # from Foundation to Tableau
            pass
        
        if not result:
            print("!!!!!Unlegal move!!!!!")
        
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
                help_string += f"{self.__get_card_name(pile=pile, index=i, faceup=faceup):{adjust_number}}|"
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
                if card.color // 2 != last_card.color // 2 and last_card.figure - card.figure == 1:
                    result = True
                    break
        if result:
            while len(cards[i:]) > 0:
                card = cards.pop(i)
                pile.append(card)
                cards_to_remove.append(card)
        return result, cards_to_remove
        

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
            output_string += f"{self.__get_card_name(pile):{adjust_number}}|"
        output_string += "\n"
        return output_string
    
    def push(self, card):
        if len(self.foundation[card.color]) == 0 and card.figure > 0:
            result = False
        elif card.figure == 0:
            self.foundation[card.color].append(card)
            result = True
        elif card.figure - self.foundation[card.color][-1].figure == 1:
            self.foundation[card.color].append(card)
            result = True
        else:
            result = False
        
        return result
        
class Stock:
    def __init__(self):
        self.reset()
    
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
    
    
if __name__ == "__main__":
    card_test = Card(0, 0)
    print(card_test)
    game = Game()
    # game.reset(seed=42)
    game.reset(seed=9)
    game.move(7, 8)
    game.move(6, 5)
    game.play()
