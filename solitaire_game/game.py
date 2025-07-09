import numpy as np

from .utils import adjust_to_print
from .game_blocks import Tableau, Foundation, Stock, Waste

    
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
        output_string += "Stock size".ljust(adjust_to_print + 1) + "|" + "Waste".ljust(adjust_to_print + 1)
        output_string += "".ljust(adjust_to_print + 1) + "Foundation\n"
        output_string += f"|{len(self.stock):<{adjust_to_print}}|{self.waste.__str__():{adjust_to_print}}|"
        output_string += "".ljust(adjust_to_print) + self.foundation.__str__()
        output_string += "Tableau:\n"
        output_string += self.tableau.__str__()
        return output_string
    
    def play(self):
        while not self.foundation.is_end():
            print(self)
            try:
                pile_from = int(input("From pile: "))
                pile_to = int(input("To pile: "))
                _ = self.move(pile_from=pile_from, pile_to=pile_to)
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
            result = self.stock.utilize(self.waste)
        
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
        
        return self
