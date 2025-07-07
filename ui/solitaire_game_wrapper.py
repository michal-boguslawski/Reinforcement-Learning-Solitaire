from solitaire_game.game import Game
from solitaire_game.game_blocks import Waste, Stock, Foundation, Tableau


class SolitaireWrapper(Game):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.list_of_cards = {}
        self.__reset_list_of_cards()
        
    def __reset_list_of_cards(self) -> None:
        self.list_of_cards = {
            "Stock": [],
            "Waste": [],
            "Foundation": [[] for _ in range(4)],
            "Tableau": [[] for _ in range(7)]
        }
        
    def move(self, pile_from: int, pile_to: int) -> dict:
        _ = super().move(pile_from=pile_from, pile_to=pile_to)
        return self.prepare_list_of_cards()
    
    def prepare_list_of_cards(self) -> dict:
        self.__reset_list_of_cards()
        self.__prepare_stock(self.stock)
        self.__prepare_waste(self.waste)
        self.__prepare_foundation(self.foundation)
        self.__prepare_tableau(self.tableau)
        return self.list_of_cards
        
    def __prepare_stock(self, stock: Stock) -> None:
        if len(stock.stock) == 0:
            self.list_of_cards["Stock"].append("*")
        else:
            for _ in range(len(stock.stock)):
                self.list_of_cards["Stock"].append("*")
                
    def __prepare_waste(self, waste: Waste) -> None:
        len_waste = len(waste.waste)
        if len_waste == 0:
            self.list_of_cards["Waste"].append("*")
        else:
            for i, card in enumerate(waste.waste):
                if i == len_waste - 1:
                    self.list_of_cards["Waste"].append((card.color, card.figure))
                else:
                    self.list_of_cards["Waste"].append("*")
                
    def __prepare_foundation(self, foundation: Foundation) -> None:
        for color, pile in enumerate(foundation.foundation):
            if len(pile) == 0:
                self.list_of_cards["Foundation"][color].append("*")
            else:
                for card in pile:
                    self.list_of_cards["Foundation"][color].append((card.color, card.figure))
                
    def __prepare_tableau(self, tableau: Tableau) -> None:
        for pile_id, (pile, faceup) in enumerate(zip(tableau.piles, tableau.faceup)):
            if len(pile) == 0:
                self.list_of_cards["Tableau"][pile_id].append("*")
            else:
                for i, card in enumerate(pile):
                    if i < faceup:
                        self.list_of_cards["Tableau"][pile_id].append("*")
                    else:
                        self.list_of_cards["Tableau"][pile_id].append((card.color, card.figure))
