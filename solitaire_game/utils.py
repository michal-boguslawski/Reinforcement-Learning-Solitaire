class Card:
    def __init__(self, figure: int, color: int):
        self.figure = figure
        self.color = color
        
    def __str__(self):
        return f"{dict_figures[self.figure]} of {dict_colors[self.color]}"

dict_figures = {
        0: "Ace",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
        5: "6",
        6: "7",
        7: "8",
        8: "9",
        9: "10",
        10: "Jack",
        11: "Queen",
        12: "King"
    }

dict_colors = {
    0: "Hearts",
    1: "Diamonds",
    2: "Clubs",
    3: "Spades"
}

adjust_to_print = 17


def card_relationship(card_from: Card, card_to: Card, type: str):
    type_list = type.split(",")
    cond_list = []
    
    # if colors extact match
    if "cem" in type_list:
        cond_list.append(card_from.color == card_to.color)
        
    # if colors don't extact match
    if "cen" in type_list:
        cond_list.append(card_from.color != card_to.color)
    
    # if colors match
    if "cm" in type_list:
        cond_list.append(( card_from.color // 2 ) == ( card_to.color // 2 ))
        
    # if colors don't match
    if "cn" in type_list:
        cond_list.append(( card_from.color // 2 ) != ( card_to.color // 2 ))
    
    #if card to figure is bigger by one than card from
    if "fb" in type_list:
        cond_list.append(card_to.figure - card_from.figure == 1)
        
    # if card to figure is lower by one than card from
    if "fl" in type_list:
        cond_list.append(card_from.figure - card_to.figure == 1)
        
    return len(cond_list) > 0 and all(cond_list)
