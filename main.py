from solitaire_game.game import Game
from solitaire_game.game_blocks import Card
    
    
if __name__ == "__main__":
    card_test = Card(0, 0)
    print(card_test)
    game = Game()
    game.reset(seed=9)
    game.move(7, 8)
    game.move(6, 5)
    game.move(0, 9)
    game.play()
