import os
import pygame
from pygame.locals import *
from pygame import Rect
from solitaire_game.utils import dict_colors, dict_figures
from ui.solitaire_game_wrapper import SolitaireWrapper


# __file__ is the current file (in ui/)
current_dir = os.path.dirname(__file__)
images_path = os.path.join(current_dir, '..', 'images')
images_path = os.path.abspath(images_path)

class UIGame:
    def __init__(self):
        # variables
        self._running = True
        self._display_surf = None
        self.card_size = (167, 242)
        self.pile_width_step = 200
        self.size = self.weight, self.height = 50 + self.pile_width_step * 7, 1000
        
        # game related
        self.game = SolitaireWrapper(verbose=False)
        
        # pygame needed
        pygame.init()
        pygame.display.set_caption("Solitaire Game")
        self.window = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.clock = pygame.time.Clock()
        
        # additional
        self.move_from = -1
        self.move_to = -1
        self.move = False
        self.fullscreen = False
        self.dragging = False
        self.collide_card = None
        self.dragged_cards = []
        self.drag_offset = (0, 0)
        self.images_path = images_path
        self.card_images_list = [[] for _ in range(4)]
        self.render_last = []
        self.__load_images(images_path)
        self.__get_starting_game_blocks()
        
    def __get_starting_game_blocks(self):
        self.game_blocks = {
            "Stock": {
                "start_width": 25,
                "start_height": 40,
                "width_step": 1,
                "height_step": 1,
                "list_of_cards": []
            },
            "Waste": {
                "start_width": 275,
                "start_height": 40,
                "width_step": 1,
                "height_step": 1,
                "list_of_cards": []
            },
            "Foundation": {
                "start_width": 25 + self.pile_width_step * 3,
                "start_height": 40,
                "width_step": 1,
                "height_step": 1,
                "list_of_cards": [[] for _ in range(4)]
            },
            "Tableau": {
                "start_width": 25,
                "start_height": 400,
                "width_step": 0,
                "height_step": 40,
                "list_of_cards": [[] for _ in range(7)]
            }
        }
        
    def __load_images(self, images_path: str) -> None:
        # Load and scale background
        bg_image_path = os.path.join(images_path, "background.jpg")
        bg_image = pygame.image.load(bg_image_path).convert()
        self.bg_image = pygame.transform.scale(bg_image, (1500, 1000))
        
        # load and scale reverse
        reverse_image_path = os.path.join(images_path, "reverse_card.png")
        reverse_image = pygame.image.load(reverse_image_path).convert_alpha()
        self.reverse_image = pygame.transform.scale(reverse_image, self.card_size)
        
        # load and scale transparent
        transparent_image_path = os.path.join(images_path, "transparent_card.png")
        transparent_image = pygame.image.load(transparent_image_path).convert_alpha()
        self.transparent_image = pygame.transform.scale(transparent_image, self.card_size)
        
        # load and scale all cards
        for color in range(4):
            for figure in range(13):
                card_name = f"{dict_figures[figure].lower()}_of_{dict_colors[color].lower()}"
                card_image_path = os.path.join(images_path, f"{card_name}{2 if 10 <= figure <= 12 else ""}.png")
                card_image = pygame.image.load(card_image_path).convert_alpha()
                card_image_scaled = pygame.transform.scale(card_image, self.card_size)
                self.card_images_list[color].append(card_image_scaled)
                
    def __reset_cards_position(self):
        list_of_cards = self.game.prepare_list_of_cards()
        for key, value in list_of_cards.items():
            self.__reset_cards_position_block(list_of_cards=value, game_block_type=key)
                
    def __reset_cards_position_block(self, list_of_cards: list, game_block_type: str) -> None:
        images_list = []
        game_block = self.game_blocks[game_block_type]
        for id, pile_card in enumerate(list_of_cards):
            if isinstance(pile_card, list):
                level2_list = []
                for i, card in enumerate(pile_card):
                    if card == "*":
                        card_image = self.reverse_image
                    elif card == "-":
                        card_image = self.transparent_image
                    else:
                        card_image = self.card_images_list[card[0]][card[1]]
                    
                    card_position = {
                        "pile_id": id,
                        "order_id": i if card != "-" else -1,
                        "image": card_image,
                        "block_type": game_block_type,
                        "relative_width": id * self.pile_width_step + i * game_block["width_step"],
                        "relative_height": i * game_block["height_step"],
                        "draggable":
                            card not in ["-", "*"] 
                            or game_block_type == "Stock"
                    }
                    level2_list.append(card_position)
                images_list.append(level2_list)
            else:
                if pile_card == "*":
                    card_image = self.reverse_image
                elif pile_card == "-":
                    card_image = self.transparent_image
                else:
                    card_image = self.card_images_list[pile_card[0]][pile_card[1]]
                    
                card_position = {
                    "pile_id": 0,
                    "order_id": id if pile_card != "-" else -1,
                    "image": card_image,
                    "block_type": game_block_type,
                    "relative_width": id * game_block["width_step"],
                    "relative_height": id * game_block["height_step"],
                    "draggable": 
                        pile_card not in ["-", "*"]
                        or game_block_type == "Stock"
                }
                
                images_list.append(card_position)
                
        self.game_blocks[game_block_type]["list_of_cards"] = images_list
        
    def __find_collidecard(self, event_pos, allow_empty: bool = False):
        for block in self.game_blocks.values():
            for pile_card in reversed(list(block["list_of_cards"])):
                if isinstance(pile_card, list):
                    for card in reversed(list(pile_card)):
                        if card["draggable"] or allow_empty:
                            card_rect = Rect(
                                block["start_width"] + card["relative_width"],
                                block["start_height"] + card["relative_height"],
                                *self.card_size
                            )
                            if card_rect.collidepoint(event_pos):
                                return card, card_rect
                        
                else:
                    if pile_card["draggable"] or allow_empty:
                        card_rect = Rect(
                            block["start_width"] + pile_card["relative_width"],
                            block["start_height"] + pile_card["relative_height"],
                            *self.card_size
                        )
                        if card_rect.collidepoint(event_pos):
                            return pile_card, card_rect
                    
    def __find_dragged_cards(self, pile_id: int, order_id: int, block_type: str, **kwargs):
        list_of_cards = self.game_blocks[block_type]["list_of_cards"]
        if isinstance(list_of_cards[0], list):
            cards = list_of_cards[pile_id][order_id:]
        else:
            cards = list_of_cards[order_id:]
        self.dragged_cards = cards
    
    def __on_release(self):
        self.dragging = False
        self.collide_card = None
        self.dragged_cards = []
        self.__reset_cards_position()
        
    @staticmethod
    def __calculate_move(block_type: str, pile_id: int):
        if block_type == "Tableau":
            return pile_id
        elif block_type == "Stock":
            return 7
        elif block_type == "Waste":
            return 8
        elif block_type == "Foundation":
            return 9
        return -1
          
                
    def processInput(self):
        self.moveCommandX = 0
        self.moveCommandY = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                break
            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
                break
            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
                else:
                    self.window = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.game.reset()
                self.__reset_cards_position()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                collide_card = self.__find_collidecard(event_pos=event.pos)
                if collide_card:
                    if collide_card[0]["block_type"] == "Stock" and collide_card[0]["order_id"] == -1:
                        self.move_from = 7
                        self.move_to = 8
                        self.move = True
                    elif collide_card:
                        self.dragging = True
                        self.collide_card = collide_card
                        self.drag_offset = (collide_card[1].x - event.pos[0], collide_card[1].y - event.pos[1])
                        self.__find_dragged_cards(**collide_card[0])
                        self.move_from = self.__calculate_move(collide_card[0]["block_type"], collide_card[0]["pile_id"])
                    
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                self.__on_release()
                collide_card = self.__find_collidecard(event_pos=event.pos, allow_empty=True)
                if collide_card:
                    self.move_to = self.__calculate_move(collide_card[0]["block_type"], collide_card[0]["pile_id"])
                    self.move = True
                
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                self.drag_offset = (self.collide_card[1].x - event.pos[0], self.collide_card[1].y - event.pos[1])
                
                
    def update(self):
        if self.move:
            self.game.move(self.move_from, self.move_to)
            self.move_from, self.move_to, self.move = -1, -1, False
            self.__reset_cards_position()
        for card in self.dragged_cards:
            card["width_offset"] = -self.drag_offset[0]
            card["height_offset"] = -self.drag_offset[1]
            
    def __render_block(
            self,
            start_width: int,
            start_height: int,
            list_of_cards: list,
            **kwargs
        ) -> None:
        area = Rect(0, 0, *self.card_size)
        for card in list_of_cards:
            if isinstance(card, list):
                self.__render_block(
                    start_width=start_width,
                    start_height=start_height,
                    list_of_cards=card
                )
            else:
                position = (
                    start_width + card["relative_width"] + card.get("width_offset", 0),
                    start_height + card["relative_height"] + card.get("height_offset", 0)
                )
                blit_values = (card["image"], position, area)
                if card in self.dragged_cards:
                    self.render_last.append(blit_values)
                else:
                    self.window.blit(*blit_values)
                
    def render(self):
        # Get current display size
        if self.fullscreen:
            screen_info = pygame.display.Info()
            screen_width, screen_height = screen_info.current_w, screen_info.current_h
            bg_image = pygame.transform.scale(self.bg_image, (screen_width, screen_height))
        else:
            bg_image = self.bg_image
        self.window.blit(bg_image, (0, 0))
        for block_list in self.game_blocks.values():
            self.__render_block(**block_list)
        while len(self.render_last) > 0:
            self.window.blit(*self.render_last.pop(0))
        pygame.display.update()  
            
    def run(self):
        self.game.reset()
        self.__reset_cards_position()
        self.render()
        pygame.display.update()  
        while self._running:
            self.processInput()
            self.update()
            self.render()
            self.clock.tick(60)
            
if __name__ == "__main__":
    theApp = UIGame()
    theApp.run()
    pygame.quit()
                