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
        self.images_path = images_path
        self.card_images_list = [[] for _ in range(4)]
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
        # print(self.reverse_image.get_size())
        
        # load and scale all cards
        for color in range(4):
            for figure in range(13):
                card_name = f"{dict_figures[figure].lower()}_of_{dict_colors[color].lower()}"
                card_image_path = os.path.join(images_path, f"{card_name}{2 if 10 <= figure <= 12 else ""}.png")
                card_image = pygame.image.load(card_image_path).convert_alpha()
                card_image_scaled = pygame.transform.scale(card_image, self.card_size)
                # print(card_image_scaled.get_size())
                self.card_images_list[color].append(card_image_scaled)
                
        
    def processInput(self):
        self.moveCommandX = 0
        self.moveCommandY = 0
        dragging = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
                break

            # elif event.type == pygame.MOUSEBUTTONDOWN:
            #     if card_rect.collidepoint(event.pos):
            #         dragging = True
            #         mouse_x, mouse_y = event.pos
            #         offset_x = card_rect.x - mouse_x
            #         offset_y = card_rect.y - mouse_y

            # elif event.type == pygame.MOUSEBUTTONUP:
            #     dragging = False

            # elif event.type == pygame.MOUSEMOTION and dragging:
            #     mouse_x, mouse_y = event.pos
            #     card_rect.x = mouse_x + offset_x
            #     card_rect.y = mouse_y + offset_y
        
    def __update_cards(self, list_of_cards: list, game_block_type: str) -> None:
        images_list = []
        for id, level1 in enumerate(list_of_cards):
            if isinstance(level1, list):
                level2_list = []
                for card in level1:
                    if card == "*":
                        level2_list.append(self.reverse_image)
                    else:
                        level2_list.append(self.card_images_list[card[0]][card[1]])
                images_list.append(level2_list)
            else:
                if level1 == "*":
                    images_list.append(self.reverse_image)
                else:
                    images_list.append(self.card_images_list[level1[0]][level1[1]])
        self.game_blocks[game_block_type]["list_of_cards"] = images_list

    def update(self):
        list_of_cards = self.game.prepare_list_of_cards()
        for key, value in list_of_cards.items():
            self.__update_cards(list_of_cards=value, game_block_type=key)
            
    def __render_block(self, block: str) -> None:
        block_attrs = self.game_blocks[block]
        start_width = block_attrs["start_width"]
        start_height = block_attrs["start_height"]
        width_step = block_attrs["width_step"]
        height_step = block_attrs["height_step"]
        block_list = block_attrs["list_of_cards"]
        area = Rect(0, 0, *self.card_size)
        
        for level1_id, level1 in enumerate(block_list):
            if isinstance(level1, list):
                for id, card in enumerate(level1):
                    position = (
                        start_width + width_step * id + self.pile_width_step * level1_id,
                        start_height + height_step * id
                    )
                    self.window.blit(card, position, area=area)
                    # print(card.get_size())
            else:
                position = (
                    start_width + width_step * level1_id,
                    start_height + height_step * level1_id
                )
                self.window.blit(level1, position, area=area)

    def render(self):
        self.window.blit(self.bg_image, (0, 0))
        for block in self.game_blocks.keys():
            self.__render_block(block)
        pygame.display.update()  

    def run(self):
        self.game.reset()
        while self._running:
            self.processInput()
            self.update()
            self.render()        
            self.clock.tick(60)

 
if __name__ == "__main__" :
    theApp = UIGame()
    theApp.run()
    pygame.quit()
