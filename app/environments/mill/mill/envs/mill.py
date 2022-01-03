import gym
import numpy as np
import re

import config

from stable_baselines import logger


class Player:
    def __init__(self, id, piece_id):
        self.id = id
        self.piece_id = piece_id
        self.pieces_to_place = 9


class AbstractAction:

    # Returns whether the player built a mill by this action
    def execute(self, mill_env):
        print("execute")
        return False

    # whether the action is valid in the current state of the given environment
    def is_legal(self, mill_env):
        return False

    def is_always_illegal(self):
        # always illegal if direction is invalid on mill board
        return False


class PlaceAction(AbstractAction):

    def __init__(self, target_field_id):
        self.target_field_id = target_field_id

    def execute(self, mill_env):
        print('Execute PlaceAction ' + str(self))
        if not self.is_legal(mill_env):
            raise Exception('Illegal action' + str(self))

        # place a new piece for current player
        mill_env.board[self.target_field_id] = mill_env.current_player.piece_id

        # reduce number of pieces left to place
        mill_env.current_player.pieces_to_place -= 1

        # Check for new mill around target_field
        return is_field_part_of_mill(mill_env.board, self.target_field_id, mill_env.current_player.piece_id)

    def is_legal(self, mill_env):
        if self.is_always_illegal():
            return False
        if mill_env.game_phase != GAME_PHASE_PLACE_PIECES:
            return False

        # target field must be empty
        return mill_env.board[self.target_field_id] is EMPTY_FIELD

    def __str__(self):
        return 'PlaceAction(target=' + str(self.target_field_id) + ')'


class RemoveAction(AbstractAction):

    def __init__(self, target_field_id):
        self.target_field_id = target_field_id

    def execute(self, mill_env):
        print('Execute RemoveAction ' + str(self))
        if not self.is_legal(mill_env):
            raise Exception('Illegal action' + str(self))

        # place a new piece for current player
        mill_env.board[self.target_field_id] = mill_env.current_player.piece_id

        # Check for new mill around target_field
        return is_field_part_of_mill(mill_env.board, self.target_field_id, mill_env.current_player.piece_id)

    def is_legal(self, mill_env):
        if self.is_always_illegal():
            return False
        if mill_env.game_phase != GAME_PHASE_REMOVE_OPP_PIECE:
            return False

        other_player_num = (mill_env.current_player_num + 1) % 2
        other_player_piece_id = mill_env.players[other_player_num].piece_id

        # target field must be a piece of opponent
        return mill_env.board[self.target_field_id] == other_player_piece_id

    def __str__(self):
        return 'RemoveAction(target=' + str(self.target_field_id) + ')'


class MoveAction(AbstractAction):

    def __init__(self, field_id, direction):
        self.field_id = field_id
        self.direction = direction

    def execute(self, mill_env):
        if not self.is_legal(mill_env):
            raise Exception('Illegal action' + str(self))

        target_field = get_field_in_direction(self.field_id, self.direction)

        # move the piece from field to target_field
        mill_env.board[self.field_id] = EMPTY_FIELD
        mill_env.board[target_field] = mill_env.current_player.piece_id

        # Check for new mill around target_field
        return is_field_part_of_mill(mill_env.board, target_field, mill_env.current_player.piece_id)

    def is_always_illegal(self):
        # always illegal if direction is invalid on mill board
        return not is_valid_direction(self.field_id, self.direction)

    def is_legal(self, mill_env):
        if self.is_always_illegal():
            return False
        if mill_env.game_phase != GAME_PHASE_MOVE:
            return False

        current_player = mill_env.current_player

        # 1. field must have a players piece
        if mill_env.board[self.field_id] != current_player.piece_id:
            return False

        # 2. target field must be empty
        target_field = get_field_in_direction(self.field_id, self.direction)
        if mill_env.board[target_field] != EMPTY_FIELD:
            return False

        return True

    def __str__(self):
        return 'MoveAction(field_id=' + str(self.field_id) + ', dir=' + self.direction + ')'


class JumpAction(AbstractAction):

    def __init__(self, origin_field_id, target_field_id):
        self.origin_field_id = origin_field_id
        self.target_field_id = target_field_id

    def execute(self, mill_env):
        if not self.is_legal(mill_env):
            raise Exception('Illegal action' + str(self))

        # move the piece from origin_field to target_field
        mill_env.board[self.origin_field_id] = EMPTY_FIELD
        mill_env.board[self.target_field_id] = mill_env.current_player.piece_id

        # Check for new mill around target_field
        return is_field_part_of_mill(mill_env.board, self.target_field_id, mill_env.current_player.piece_id)

    def is_legal(self, mill_env):
        if self.is_always_illegal():
            return False
        if mill_env.game_phase != GAME_PHASE_JUMP:
            return False

        current_player = mill_env.current_player

        # 1. field must have a players piece
        if mill_env.board[self.origin_field_id] != current_player.piece_id:
            return False

        # 2. target field must be empty
        if mill_env.board[self.target_field_id] != 0:
            return False

        return True

    def __str__(self):
        return 'JumpAction(origin_field=' + str(self.origin_field_id) + ', target_field=' + self.target_field_id + ')'


GAME_PHASE_PLACE_PIECES = 0
GAME_PHASE_MOVE = 1
GAME_PHASE_REMOVE_OPP_PIECE = 2
GAME_PHASE_JUMP = 3

# move directions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

EMPTY_FIELD = 0

UP_FIELD_IDS = {
    21: 9, 9: 0,
    18: 10, 10: 3,
    15: 11, 11: 6,
    7: 4, 4: 1,
    22: 19, 19: 16,
    17: 12, 12: 8,
    20: 13, 13: 5,
    23: 14, 14: 2
}
DOWN_FIELD_IDS = {
    0: 9, 9: 21,
    3: 10, 10: 18,
    6: 11, 11: 15,
    1: 4, 4: 7,
    16: 19, 19: 22,
    8: 12, 12: 17,
    5: 13, 13: 20,
    2: 14, 14: 23
}


# returns the field id when moving from the given field in
# the given direction
def get_field_in_direction(field_id, direction):
    if direction == LEFT:
        if field_id % 3 > 0:
            return field_id - 1
        else:
            return None  # invalid

    if direction == RIGHT:
        if field_id % 3 < 2:
            return field_id + 1
        else:
            return None  # invalid

    if direction == UP:
        return UP_FIELD_IDS.get(field_id)

    if direction == DOWN:
        return DOWN_FIELD_IDS.get(field_id)

    raise Exception('Invalid direction: ' + direction)


def is_valid_direction(field_id, direction):
    return get_field_in_direction(field_id, direction) is not None


def is_field_part_of_mill(board, target_field, piece_id):
    if board[target_field] != piece_id:
        # should not be the case, because we call this only for fields with piece
        return False

    # check for horizontal mill
    if is_valid_direction(target_field, LEFT):
        if is_valid_direction(target_field, RIGHT):
            # a, target, b
            a = get_field_in_direction(target_field, LEFT)
            b = get_field_in_direction(target_field, RIGHT)
            if board[a] == board[b] == piece_id:
                return True
        else:
            # b, a, target
            a = get_field_in_direction(target_field, LEFT)
            b = get_field_in_direction(a, LEFT)
            if board[a] == board[b] == piece_id:
                return True
    else:
        # target, a, b
        a = get_field_in_direction(target_field, RIGHT)
        b = get_field_in_direction(a, RIGHT)
        if board[a] == board[b] == piece_id:
            return True

    # check for vertical mill
    if is_valid_direction(target_field, UP):
        if is_valid_direction(target_field, DOWN):
            #   a
            # target
            #   b
            a = get_field_in_direction(target_field, UP)
            b = get_field_in_direction(target_field, DOWN)
            if board[a] == board[b] == piece_id:
                return True
        else:
            #   b
            #   a
            # target
            a = get_field_in_direction(target_field, UP)
            b = get_field_in_direction(a, UP)
            if board[a] == board[b] == piece_id:
                return True
    else:
        # target
        #   a
        #   b
        a = get_field_in_direction(target_field, DOWN)
        b = get_field_in_direction(a, DOWN)
        if board[a] == board[b] == piece_id:
            return True
    # no mill found
    return False


# TODO Remove ??
def field_id_to_xy(field_id):
    y = field_id // 3 # int division
    if y > 3:
        y = y - 1 # row 3 and 4 are actually on the same level

    x = field_id % 3


class MillEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(MillEnv, self).__init__()
        self.name = 'tictactoe'
        self.manual = manual
        
        self.grid_width = 3
        self.grid_height = 8
        self.n_players = 2
        self.num_squares = self.grid_width * self.grid_height
        self.grid_shape = (self.grid_width, self.grid_height)

        # self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(2,))
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape)

        actions = []
        for field_id in range(NUM_FIELDS):
            actions.append(PlaceAction(field_id))
            actions.append(RemoveAction(field_id))
            for direction in range(NUM_DIRECTIONS):
                actions.append(MoveAction(field_id, direction))
            for target_field_id in range(NUM_FIELDS):
                actions.append(JumpAction(field_id, target_field_id))
        self.actions = actions

        self.action_space = gym.spaces.Discrete(len(actions))


        self.verbose = verbose

        # other properties are initialized in reset()
       # self.reset()
        

    @property
    def observation(self):
        print('board: ' + str(self.board))
        print('grid_shape: ' + str(self.grid_shape))
        if self.players[self.current_player_num].piece_id == 1:
            position = np.array(self.board).reshape(self.grid_shape)
        else:
            # invert piece id so view is same for agent independent
            # of playing player 1 or 2
            position = np.array([-x for x in self.board]).reshape(self.grid_shape)

        #la_grid = np.array(self.legal_actions).reshape(self.grid_shape)
        #out = np.stack([position,la_grid], axis=-1)
        out = position
        return out

    @property
    def legal_actions(self):
        legal_actions = filter(lambda action: action.is_legal(self), self.actions)

        return range(MAX_JUMP_ACTION)

    def check_game_over(self, mill_built):
        other_player_num = (self.current_player_num + 1) % 2
        other_player_piece_id = self.players[other_player_num].piece_id

        # check player wins
        if mill_built and self.count_pieces(other_player_piece_id) == 3:
            # player has a mill and thus can remove piece
            # Since opponent only has 3 pieces the current player wins the game
            return 1, True

        # check win by no move (can next player move?)
        if not self.has_player_any_move(other_player_num):
            logger.debug("Other player has no moves. Current player wins!")
            return 1, True

        return 0, False

    @property
    def current_player(self):
        return self.players[self.current_player_num]

    def count_pieces(self, piece_id):
        count = 0
        for field in self.board:
            if field == piece_id:
                count += 1
        return count

    def has_player_any_move(self, player_num):
        piece_id = self.players[player_num]

        if self.players[player_num].pieces_to_place > 0:
            # player is still in phase of placing pieces on board
            # there is definitely a possible move
            return True

        if self.count_pieces(piece_id) == 3:
            # player can jump, so an action is definitely possible
            return True

        for field_id in range(NUM_FIELDS):
            if self.board[field_id] != piece_id:
                # skip! only check fields with a players pieces
                continue
            l = get_field_in_direction(field_id, LEFT)
            if (l is not None) and (self.board[l] == EMPTY_FIELD):
                return True
            r = get_field_in_direction(field_id, RIGHT)
            if (r is not None) and (self.board[r] == EMPTY_FIELD):
                return True
            u = get_field_in_direction(field_id, UP)
            if (u is not None) and (self.board[u] == EMPTY_FIELD):
                return True
            d = get_field_in_direction(field_id, DOWN)
            if (d is not None) and (self.board[d] == EMPTY_FIELD):
                return True
        # player does not have any move -> other player wins
        return False

    def step(self, action):

        reward = [0, 0]

        logger.debug('Doing step (action: ' + str(action) + ')')
        logger.debug('Current game phase: ' + str(self.game_phase))

        decoded_action = decode_action(action)
        logger.debug('decoded action: ' + str(decoded_action))

        if not decoded_action.is_legal(self):
            logger.debug('Illegal action!')
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        else:
            mill_built = decoded_action.execute(self)

            if not isinstance(decoded_action, RemoveAction):
                # avoid counting remove action as additional turn in game
                self.turns_taken += 1

            # Check game over
            r, done = self.check_game_over(mill_built)

            # Check for mill
            if mill_built and not done:
                # player built a new mill in this step
                # next step -> same player with remove action
                self.game_phase = GAME_PHASE_REMOVE_OPP_PIECE
                # current_player keeps the same

                # TODO apply reward !?! building a mill is good!
                done = False
                return self.observation, reward, done, {}

            reward = [-r, -r]
            reward[self.current_player_num] = r

        self.done = done

        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2

            if self.current_player.pieces_to_place > 0:
                self.game_phase = GAME_PHASE_PLACE_PIECES
            elif self.count_pieces(self.current_player.piece_id) == 3:
                # player can jump
                self.game_phase = GAME_PHASE_JUMP
            else:
                # default mode
                self.game_phase = GAME_PHASE_MOVE

        return self.observation, reward, done, {}

    def reset(self):
        self.board = [EMPTY_FIELD] * self.num_squares
        self.players = [Player('1', 1), Player('2', -1)]
        self.current_player_num = 0
        self.turns_taken = 0
        self.game_phase = GAME_PHASE_PLACE_PIECES
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
            logger.debug('Game phase: ' + GAME_PHASE_STR[self.game_phase])
            
        # actual render
        output = RENDER_TEMPLATE
        for field_id in range(NUM_FIELDS):
            symbol = get_symbol_for_field_state(self.board[field_id])
            index = TEMPLATE_INDICES[field_id]
            output = output[:index] + symbol + output[index + 1:]

        logger.debug(output)

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Mill!')


NUM_FIELDS = 24
MAX_FIELD_ID = NUM_FIELDS - 1
NUM_DIRECTIONS = 4

# PLACE_ACTION:
# action = field_id
PLACE_ACTION_SIZE = NUM_FIELDS
MAX_PLACE_ACTION = PLACE_ACTION_SIZE - 1


def decode_place_action(action):
    assert 0 <= action <= MAX_PLACE_ACTION

    target_field_id = action
    return PlaceAction(target_field_id)


# REMOVE_ACTION:
# action = MAX_PLACE_ACTION + 1 + field_id
REMOVE_ACTION_SIZE = NUM_FIELDS
MAX_REMOVE_ACTION = MAX_PLACE_ACTION + REMOVE_ACTION_SIZE


def decode_remove_action(action):
    assert MAX_PLACE_ACTION < action <= MAX_REMOVE_ACTION

    target_field_id = action - PLACE_ACTION_SIZE
    return RemoveAction(target_field_id)


# MOVE_ACTION:
# action = MAX_REMOVE_ACTION + 1 + field_id + direction
MOVE_ACTION_SIZE = NUM_FIELDS + NUM_DIRECTIONS
MAX_MOVE_ACTION = MAX_REMOVE_ACTION + MOVE_ACTION_SIZE


def decode_move_action(action):
    assert MAX_REMOVE_ACTION < action <= MAX_MOVE_ACTION

    action_without_offset = action - MAX_REMOVE_ACTION

    direction = action_without_offset % NUM_DIRECTIONS
    field_id = (action_without_offset - direction) / NUM_FIELDS
    return MoveAction(field_id, direction)


# JUMP_ACTION:
# action = MAX_MOVE_ACTION + 1 + field_id + field_id
JUMP_ACTION_SIZE = NUM_FIELDS + NUM_FIELDS
MAX_JUMP_ACTION = MAX_MOVE_ACTION + JUMP_ACTION_SIZE


def decode_jump_action(action):
    assert MAX_MOVE_ACTION < action <= MAX_JUMP_ACTION

    action_without_offset = action - MAX_MOVE_ACTION

    target_field_id = action_without_offset % NUM_FIELDS
    origin_field_id = (action_without_offset - target_field_id) / NUM_FIELDS
    return JumpAction(origin_field_id, target_field_id)


def decode_action(action):
    if action <= MAX_PLACE_ACTION:
        return decode_place_action(action)
    elif action <= MAX_REMOVE_ACTION:
        return decode_remove_action(action)
    elif action <= MAX_MOVE_ACTION:
        return decode_move_action(action)
    elif action <= MAX_JUMP_ACTION:
        return decode_jump_action(action)
    else:
        raise Exception('Invalid action: ' + action)


RENDER_TEMPLATE = """
O--------O--------O
|        |        |
|  O-----O-----O  |
|  |     |     |  |
|  |  O--O--O  |  |
|  |  |     |  |  |
O--O--O     O--O--O
|  |  |     |  |  |
|  |  O--O--O  |  |
|  |     |     |  |
|  O-----O-----O  |
|        |        |
O--------O--------O
"""

# indices at which we can replace the character with the current game state
# (either space ' ', X, or O)
TEMPLATE_INDICES = [m.start() for m in re.finditer('O', RENDER_TEMPLATE)]


def get_symbol_for_field_state(field_state):
    if field_state == EMPTY_FIELD:
        return ' '
    elif field_state == 1:
        return 'O'
    elif field_state == -1:
        return 'X'
    else:
        raise Exception('Invalid field state: ' + field_state)


GAME_PHASE_STR = ['Place Piece', 'Move Piece', 'Remove Opponent Piece', 'Jump with Piece']