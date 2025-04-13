import chess
import numpy as np

Action = int
NUM_ACTIONS = 64 * 72


def action_array():
    return np.zeros(NUM_ACTIONS)


def move_to_action(move: chess.Move) -> Action:
    fs = move.from_square

    # for underpromotion the src square is the combination
    # of move direction and the target piece type
    if move.promotion and move.promotion != chess.QUEEN:
        fs = 64 + 3 * (move.to_square - fs - 7) + (move.promotion - 2)

    return 64 * fs + move.to_square


def action_to_move(a: Action) -> chess.Move:
    fs, ts = a // 64, a % 64

    # underpromotion
    if fs >= 64:
        fs -= 64
        p = fs % 3
        fs = ts - 7 - fs // 3
        return chess.Move(fs, ts, p)

    # promotion to queen
    if ts >= 56:
        return chess.Move(fs, ts, chess.QUEEN)

    # normal move
    return chess.Move(fs, ts)


def unpackbits(x):
    x = np.asarray([x], dtype=np.uint64)
    return np.unpackbits(x.view(np.uint8))


def bitboards(board: chess.Board):
    return [
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        *board.occupied_co,
    ]


def castling_to_bits(cr):
    return np.array(
        [
            bool(cr & chess.BB_A1),
            bool(cr & chess.BB_H1),
            bool(cr & chess.BB_A8),
            bool(cr & chess.BB_H8),
        ],
        dtype=np.uint8,
    )


class State:
    def __init__(self, board: chess.Board):
        self.board = board

    def actions(self):
        return map(move_to_action, self.board.legal_moves)

    def push(self, action: int) -> None:
        move = action_to_move(action)
        self.board.push(move)

    def pop(self) -> None:
        self.board.pop()

    def hash(self):
        return self.board._transposition_key()

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    @property
    def reward(self) -> int:
        outcome = self.board.outcome(claim_draw=True)
        r = 0 if outcome.winner is None else outcome.winner * 2 - 1
        return r if outcome.winner == self.board.turn else -r

    def asarrays(self):
        # chess is a symmetrical game, so we can treat both players the
        # same by always viewing the board from white's perspective.
        # this helps reduce the number of positions the model needs
        # to learn by eliminating redundant mirrored states.
        board = self.board.transform(chess.flip_vertical)

        # we also need to flip the castling rights
        cr = board.castling_rights
        if not board.turn:
            cr = chess.flip_vertical(board.castling_rights)

        # bitboards converted to a numpy array of shape (8, 64)
        bbs = np.stack([unpackbits(b) for b in bitboards(board)])

        # if en passant is available we convert its file to binary
        # and use it as flags (e.g [0 1 1 1] for the H file). otherwise,
        # we set all flags to 0 and use a "not-ep" flag instead ([1 0 0 0])
        ep_file = board.ep_square & 7 if board.ep_square else 8
        ep_bits = unpackbits(ep_file)[4:8]

        # castling flags (e.g [1 0 0 1] for white qs and black ks)
        cr_bits = castling_to_bits(cr)

        flags = np.concat([ep_bits, cr_bits])
        return bbs, flags
