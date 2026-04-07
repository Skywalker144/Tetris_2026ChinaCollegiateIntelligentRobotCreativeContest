#include "tetris.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

const std::array<char, 7> Tetris::SHAPES = {'L', 'J', 'S', 'Z', 'T', 'O', 'I'};

const std::unordered_map<char, int> Tetris::PIECE_TO_VALUE = {
    {'L', 1}, {'J', 2}, {'S', 3}, {'Z', 4}, {'T', 5}, {'O', 6}, {'I', 7},
};

const std::unordered_map<char, std::vector<Tetris::Cells>> Tetris::PIECE_ROTATIONS = {
    {'I', {{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {{0, 0}, {0, 1}, {0, 2}, {0, 3}}}},
    {'O', {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}}},
    {'T', {{{0, 0}, {1, 0}, {2, 0}, {1, 1}},
           {{0, 0}, {0, 1}, {1, 1}, {0, 2}},
           {{1, 0}, {0, 1}, {1, 1}, {2, 1}},
           {{1, 0}, {0, 1}, {1, 1}, {1, 2}}}},
    {'S', {{{1, 0}, {2, 0}, {0, 1}, {1, 1}}, {{0, 0}, {0, 1}, {1, 1}, {1, 2}}}},
    {'Z', {{{0, 0}, {1, 0}, {1, 1}, {2, 1}}, {{1, 0}, {0, 1}, {1, 1}, {0, 2}}}},
    {'J', {{{0, 0}, {0, 1}, {1, 1}, {2, 1}},
           {{0, 0}, {1, 0}, {0, 1}, {0, 2}},
           {{0, 0}, {1, 0}, {2, 0}, {2, 1}},
           {{1, 0}, {1, 1}, {1, 2}, {0, 2}}}},
    {'L', {{{2, 0}, {0, 1}, {1, 1}, {2, 1}},
           {{0, 0}, {0, 1}, {0, 2}, {1, 2}},
           {{0, 0}, {1, 0}, {2, 0}, {0, 1}},
           {{0, 0}, {1, 0}, {1, 1}, {1, 2}}}},
};

Tetris::Tetris(
    int width,
    int height,
    std::optional<std::vector<char>> piece_sequence,
    std::optional<unsigned int> seed
)
    : width_(width),
      height_(height),
      rng_(seed.has_value() ? *seed : std::random_device{}()) {
    if (!piece_sequence.has_value()) {
        piece_sequence_ = buildRandomPieceSequence();
        return;
    }

    piece_sequence_.reserve(piece_sequence->size());
    for (char piece : *piece_sequence) {
        char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(piece)));
        if (PIECE_ROTATIONS.find(upper) == PIECE_ROTATIONS.end()) {
            throw std::invalid_argument("Unknown piece type in sequence");
        }
        piece_sequence_.push_back(upper);
    }
}

std::vector<char> Tetris::buildRandomPieceSequence() {
    std::vector<char> sequence;
    std::uniform_int_distribution<int> dist(1, 5);
    for (char piece : SHAPES) {
        int count = dist(rng_);
        for (int i = 0; i < count; ++i) {
            sequence.push_back(piece);
        }
    }
    std::shuffle(sequence.begin(), sequence.end(), rng_);
    return sequence;
}

TetrisState Tetris::getInitialState() const {
    std::vector<std::vector<int>> board(height_, std::vector<int>(width_, 0));
    std::vector<std::vector<int>> piece_ids(height_, std::vector<int>(width_, 0));
    return TetrisState{board, piece_ids, 0, 0};
}

TetrisState Tetris::cloneState(const TetrisState& state) const {
    return state;
}

std::vector<Action> Tetris::getLegalActions(const TetrisState& state) const {
    if (state.piece_index >= static_cast<int>(piece_sequence_.size())) {
        return {};
    }

    const char piece = piece_sequence_[state.piece_index];
    const auto& rotations = PIECE_ROTATIONS.at(piece);
    std::vector<Action> legal_actions;

    for (int rotation_idx = 0; rotation_idx < static_cast<int>(rotations.size()); ++rotation_idx) {
        const Cells& cells = rotations[rotation_idx];
        int max_dx = 0;
        for (const auto& [dx, _] : cells) {
            max_dx = std::max(max_dx, dx);
        }

        for (int x = 0; x < width_ - max_dx; ++x) {
            if (getDropY(state.board, cells, x).has_value()) {
                legal_actions.push_back(Action{rotation_idx, x});
            }
        }
    }

    return legal_actions;
}

TetrisState Tetris::getNextState(const TetrisState& state, const Action& action) const {
    if (state.piece_index >= static_cast<int>(piece_sequence_.size())) {
        throw std::invalid_argument("Game already ended: piece sequence exhausted");
    }

    const char piece = piece_sequence_[state.piece_index];
    const auto& rotations = PIECE_ROTATIONS.at(piece);

    if (action.rotation_idx < 0 || action.rotation_idx >= static_cast<int>(rotations.size())) {
        throw std::invalid_argument("Invalid rotation index");
    }

    const Cells& cells = rotations[action.rotation_idx];
    std::optional<int> drop_y = getDropY(state.board, cells, action.x);
    if (!drop_y.has_value()) {
        throw std::invalid_argument("Illegal action: piece cannot be dropped at this position");
    }

    std::vector<std::vector<int>> next_board = state.board;
    std::vector<std::vector<int>> next_piece_ids = state.piece_ids;

    lockPiece(
        next_board,
        next_piece_ids,
        cells,
        action.x,
        *drop_y,
        PIECE_TO_VALUE.at(piece),
        state.piece_index + 1
    );

    int full_rows = 0;
    for (const auto& row : next_board) {
        bool full = true;
        for (int cell : row) {
            if (cell == 0) {
                full = false;
                break;
            }
        }
        if (full) {
            ++full_rows;
        }
    }

    return TetrisState{next_board, next_piece_ids, state.piece_index + 1, full_rows};
}

bool Tetris::isTerminal(const TetrisState& state) const {
    if (state.piece_index >= static_cast<int>(piece_sequence_.size())) {
        return true;
    }
    return getLegalActions(state).empty();
}

double Tetris::getReward(const TetrisState& state) const {
    const double line_clear_reward = static_cast<double>(state.score);
    if (!isTerminal(state)) {
        return line_clear_reward;
    }
    return line_clear_reward + LEFT_CENTER_WEIGHT * getLeftCenterReward(state);
}

double Tetris::getMaxReward() const {
    return static_cast<double>(height_) + LEFT_CENTER_WEIGHT;
}

const std::vector<char>& Tetris::pieceSequence() const {
    return piece_sequence_;
}

bool Tetris::canPlace(const std::vector<std::vector<int>>& board, const Cells& cells, int x, int y) const {
    for (const auto& [dx, dy] : cells) {
        int bx = x + dx;
        int by = y + dy;
        if (bx < 0 || bx >= width_ || by < 0 || by >= height_) {
            return false;
        }
        if (board[by][bx] != 0) {
            return false;
        }
    }
    return true;
}

std::optional<int> Tetris::getDropY(const std::vector<std::vector<int>>& board, const Cells& cells, int x) const {
    if (x < 0) {
        return std::nullopt;
    }

    int max_dx = 0;
    for (const auto& [dx, _] : cells) {
        max_dx = std::max(max_dx, dx);
    }
    if (x + max_dx >= width_) {
        return std::nullopt;
    }

    int y = 0;
    if (!canPlace(board, cells, x, y)) {
        return std::nullopt;
    }

    while (canPlace(board, cells, x, y + 1)) {
        ++y;
    }
    return y;
}

double Tetris::getLeftCenterReward(const TetrisState& state) const {
    long long occupied_count = 0;
    long long x_sum = 0;

    for (const auto& row : state.board) {
        for (int x = 0; x < static_cast<int>(row.size()); ++x) {
            if (row[x] == 0) {
                continue;
            }
            x_sum += x;
            occupied_count += 1;
        }
    }

    if (occupied_count == 0) {
        return 0.0;
    }
    if (width_ <= 1) {
        return 1.0;
    }

    const double center_x = static_cast<double>(x_sum) / static_cast<double>(occupied_count);
    const double normalized_center = center_x / static_cast<double>(width_ - 1);
    return std::clamp(1.0 - normalized_center, 0.0, 1.0);
}

void Tetris::lockPiece(
    std::vector<std::vector<int>>& board,
    std::vector<std::vector<int>>& piece_ids,
    const Cells& cells,
    int x,
    int y,
    int piece_value,
    int piece_id
) const {
    for (const auto& [dx, dy] : cells) {
        board[y + dy][x + dx] = piece_value;
        piece_ids[y + dy][x + dx] = piece_id;
    }
}
