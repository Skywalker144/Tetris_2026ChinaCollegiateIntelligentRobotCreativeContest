#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Action {
    int rotation_idx;
    int x;
};

struct TetrisState {
    std::vector<std::vector<int>> board;
    std::vector<std::vector<int>> piece_ids;
    int piece_index;
    int score;
};

class Tetris {
public:
    using Cells = std::vector<std::pair<int, int>>;

    static const std::array<char, 7> SHAPES;
    static const std::unordered_map<char, int> PIECE_TO_VALUE;
    static const std::unordered_map<char, std::vector<Cells>> PIECE_ROTATIONS;

    Tetris(
        int width = 10,
        int height = 14,
        std::optional<std::vector<char>> piece_sequence = std::nullopt,
        std::optional<unsigned int> seed = std::nullopt
    );

    TetrisState getInitialState() const;
    TetrisState cloneState(const TetrisState& state) const;
    std::vector<Action> getLegalActions(const TetrisState& state) const;
    TetrisState getNextState(const TetrisState& state, const Action& action) const;
    bool isTerminal(const TetrisState& state) const;
    double getReward(const TetrisState& state) const;
    double getMaxReward() const;

    const std::vector<char>& pieceSequence() const;

private:
    static constexpr double LEFT_CENTER_WEIGHT = 1.0;

    int width_;
    int height_;
    mutable std::mt19937 rng_;
    std::vector<char> piece_sequence_;

    std::vector<char> buildRandomPieceSequence();
    bool canPlace(const std::vector<std::vector<int>>& board, const Cells& cells, int x, int y) const;
    std::optional<int> getDropY(const std::vector<std::vector<int>>& board, const Cells& cells, int x) const;
    double getLeftCenterReward(const TetrisState& state) const;
    void lockPiece(
        std::vector<std::vector<int>>& board,
        std::vector<std::vector<int>>& piece_ids,
        const Cells& cells,
        int x,
        int y,
        int piece_value,
        int piece_id
    ) const;
};
