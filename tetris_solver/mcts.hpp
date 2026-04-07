#pragma once

#include "tetris.hpp"

#include <memory>
#include <optional>
#include <random>
#include <vector>

class MCTS {
public:
    struct Args {
        int num_simulations = 500;
        double exploration_constant = 1.41;
        bool normalize_reward = true;
    };

    explicit MCTS(Tetris& game, Args args, std::optional<unsigned int> seed = std::nullopt);

    std::optional<Action> search(const TetrisState& state);

private:
    struct Node {
        TetrisState state;
        std::optional<Action> action_taken;
        Node* parent;
        std::vector<std::unique_ptr<Node>> children;
        double v;
        int n;
        bool untried_initialized;
        std::vector<Action> untried_actions;

        explicit Node(TetrisState s, std::optional<Action> action = std::nullopt, Node* p = nullptr)
            : state(std::move(s)),
              action_taken(action),
              parent(p),
              v(0.0),
              n(0),
              untried_initialized(false) {}

        bool isFullyExpanded(Tetris& game);
        double getUcb(double exploration_constant) const;
        void update(double value);
    };

    Tetris& game_;
    Args args_;
    double reward_scale_;
    std::mt19937 rng_;

    Node* select(Node* node);
    Node* expand(Node* node);
    double simulate(const TetrisState& state);
    void backpropagate(Node* node, double value);
};
