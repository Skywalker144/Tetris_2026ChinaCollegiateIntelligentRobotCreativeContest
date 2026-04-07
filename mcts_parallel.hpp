#pragma once

#include "tetris.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

class MCTSParallel {
public:
    struct Args {
        int num_simulations = 500;
        double exploration_constant = 1.41;
        bool normalize_reward = true;
        int num_threads = 0;
        double virtual_loss = 1.0;
    };

    explicit MCTSParallel(Tetris& game, Args args, std::optional<unsigned int> seed = std::nullopt);

    std::optional<Action> search(const TetrisState& state);

private:
    struct Node {
        const TetrisState state;
        std::optional<Action> action_taken;
        Node* parent;
        std::vector<std::unique_ptr<Node>> children;
        double v;
        int n;
        int virtual_n;
        bool untried_initialized;
        std::vector<Action> untried_actions;
        mutable std::mutex mutex;

        explicit Node(TetrisState s, std::optional<Action> action = std::nullopt, Node* p = nullptr)
            : state(std::move(s)),
              action_taken(action),
              parent(p),
              v(0.0),
              n(0),
              virtual_n(0),
              untried_initialized(false) {}

        bool canExpand(Tetris& game);
        Node* expand(Tetris& game, std::mt19937& rng);
        std::vector<Node*> childrenSnapshot();
        int totalVisits() const;
        double ucb(double exploration_constant, int parent_total_visits, double virtual_loss) const;
        void addVirtualVisit();
        void completeVisit(double value);
    };

    Tetris& game_;
    Args args_;
    double reward_scale_;
    unsigned int base_seed_;

    Node* select(Node* node);
    double simulate(const TetrisState& state, std::mt19937& rng);
};
