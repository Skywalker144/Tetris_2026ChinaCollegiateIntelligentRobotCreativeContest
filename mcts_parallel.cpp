#include "mcts_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

MCTSParallel::MCTSParallel(Tetris& game, Args args, std::optional<unsigned int> seed)
    : game_(game),
      args_(args),
      reward_scale_(std::max(1.0, static_cast<double>(game_.getMaxReward()))),
      base_seed_(seed.has_value() ? *seed : std::random_device{}()) {
    if (args_.num_threads <= 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        args_.num_threads = static_cast<int>(hw == 0 ? 4 : hw);
    }
    args_.num_simulations = std::max(1, args_.num_simulations);
    args_.num_threads = std::max(1, args_.num_threads);
    args_.virtual_loss = std::max(0.0, args_.virtual_loss);
}

bool MCTSParallel::Node::canExpand(Tetris& game) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!untried_initialized) {
        untried_actions = game.getLegalActions(state);
        untried_initialized = true;
    }
    return !untried_actions.empty();
}

MCTSParallel::Node* MCTSParallel::Node::expand(Tetris& game, std::mt19937& rng) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!untried_initialized) {
        untried_actions = game.getLegalActions(state);
        untried_initialized = true;
    }
    if (untried_actions.empty()) {
        return nullptr;
    }

    std::uniform_int_distribution<int> dist(0, static_cast<int>(untried_actions.size()) - 1);
    const int action_idx = dist(rng);
    const Action action = untried_actions[action_idx];
    untried_actions[action_idx] = untried_actions.back();
    untried_actions.pop_back();

    TetrisState next_state = game.getNextState(state, action);
    children.push_back(std::make_unique<Node>(std::move(next_state), action, this));
    return children.back().get();
}

std::vector<MCTSParallel::Node*> MCTSParallel::Node::childrenSnapshot() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<Node*> snapshot;
    snapshot.reserve(children.size());
    for (const std::unique_ptr<Node>& child : children) {
        snapshot.push_back(child.get());
    }
    return snapshot;
}

int MCTSParallel::Node::totalVisits() const {
    std::lock_guard<std::mutex> lock(mutex);
    return n + virtual_n;
}

double MCTSParallel::Node::ucb(double exploration_constant, int parent_total_visits, double virtual_loss) const {
    std::lock_guard<std::mutex> lock(mutex);
    const int denom = n + virtual_n;
    if (denom == 0) {
        return std::numeric_limits<double>::infinity();
    }

    const double q = (v - virtual_loss * static_cast<double>(virtual_n)) / static_cast<double>(denom);
    const int parent_visits = std::max(parent_total_visits, 1);
    const double u = exploration_constant * std::sqrt(std::log(static_cast<double>(parent_visits)) / denom);
    return q + u;
}

void MCTSParallel::Node::addVirtualVisit() {
    std::lock_guard<std::mutex> lock(mutex);
    virtual_n += 1;
}

void MCTSParallel::Node::completeVisit(double value) {
    std::lock_guard<std::mutex> lock(mutex);
    if (virtual_n > 0) {
        virtual_n -= 1;
    }
    v += value;
    n += 1;
}

std::optional<Action> MCTSParallel::search(const TetrisState& state) {
    if (game_.isTerminal(state)) {
        return std::nullopt;
    }

    Node root(game_.cloneState(state));

    const int num_threads = std::min(args_.num_threads, args_.num_simulations);
    const int base_sims = args_.num_simulations / num_threads;
    const int remainder = args_.num_simulations % num_threads;

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        const int sims_for_thread = base_sims + (thread_id < remainder ? 1 : 0);
        workers.emplace_back([this, &root, thread_id, sims_for_thread]() {
            std::seed_seq seq{
                base_seed_,
                static_cast<unsigned int>(thread_id),
                static_cast<unsigned int>(0x9e3779b9u + thread_id)
            };
            std::mt19937 rng(seq);

            for (int i = 0; i < sims_for_thread; ++i) {
                Node* node = &root;
                std::vector<Node*> path;
                path.reserve(32);
                path.push_back(node);
                node->addVirtualVisit();

                while (!game_.isTerminal(node->state)) {
                    if (node->canExpand(game_)) {
                        Node* expanded = node->expand(game_, rng);
                        if (expanded != nullptr) {
                            node = expanded;
                            path.push_back(node);
                            node->addVirtualVisit();
                        }
                        break;
                    }

                    std::vector<Node*> children = node->childrenSnapshot();
                    if (children.empty()) {
                        break;
                    }

                    const int parent_total = node->totalVisits();
                    Node* best_child = nullptr;
                    double best_score = -std::numeric_limits<double>::infinity();

                    for (Node* child : children) {
                        const double score =
                            child->ucb(args_.exploration_constant, parent_total, args_.virtual_loss);
                        if (score > best_score) {
                            best_score = score;
                            best_child = child;
                        }
                    }

                    if (best_child == nullptr) {
                        break;
                    }
                    node = best_child;
                    path.push_back(node);
                    node->addVirtualVisit();
                }

                const double value = simulate(node->state, rng);
                for (Node* path_node : path) {
                    path_node->completeVisit(value);
                }
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    std::vector<Node*> root_children = root.childrenSnapshot();
    if (root_children.empty()) {
        return std::nullopt;
    }

    Node* best_child = nullptr;
    int best_visits = -1;
    for (Node* child : root_children) {
        const int visits = child->totalVisits();
        if (visits > best_visits) {
            best_visits = visits;
            best_child = child;
        }
    }

    if (best_child == nullptr) {
        return std::nullopt;
    }
    return best_child->action_taken;
}

MCTSParallel::Node* MCTSParallel::select(Node* node) {
    std::vector<Node*> children = node->childrenSnapshot();
    if (children.empty()) {
        return nullptr;
    }

    const int parent_total = node->totalVisits();
    Node* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (Node* child : children) {
        const double score = child->ucb(args_.exploration_constant, parent_total, args_.virtual_loss);
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    return best_child;
}

double MCTSParallel::simulate(const TetrisState& state, std::mt19937& rng) {
    TetrisState current_state = game_.cloneState(state);

    while (!game_.isTerminal(current_state)) {
        std::vector<Action> legal_actions = game_.getLegalActions(current_state);
        if (legal_actions.empty()) {
            break;
        }
        std::uniform_int_distribution<int> dist(0, static_cast<int>(legal_actions.size()) - 1);
        const Action action = legal_actions[dist(rng)];
        current_state = game_.getNextState(current_state, action);
    }

    const double reward = game_.getReward(current_state);
    if (args_.normalize_reward) {
        return reward / static_cast<double>(reward_scale_);
    }
    return reward;
}
