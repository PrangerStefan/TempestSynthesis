#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <boost/optional.hpp>

#include "storm/models/ModelType.h"
#include "storm/models/sparse/StateLabeling.h"
#include "storm/models/sparse/ChoiceLabeling.h"
#include "storm/storage/sparse/StateType.h"
#include "storm/storage/sparse/StateValuations.h"
#include "storm/storage/sparse/ChoiceOrigins.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/BitVector.h"
#include "storm/models/sparse/StandardRewardModel.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/InvalidOperationException.h"

namespace storm {
    namespace storage {
        namespace sparse {

            template<typename ValueType, typename RewardModelType = storm::models::sparse::StandardRewardModel<ValueType>>
            struct ModelComponents {

                ModelComponents(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                storm::models::sparse::StateLabeling const& stateLabeling = storm::models::sparse::StateLabeling(),
                                std::unordered_map<std::string, RewardModelType> const& rewardModels =  std::unordered_map<std::string, RewardModelType>(),
                                bool rateTransitions = false,
                                boost::optional<storm::storage::BitVector> const& markovianStates = boost::none,
                                boost::optional<storm::storage::SparseMatrix<storm::storage::sparse::state_type>> const& player1Matrix = boost::none,
                                boost::optional<std::vector<std::pair<std::string, uint_fast32_t>>> const& playerActionIndices = boost::none)
                        : transitionMatrix(transitionMatrix), stateLabeling(stateLabeling), rewardModels(rewardModels), rateTransitions(rateTransitions), markovianStates(markovianStates), player1Matrix(player1Matrix), playerActionIndices(playerActionIndices) {
                    // Intentionally left empty
                }

                ModelComponents(storm::storage::SparseMatrix<ValueType>&& transitionMatrix = storm::storage::SparseMatrix<ValueType>(),
                                storm::models::sparse::StateLabeling&& stateLabeling = storm::models::sparse::StateLabeling(),
                                std::unordered_map<std::string, RewardModelType>&& rewardModels =  std::unordered_map<std::string, RewardModelType>(),
                                bool rateTransitions = false,
                                boost::optional<storm::storage::BitVector>&& markovianStates = boost::none,
                                boost::optional<storm::storage::SparseMatrix<storm::storage::sparse::state_type>>&& player1Matrix = boost::none,
                                boost::optional<std::vector<std::pair<std::string, uint_fast32_t>>>&& playerActionIndices = boost::none)
                        : transitionMatrix(std::move(transitionMatrix)), stateLabeling(std::move(stateLabeling)), rewardModels(std::move(rewardModels)), rateTransitions(rateTransitions), markovianStates(std::move(markovianStates)), player1Matrix(std::move(player1Matrix)), playerActionIndices(std::move(playerActionIndices)) {
                    // Intentionally left empty
                }




                // General components (applicable for all model types):

                // The transition matrix.
                storm::storage::SparseMatrix<ValueType> transitionMatrix;
                // The state labeling.
                storm::models::sparse::StateLabeling stateLabeling;
                // The reward models associated with the model.
                std::unordered_map<std::string, RewardModelType> rewardModels;
                // A vector that stores a labeling for each choice.
                boost::optional<storm::models::sparse::ChoiceLabeling> choiceLabeling;
                // stores for each state to which variable valuation it belongs
                boost::optional<storm::storage::sparse::StateValuations> stateValuations;
                // stores for each choice from which parts of the input model description it originates
                boost::optional<std::shared_ptr<storm::storage::sparse::ChoiceOrigins>> choiceOrigins;

                // POMDP specific components
                // The POMDP observations
                boost::optional<std::vector<uint32_t>> observabilityClasses;

                boost::optional<storm::storage::sparse::StateValuations> observationValuations;

                // Continuous time specific components (CTMCs, Markov Automata):
                // True iff the transition values (for Markovian choices) are interpreted as rates.
                bool rateTransitions;
                // The exit rate for each state. Must be given for CTMCs and MAs, if rateTransitions is false. Otherwise, it is optional.
                boost::optional<std::vector<ValueType>> exitRates;
                // A vector that stores which states are markovian (only for Markov Automata).
                boost::optional<storm::storage::BitVector> markovianStates;

                // Stochastic two player game specific components:
                // The matrix of player 1 choices (needed for stochastic two player games
                boost::optional<storm::storage::SparseMatrix<storm::storage::sparse::state_type>> player1Matrix;

                // Stochastic multiplayer game specific components:
                // The vector mapping state choices to players
                boost::optional<std::vector<std::pair<std::string, uint_fast32_t>>> playerActionIndices;
            };
        }
    }
}
