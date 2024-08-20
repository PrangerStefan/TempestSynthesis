#pragma once

#include "storm/storage/SparseMatrix.h"
#include "storm/solver/LinearEquationSolver.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/solver/Multiplier.h"
#include "storm/storage/MaximalEndComponentDecomposition.h"

namespace storm {
    class Environment;

    namespace storage {
        template <typename VT> class Scheduler;
    }

    namespace modelchecker {
        namespace helper {
            namespace internal {

                template <typename ValueType>
                class SoundGameViHelper {
                public:
                    SoundGameViHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector statesOfCoalition, storm::storage::BitVector psiStates, OptimizationDirection const& optimizationDirection);

                    void prepareSolversAndMultipliers(const Environment& env);

                    /*!
                     * Perform value iteration until convergence
                     */
                    void performValueIteration(Environment const& env, std::vector<ValueType>& xL, std::vector<ValueType>& xU, storm::solver::OptimizationDirection const dir, std::vector<ValueType>& constrainedChoiceValues);

                    /*!
                     * Sets whether an optimal scheduler shall be constructed during the computation
                     */
                    void setProduceScheduler(bool value);

                    /*!
                     * @return whether an optimal scheduler shall be constructed during the computation
                     */
                    bool isProduceSchedulerSet() const;

                    /*!
                     * Sets whether an optimal scheduler shall be constructed during the computation
                     */
                    void setShieldingTask(bool value);

                    /*!
                     * @return whether an optimal scheduler shall be constructed during the computation
                     */
                    bool isShieldingTask() const;

                    /*!
                     * Changes the transitionMatrix to the given one.
                     */
                    void updateTransitionMatrix(storm::storage::SparseMatrix<ValueType> newTransitionMatrix);

                    /*!
                     * Changes the statesOfCoalition to the given one.
                     */
                    void updateStatesOfCoalition(storm::storage::BitVector newStatesOfCoalition);

                    storm::storage::Scheduler<ValueType> extractScheduler() const;

                    void getChoiceValues(Environment const& env, std::vector<ValueType> const& x, std::vector<ValueType>& choiceValues);

                    /*!
                     * Fills the choice values vector to the original size with zeros for ~psiState choices.
                     */
                    void fillChoiceValuesVector(std::vector<ValueType>& choiceValues, storm::storage::BitVector psiStates, std::vector<storm::storage::SparseMatrix<double>::index_type> rowGroupIndices);

                    void deflate(storm::storage::MaximalEndComponentDecomposition<ValueType> const MECD, storage::SparseMatrix<ValueType> const restrictedMatrix, std::vector<ValueType>& xU, std::vector<ValueType> choiceValues);

                    void reduceChoiceValues(std::vector<ValueType>& choiceValues, storm::storage::BitVector* result, std::vector<ValueType>& x);

                    // multiplier now public for testing
                    std::unique_ptr<storm::solver::Multiplier<ValueType>> _multiplier;
                private:
                    /*!
                     * Performs one iteration step for value iteration
                     */
                    void performIterationStep(Environment const& env, storm::solver::OptimizationDirection const dir, std::vector<uint64_t>* choices = nullptr);

                    /*!
                     * Checks whether the curently computed value achieves the desired precision
                     */
                    bool checkConvergence(ValueType precision) const;

                    std::vector<ValueType>& xNewL();
                    std::vector<ValueType> const& xNewL() const;

                    std::vector<ValueType>& xOldL();
                    std::vector<ValueType> const& xOldL() const;

                    std::vector<ValueType>& xNewU();
                    std::vector<ValueType> const& xNewU() const;

                    std::vector<ValueType>& xOldU();
                    std::vector<ValueType> const& xOldU() const;

                    bool _x1IsCurrent;

                    storm::storage::BitVector _minimizerStates;

                    /*!
                     * @pre before calling this, a computation call should have been performed during which scheduler production was enabled.
                     * @return the produced scheduler of the most recent call.
                     */
                    std::vector<uint64_t> const& getProducedOptimalChoices() const;

                    /*!
                     * @pre before calling this, a computation call should have been performed during which scheduler production was enabled.
                     * @return the produced scheduler of the most recent call.
                     */
                    std::vector<uint64_t>& getProducedOptimalChoices();

                    storm::storage::SparseMatrix<ValueType> _transitionMatrix;
                    storm::storage::SparseMatrix<ValueType> _backwardTransitions;
                    storm::storage::BitVector _statesOfCoalition;
                    storm::storage::BitVector _psiStates;
                    std::vector<ValueType> _x, _x1L, _x2L, _x1U, _x2U;
                    OptimizationDirection _optimizationDirection;

                    bool _produceScheduler = false;
                    bool _shieldingTask = false;
                    boost::optional<std::vector<uint64_t>> _producedOptimalChoices;
                };
            }
        }
    }
}
