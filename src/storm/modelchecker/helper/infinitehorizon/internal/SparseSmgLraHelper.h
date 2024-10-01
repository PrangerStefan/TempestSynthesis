#pragma once

#include "storm/storage/SparseMatrix.h"
#include "storm/storage/BitVector.h"
#include "storm/solver/LinearEquationSolver.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/solver/Multiplier.h"
#include "storm/logic/ComparisonType.h"


namespace storm {
    class Environment;


    namespace modelchecker {
        namespace helper {
            namespace internal {

                enum class MinMaxStrategy {
                MaxStrategy,
                MinStrategy
            };

                template <typename ValueType>
                class SparseSmgLraHelper {
                public:
                    /// Function mapping from indices to values
                    typedef std::function<ValueType(uint64_t)> ValueGetter;

                    SparseSmgLraHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const statesOfCoalition);

                    /*!
                     * Performs value iteration with the given state- and action values.
                     * @param env The environment, containing information on the precision of this computation.
                     * @param stateValueGetter function that returns for each state index (w.r.t. the input transition matrix) the reward for staying in state. Will only be called for timed states.
                     * @param actionValueGetter function that returns for each global choice index (w.r.t. the input transition matrix) the reward for taking that choice
                     * @param exitRates (as in the constructor)
                     * @param dir Optimization direction. Must be not nullptr in case of nondeterminism
                     * @param choices if not nullptr, the optimal choices will be inserted in this vector. The vector's size must then be equal to the number of row groups of the input transition matrix.
                     * @return The (optimal) long run average value of the specified component.
                     * @note it is possible to call this method multiple times with different values. However, other changes to the environment or the optimization direction might not have the expected effect due to caching.
                     */
                    void performValueIteration(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel, std::vector<ValueType> const& stateValueGetter, ValueGetter const& actionValueGetter, std::vector<ValueType>& result, std::vector<uint64_t>* choices = nullptr, std::vector<ValueType>* choiceValues = nullptr);



                    void prepareMultiplier(const Environment& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel);

                    std::vector<ValueType> computeLongRunAverageRewardsSound(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel);

                    void setRelevantStates(storm::storage::BitVector relevantStates);

                    void setValueThreshold(storm::logic::ComparisonType const& comparisonType, ValueType const& thresholdValue);

                    void setOptimizationDirection(storm::solver::OptimizationDirection const& direction);

                    void setProduceScheduler(bool value);

                    void setProduceChoiceValues(bool value);

                    void setQualitative(bool value);

                private:

                    /*!
                     * Initializes the value iterations with the provided values.
                     * Resets all information from potential previous calls.
                     * Must be called before the first call to performIterationStep.
                     * @param stateValueGetter Function that returns for each state index (w.r.t. the input transitions) the value (e.g. reward) for that state
                     * @param stateValueGetter Function that returns for each global choice index (w.r.t. the input transitions) the value (e.g. reward) for that choice
                     */
                    void initializeNewValues(ValueGetter const& stateValueGetter, ValueGetter const& actionValueGetter, std::vector<ValueType> const* exitRates = nullptr);

                    bool checkConvergence(ValueType threshold) const;


                    /*!
                     * Performs a single iteration step.
                     * @param env The environment.
                     * @param dir The optimization direction. Has to be given if there is nondeterminism (otherwise it will be ignored)
                     * @param choices If given, the optimal choices will be inserted at the appropriate states.
                     *                Note that these choices will be inserted w.r.t. the original model states/choices, i.e. the size of the vector should match the state-count of the input model
                     * @pre when calling this the first time, initializeNewValues must have been called before. Moreover, prepareNextIteration must be called between two calls of this.
                    */
                    void performIterationStep(Environment const& env, storm::solver::OptimizationDirection const* dir = nullptr, std::vector<uint64_t>* choices = nullptr, std::vector<ValueType>* choiceValues = nullptr);

                    struct ConvergenceCheckResult {
                        bool isPrecisionAchieved;
                        ValueType currentValue;
                    };

                    storm::storage::BitVector getStrategyFixedBitVec(std::vector<uint64_t> const& choices, MinMaxStrategy strategy);

                    /*!
                     * Must be called between two calls of performIterationStep.
                     */
                    void prepareNextIteration(Environment const& env);

                    /// Prepares the necessary solvers and multipliers for doing the iterations.
                    void prepareSolversAndMultipliers(Environment const& env, storm::solver::OptimizationDirection const* dir = nullptr);

                    void setInputModelChoices(std::vector<uint64_t>& choices, std::vector<uint64_t> const& localMecChoices, bool setChoiceZeroToMarkovianStates = false, bool setChoiceZeroToProbabilisticStates = false) const;

                    void setInputModelChoiceValues(std::vector<ValueType>& choiceValues, std::vector<ValueType> const& localMecChoiceValues) const;

                    /// Returns true iff the given state is a timed state
                    bool isTimedState(uint64_t const& inputModelStateIndex) const;

                    std::vector<ValueType>& xNew();
                    std::vector<ValueType> const& xNew() const;

                    std::vector<ValueType>& xOld();
                    std::vector<ValueType> const& xOld() const;

                    std::vector<ValueType>& xNewL();
                    std::vector<ValueType> const& xNewL() const;

                    std::vector<ValueType>& xOldL();
                    std::vector<ValueType> const& xOldL() const;

                    std::vector<ValueType>& xNewU();
                    std::vector<ValueType> const& xNewU() const;

                    std::vector<ValueType>& xOldU();
                    std::vector<ValueType> const& xOldU() const;

                    storm::storage::SparseMatrix<ValueType> const& _transitionMatrix;
                    storm::storage::BitVector const _statesOfCoalition;
                    ValueType _strategyVIPrecision;


                    storm::storage::BitVector _relevantStates;
                    storm::storage::BitVector _minimizerStates;
                    boost::optional<std::pair<storm::logic::ComparisonType, ValueType>> _valueThreshold;
                    storm::solver::OptimizationDirection _optimizationDirection;
                    bool _produceScheduler;
                    bool _produceChoiceValues;
                    bool _isQualitativeSet;

                    ValueType _uniformizationRate;
                    std::vector<ValueType> _x1, _x2, _x1L, _x2L, _x1U, _x2U;
                    std::vector<ValueType> _Tsx1, _Tsx2, _TsChoiceValues;
                    bool _x1IsCurrent;
                    bool _x1IsCurrentStrategyVI;
                    std::vector<ValueType> _Isx, _Isb, _IsChoiceValues;
                    std::unique_ptr<storm::solver::Multiplier<ValueType>> _multiplier;
                    std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> _Solver;
                    std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> _DetIsSolver;
                    std::unique_ptr<storm::Environment> _IsSolverEnv;
                };
            }
        }
    }
}
