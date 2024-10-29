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
                    // Function mapping from indices to values
                    typedef std::function<ValueType(uint64_t)> ValueGetter;

                    SparseSmgLraHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const statesOfCoalition);

                    void performValueIteration(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel, std::vector<ValueType> const& b, std::vector<ValueType>& result);

                    std::vector<ValueType> getChoiceValues() const;

                    storm::storage::Scheduler<ValueType> extractScheduler() const;

                    std::vector<uint64_t> const& getProducedOptimalChoices() const;

                    void prepareMultiplier(const Environment& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel);

                    std::vector<ValueType> computeLongRunAverageRewardsSound(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel);

                    void setRelevantStates(storm::storage::BitVector relevantStates);

                    void setValueThreshold(storm::logic::ComparisonType const& comparisonType, ValueType const& thresholdValue);

                    void setOptimizationDirection(storm::solver::OptimizationDirection const& direction);

                    void setProduceScheduler(bool value);

                    void setProduceChoiceValues(bool value);

                    void setQualitative(bool value);

                private:

                    bool checkConvergence(ValueType threshold) const;

                    storm::storage::BitVector getStrategyFixedBitVec(std::vector<uint64_t> const& choices, MinMaxStrategy strategy);

                    std::vector<ValueType> getBVector(std::vector<ValueType> const& stateRewardsGetter, ValueGetter const& actionRewardsGetter);

                    std::vector<ValueType> calcChoiceValues(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel);

                    std::vector<ValueType>& xNew();
                    std::vector<ValueType> const& xNew() const;

                    std::vector<ValueType>& xNewL();
                    std::vector<ValueType> const& xNewL() const;

                    std::vector<ValueType>& xNewU();
                    std::vector<ValueType> const& xNewU() const;

                    storm::storage::SparseMatrix<ValueType> const& _transitionMatrix;
                    storm::storage::BitVector const _statesOfCoalition;

                    storm::storage::BitVector _relevantStates;
                    storm::storage::BitVector _minimizerStates;

                    storm::storage::BitVector _fixedMinStrat;
                    storm::storage::BitVector _fixedMaxStrat;
                    std::vector<ValueType> _resultForMax;
                    std::vector<ValueType> _resultForMin;

                    std::vector<ValueType> _b;

                    boost::optional<std::pair<storm::logic::ComparisonType, ValueType>> _valueThreshold;
                    storm::solver::OptimizationDirection _optimizationDirection;
                    bool _produceScheduler;
                    bool _produceChoiceValues;
                    bool _isQualitativeSet;

                    std::vector<ValueType> _x, _xL, _xU;
                    std::vector<ValueType> _Tsx1, _Tsx2, _TsChoiceValues;
                    std::vector<ValueType> _Isx, _Isb, _IsChoiceValues;
                    std::unique_ptr<storm::solver::Multiplier<ValueType>> _multiplier;
                    std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> _Solver;
                    std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> _DetIsSolver;
                    std::unique_ptr<storm::Environment> _IsSolverEnv;

                    boost::optional<std::vector<uint64_t>> _producedOptimalChoices;
                    boost::optional<std::vector<ValueType>> _choiceValues;
                };
            }
        }
    }
}
