#include "SparseSmgLraHelper.h"

#include "storm/storage/MaximalEndComponent.h"
#include "storm/storage/StronglyConnectedComponent.h"

#include "storm/utility/graph.h"
#include "storm/utility/vector.h"
#include "storm/utility/macros.h"
#include "storm/utility/SignalHandler.h"

#include "storm/environment/solver/SolverEnvironment.h"
#include "storm/environment/solver/LongRunAverageSolverEnvironment.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include "storm/environment/solver/MultiplierEnvironment.h"
#include "storm/environment/solver/GameSolverEnvironment.h"

#include "modelchecker/helper/infinitehorizon/SparseNondeterministicInfiniteHorizonHelper.h"
#include "storm/exceptions/UnmetRequirementException.h"

namespace storm {
    namespace modelchecker {
        namespace helper {
            namespace internal {

                template <typename ValueType>
                SparseSmgLraHelper<ValueType>::SparseSmgLraHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector const statesOfCoalition) : _transitionMatrix(transitionMatrix), _x1IsCurrent(false), _x1IsCurrentStrategyVI(false), _statesOfCoalition(statesOfCoalition) {

                }

                template <typename ValueType>
                std::vector<ValueType> SparseSmgLraHelper<ValueType>::computeLongRunAverageRewardsSound(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel) {
                    STORM_LOG_DEBUG("Transition Matrix:\n" << _transitionMatrix);
                    std::vector<ValueType> result;
                    std::vector<ValueType>  stateRewardsGetter;
                    if (rewardModel.hasStateRewards()) {
                        stateRewardsGetter = rewardModel.getStateRewardVector();
                    }
                    ValueGetter actionRewardsGetter;
                    if (rewardModel.hasStateActionRewards() || rewardModel.hasTransitionRewards()) {
                        if (rewardModel.hasTransitionRewards()) {
                            actionRewardsGetter = [&] (uint64_t globalChoiceIndex) { return rewardModel.getStateActionAndTransitionReward(globalChoiceIndex, this->_transitionMatrix); };
                        } else {
                            actionRewardsGetter = [&] (uint64_t globalChoiceIndex) { return rewardModel.getStateActionReward(globalChoiceIndex); };
                        }
                    } else {
                        actionRewardsGetter = [] (uint64_t) { return storm::utility::zero<ValueType>(); };
                    }
                    STORM_LOG_DEBUG("rewards: " << rewardModel.getStateRewardVector());
                    prepareMultiplier(env, rewardModel);
                    performValueIteration(env, rewardModel, stateRewardsGetter, actionRewardsGetter, result);

                    return result;
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::performValueIteration(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel, std::vector<ValueType> const& stateValueGetter, ValueGetter const& actionValueGetter, std::vector<ValueType>& result, std::vector<uint64_t>* choices, std::vector<ValueType>* choiceValues)
                {
                    std::vector<uint64_t> choicesForStrategies = std::vector<uint64_t>(_transitionMatrix.getRowGroupCount(), 0);
                    ValueType precision = storm::utility::convertNumber<ValueType>(env.solver().game().getPrecision());

                    do
                    {
                        _x1IsCurrent = !_x1IsCurrent;
                        // Convergent recommender procedure

                        _multiplier->multiplyAndReduce(env, _optimizationDirection, xOld(), nullptr, xNew(), &choicesForStrategies, &_statesOfCoalition);
                        for (size_t i = 0; i < xNew().size(); i++)
                        {
                            xNew()[i] = xNew()[i] + stateValueGetter[i];
                        }

                        storm::storage::BitVector fixedMaxStrat = getStrategyFixedBitVec(choicesForStrategies, MinMaxStrategy::MaxStrategy);
                        storm::storage::BitVector fixedMinStrat = getStrategyFixedBitVec(choicesForStrategies, MinMaxStrategy::MinStrategy);
                        storm::storage::SparseMatrix<ValueType> restrictedMaxMatrix = _transitionMatrix.restrictRows(fixedMaxStrat);

                        storm::storage::SparseMatrix<ValueType> restrictedMinMatrix = _transitionMatrix.restrictRows(fixedMinStrat);

                        // compute bounds
                        storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> MaxSolver(restrictedMaxMatrix);
                        MaxSolver.setOptimizationDirection(OptimizationDirection::Minimize);
                        std::vector<ValueType> resultForMax = MaxSolver.computeLongRunAverageRewards(env, rewardModel);

                        for (size_t i = 0; i < xNewL().size(); i++)
                        {
                            xNewL()[i] = std::max(xOldL()[i], resultForMax[i]);
                        }

                        storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> MinSolver(restrictedMinMatrix);
                        MinSolver.setOptimizationDirection(OptimizationDirection::Maximize);
                        std::vector<ValueType> resultForMin = MinSolver.computeLongRunAverageRewards(env, rewardModel);

                        for (size_t i = 0; i < xNewU().size(); i++)
                        {
                            xNewU()[i] = std::min(xOldU()[i], resultForMin[i]);
                        }

                        STORM_LOG_DEBUG("xL " << xNewL());
                        STORM_LOG_DEBUG("xU " << xNewU());

                    } while (!checkConvergence(precision));
                    result = xNewU();
                }


                template <typename ValueType>
                storm::storage::BitVector SparseSmgLraHelper<ValueType>::getStrategyFixedBitVec(std::vector<uint64_t> const& choices, MinMaxStrategy strategy) {
                    storm::storage::BitVector restrictBy(_transitionMatrix.getRowCount(), true);
                    auto rowGroupIndices = this->_transitionMatrix.getRowGroupIndices();
                    STORM_LOG_DEBUG("choices " << choices);

                    for(uint state = 0; state < rowGroupIndices.size() - 1; state++) {
                        if ((_minimizerStates[state] && strategy == MinMaxStrategy::MaxStrategy) || (!_minimizerStates[state] && strategy == MinMaxStrategy::MinStrategy))
                            continue;

                        uint rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndices[state];
                        for(uint rowGroupIndex = 0; rowGroupIndex < rowGroupSize; rowGroupIndex++) {
                            if ((rowGroupIndex) != choices[state]) {
                                restrictBy.set(rowGroupIndex + rowGroupIndices[state], false);
                            }
                        }
                    }
                    return restrictBy;
                }


                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::prepareMultiplier(const Environment& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel)
                {
                    _multiplier = storm::solver::MultiplierFactory<ValueType>().create(env, _transitionMatrix);
                    _minimizerStates = _optimizationDirection == OptimizationDirection::Maximize ? _statesOfCoalition : ~_statesOfCoalition;

                    _x1L = std::vector<ValueType>(_transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                    _x2L = _x1L;
                    _x1 = _x1L;
                    _x2 = _x1;

                    _x1U = std::vector<ValueType>(_transitionMatrix.getRowGroupCount(), std::numeric_limits<ValueType>::infinity());
                    _x2U = _x1U;
                }

                template <typename ValueType>
                bool SparseSmgLraHelper<ValueType>::checkConvergence(ValueType threshold) const {
                    STORM_LOG_ASSERT(_multiplier, "tried to check for convergence without doing an iteration first.");
                    // Now check whether the currently produced results are precise enough
                    STORM_LOG_ASSERT(threshold > storm::utility::zero<ValueType>(), "Did not expect a non-positive threshold.");
                    auto x1It = xNewL().begin();
                    auto x1Ite = xNewL().end();
                    auto x2It = xNewU().begin();
                    ValueType maxDiff = ((*x2It) - (*x1It));
                    ValueType minDiff = maxDiff;
                    // The difference between maxDiff and minDiff is zero at this point. Thus, it doesn't make sense to check the threshold now.
                    for (++x1It, ++x2It; x1It != x1Ite; ++x1It, ++x2It) {
                        ValueType diff = (*x2It - *x1It);
                        // Potentially update maxDiff or minDiff
                        bool skipCheck = false;
                        if (maxDiff < diff) {
                            maxDiff = diff;
                        } else if (minDiff > diff) {
                            minDiff = diff;
                        } else {
                            skipCheck = true;
                        }
                        // Check convergence
                        if (!skipCheck && (maxDiff - minDiff) > threshold) {
                            return false;
                        }
                    }
                    return true;
                }

                template <typename ValueType>
                std::vector<ValueType>& SparseSmgLraHelper<ValueType>::xNewL() {
                    return _x1IsCurrent ? _x1L : _x2L;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SparseSmgLraHelper<ValueType>::xNewL() const {
                    return _x1IsCurrent ? _x1L : _x2L;
                }

                template <typename ValueType>
                std::vector<ValueType>& SparseSmgLraHelper<ValueType>::xOldL() {
                    return _x1IsCurrent ? _x2L : _x1L;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SparseSmgLraHelper<ValueType>::xOldL() const {
                    return _x1IsCurrent ? _x2L : _x1L;
                }

                template <typename ValueType>
                std::vector<ValueType>& SparseSmgLraHelper<ValueType>::xNewU() {
                    return _x1IsCurrent ? _x1U : _x2U;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SparseSmgLraHelper<ValueType>::xNewU() const {
                    return _x1IsCurrent ? _x1U : _x2U;
                }

                template <typename ValueType>
                std::vector<ValueType>& SparseSmgLraHelper<ValueType>::xOldU() {
                    return _x1IsCurrent ? _x2U : _x1U;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SparseSmgLraHelper<ValueType>::xOldU() const {
                    return _x1IsCurrent ? _x2U : _x1U;
                }

                template <typename ValueType>
                std::vector<ValueType>& SparseSmgLraHelper<ValueType>::xOld() {
                    return _x1IsCurrent ? _x2 : _x1;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SparseSmgLraHelper<ValueType>::xOld() const {
                    return _x1IsCurrent ? _x2 : _x1;
                }

                template <typename ValueType>
                std::vector<ValueType>& SparseSmgLraHelper<ValueType>::xNew() {
                    return _x1IsCurrent ? _x1 : _x2;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SparseSmgLraHelper<ValueType>::xNew() const {
                    return _x1IsCurrent ? _x1 : _x2;
                }


                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::setRelevantStates(storm::storage::BitVector relevantStates){
                    _relevantStates = relevantStates;
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::setValueThreshold(storm::logic::ComparisonType const& comparisonType, const ValueType &thresholdValue) {
                    _valueThreshold = std::make_pair(comparisonType, thresholdValue);
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::setOptimizationDirection(storm::solver::OptimizationDirection const& direction) {
                    _optimizationDirection = direction;
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::setProduceScheduler(bool value) {
                    _produceScheduler = value;
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::setProduceChoiceValues(bool value) {
                    _produceChoiceValues = value;
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::setQualitative(bool value) {
                    _isQualitativeSet = value;
                }

                template class SparseSmgLraHelper<double>;
#ifdef STORM_HAVE_CARL
                template class SparseSmgLraHelper<storm::RationalNumber>;
#endif

            }
        }
    }
}

