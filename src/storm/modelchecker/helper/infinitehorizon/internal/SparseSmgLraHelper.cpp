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
                    // STORM_LOG_DEBUG("Transition Matrix:\n" << _transitionMatrix);
                    std::vector<ValueType> result;
                    std::vector<ValueType>  stateRewardsGetter = std::vector<ValueType>(_transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
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
                    std::vector<ValueType> b = getBVector(stateRewardsGetter, actionRewardsGetter);

                    // If requested, allocate memory for the choices made
                    if (this->_produceScheduler) {
                        if (!this->_producedOptimalChoices.is_initialized()) {
                            _producedOptimalChoices.emplace();
                        }
                        _producedOptimalChoices->resize(_transitionMatrix.getRowGroupCount());
                    }

                    prepareMultiplier(env, rewardModel);
                    performValueIteration(env, rewardModel, b, actionRewardsGetter, result);

                    return result;
                }

                template <typename ValueType>
                std::vector<ValueType> SparseSmgLraHelper<ValueType>::getBVector(std::vector<ValueType> const& stateRewardsGetter, ValueGetter const& actionRewardsGetter) {
                    std::vector<ValueType> b = std::vector<ValueType>(_transitionMatrix.getRowCount());
                    size_t globalChoiceCount = 0;
                    auto rowGroupIndices = _transitionMatrix.getRowGroupIndices();
                    for (size_t state = 0; state < _transitionMatrix.getRowGroupCount(); state++) {
                        size_t rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndices[state];
                        for (size_t choice = 0; choice < rowGroupSize; choice++, globalChoiceCount++)
                        {
                            b[globalChoiceCount] = stateRewardsGetter[state] + actionRewardsGetter(globalChoiceCount);
                        }
                    }
                    return b;
                }

                template <typename ValueType>
                void SparseSmgLraHelper<ValueType>::performValueIteration(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel, std::vector<ValueType> const& b, ValueGetter const& actionValueGetter, std::vector<ValueType>& result, std::vector<uint64_t>* choices, std::vector<ValueType>* choiceValues)
                {
                    std::vector<uint64_t> choicesForStrategies = std::vector<uint64_t>(_transitionMatrix.getRowGroupCount(), 0);
                    auto precision = storm::utility::convertNumber<ValueType>(env.solver().lra().getPrecision());

                    Environment envMinMax = env;
                    envMinMax.solver().lra().setPrecision(precision / 2.0);
                    STORM_LOG_DEBUG(envMinMax.solver().lra().getPrecision());
                    do
                    {
                        size_t iteration_count = 0;
                        // Convergent recommender procedure

                        _multiplier->multiplyAndReduce(env, _optimizationDirection, xNew(), &b, xNew(), &choicesForStrategies, &_statesOfCoalition);

                        if (iteration_count % 5 == 0) { // only every 5th iteration
                        storm::storage::BitVector fixedMaxStrat = getStrategyFixedBitVec(choicesForStrategies, MinMaxStrategy::MaxStrategy);
                        storm::storage::BitVector fixedMinStrat = getStrategyFixedBitVec(choicesForStrategies, MinMaxStrategy::MinStrategy);

                        // compute bounds
                            if (fixedMaxStrat != _fixedMaxStrat) {
                                storm::storage::SparseMatrix<ValueType> restrictedMaxMatrix = _transitionMatrix.restrictRows(fixedMaxStrat);

                                storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> MaxSolver(restrictedMaxMatrix);
                                MaxSolver.setOptimizationDirection(OptimizationDirection::Minimize);
                                _resultForMax = MaxSolver.computeLongRunAverageRewards(envMinMax, rewardModel);
                                STORM_LOG_DEBUG("resultMax: " << _resultForMax);
                                _fixedMaxStrat = fixedMaxStrat;

                                for (size_t i = 0; i < xNewL().size(); i++) {
                                    xNewL()[i] = std::max(xNewL()[i], _resultForMax[i]);
                                }
                            }

                            if (fixedMinStrat != _fixedMinStrat) {
                                storm::storage::SparseMatrix<ValueType> restrictedMinMatrix = _transitionMatrix.restrictRows(fixedMinStrat);

                                storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> MinSolver(restrictedMinMatrix);
                                MinSolver.setOptimizationDirection(OptimizationDirection::Maximize);
                                _resultForMin = MinSolver.computeLongRunAverageRewards(envMinMax, rewardModel);
                                STORM_LOG_DEBUG("resultMin: " << _resultForMin);
                                _fixedMinStrat = fixedMinStrat;

                                for (size_t i = 0; i < xNewU().size(); i++) {
                                    xNewU()[i] = std::min(xNewU()[i], _resultForMin[i]);
                                }
                            }
                        }

                        STORM_LOG_DEBUG("xL " << xNewL());
                        STORM_LOG_DEBUG("xU " << xNewU());

                    } while (!checkConvergence(precision));

                    if (_produceScheduler) {
                        _multiplier->multiplyAndReduce(env, _optimizationDirection, xNew(), &b, xNew(), &_producedOptimalChoices.get(), &_statesOfCoalition);
                    }

                    if (_produceChoiceValues) {
                        if (!this->_choiceValues.is_initialized()) {
                            this->_choiceValues.emplace();
                        }
                        this->_choiceValues->resize(this->_transitionMatrix.getRowCount());
                        _choiceValues = calcChoiceValues(envMinMax, rewardModel);
                    }
                    result = xNewL();
                }


                template <typename ValueType>
                storm::storage::BitVector SparseSmgLraHelper<ValueType>::getStrategyFixedBitVec(std::vector<uint64_t> const& choices, MinMaxStrategy strategy) {
                    storm::storage::BitVector restrictBy(_transitionMatrix.getRowCount(), true);
                    auto rowGroupIndices = this->_transitionMatrix.getRowGroupIndices();
                    STORM_LOG_DEBUG("choices " << choices);

                    for(uint state = 0; state < _transitionMatrix.getRowGroupCount(); state++) {
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
                std::vector<ValueType> SparseSmgLraHelper<ValueType>::calcChoiceValues(Environment const& env, storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel) {
                    std::vector<ValueType> choiceValues(_transitionMatrix.getRowCount());

                    storm::storage::SparseMatrix<ValueType> restrictedMaxMatrix = _transitionMatrix.restrictRows(_fixedMaxStrat);
                    storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> MaxSolver(restrictedMaxMatrix);
                    MaxSolver.setOptimizationDirection(OptimizationDirection::Minimize);
                    MaxSolver.setProduceChoiceValues(true);
                    MaxSolver.computeLongRunAverageRewards(env, rewardModel);
                    std::vector<ValueType> minimizerChoices = MaxSolver.getChoiceValues();

                    storm::storage::SparseMatrix<ValueType> restrictedMinMatrix = _transitionMatrix.restrictRows(_fixedMinStrat);
                    storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> MinSolver(restrictedMinMatrix);
                    MinSolver.setOptimizationDirection(OptimizationDirection::Maximize);
                    MinSolver.setProduceChoiceValues(true);
                    MinSolver.computeLongRunAverageRewards(env, rewardModel);
                    std::vector<ValueType> maximizerChoices = MinSolver.getChoiceValues();

                    auto rowGroupIndices = this->_transitionMatrix.getRowGroupIndices();

                    auto minIt = minimizerChoices.begin();
                    auto maxIt = maximizerChoices.begin();
                    size_t globalCounter = 0;
                    for(uint state = 0; state < _transitionMatrix.getRowGroupCount(); state++) {
                        uint rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndices[state];
                        for(uint rowGroupIndex = 0; rowGroupIndex < rowGroupSize; rowGroupIndex++) {
                            if (_minimizerStates[state]) {
                                choiceValues[globalCounter] = *minIt;
                                minIt++;
                            }
                            else {
                                choiceValues[globalCounter] = *maxIt;
                                maxIt++;
                            }
                            globalCounter++;
                        }
                        if (_minimizerStates[state]) {
                            maxIt++;
                        }
                        else {
                            minIt++;
                        }
                    }
                    return choiceValues;
                }

                template <typename ValueType>
                std::vector<ValueType> SparseSmgLraHelper<ValueType>::getChoiceValues() const {
                    STORM_LOG_ASSERT(_produceChoiceValues, "Trying to get the computed choice values although this was not requested.");
                    STORM_LOG_ASSERT(this->_choiceValues.is_initialized(), "Trying to get the computed choice values but none were available. Was there a computation call before?");
                    return this->_choiceValues.get();
                }

                template <typename ValueType>
                storm::storage::Scheduler<ValueType> SparseSmgLraHelper<ValueType>::extractScheduler() const{
                    auto const& optimalChoices = getProducedOptimalChoices();
                    storm::storage::Scheduler<ValueType> scheduler(optimalChoices.size());
                    for (uint64_t state = 0; state < optimalChoices.size(); ++state) {
                        scheduler.setChoice(optimalChoices[state], state);
                    }
                    return scheduler;
                }

                template <typename ValueType>
                std::vector<uint64_t> const& SparseSmgLraHelper<ValueType>::getProducedOptimalChoices() const {
                    STORM_LOG_ASSERT(_produceScheduler, "Trying to get the produced optimal choices although no scheduler was requested.");
                    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(), "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
                    return this->_producedOptimalChoices.get();
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

                    _fixedMaxStrat = storm::storage::BitVector(_transitionMatrix.getRowCount(), false);
                    _fixedMinStrat = storm::storage::BitVector(_transitionMatrix.getRowCount(), false);

                    _resultForMin = std::vector<ValueType>(_transitionMatrix.getRowGroupCount());
                    _resultForMax = std::vector<ValueType>(_transitionMatrix.getRowGroupCount());

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
                    for (; x1It != x1Ite; x1It++, x2It++) {
                        ValueType diff = (*x2It - *x1It);
                        if (diff > threshold) {
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

