#include "SoundGameViHelper.h"

#include "storm/environment/Environment.h"
#include "storm/environment/solver/SolverEnvironment.h"
#include "storm/environment/solver/GameSolverEnvironment.h"


#include "storm/utility/SignalHandler.h"
#include "storm/utility/vector.h"

namespace storm {
    namespace modelchecker {
        namespace helper {
            namespace internal {

                template <typename ValueType>
                SoundGameViHelper<ValueType>::SoundGameViHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector statesOfCoalition, storm::storage::BitVector psiStates, OptimizationDirection const& optimizationDirection) : _transitionMatrix(transitionMatrix), _backwardTransitions(backwardTransitions), _statesOfCoalition(statesOfCoalition), _psiStates(psiStates), _optimizationDirection(optimizationDirection) {
                    // Intentionally left empty.
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::prepareSolversAndMultipliers(const Environment& env) {
                    STORM_LOG_DEBUG("\n" << _transitionMatrix);
                    _multiplier = storm::solver::MultiplierFactory<ValueType>().create(env, _transitionMatrix);
                    _x1IsCurrent = false;
                    _minimizerStates = _optimizationDirection == OptimizationDirection::Maximize ? _statesOfCoalition : ~_statesOfCoalition;
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::performValueIteration(Environment const& env, std::vector<ValueType>& xL, std::vector<ValueType>& xU, storm::solver::OptimizationDirection const dir, std::vector<ValueType>& constrainedChoiceValues) {
                    // new pair (x_old, x_new) for over_approximation()

                    prepareSolversAndMultipliers(env);
                    // Get precision for convergence check.
                    ValueType precision = storm::utility::convertNumber<ValueType>(env.solver().game().getPrecision());

                    uint64_t maxIter = env.solver().game().getMaximalNumberOfIterations();
                    //_x1.assign(_transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                    _x1L = xL;
                    _x2L = _x1L;

                    _x1U = xU;
                    _x2U = _x1U;

                    if (this->isProduceSchedulerSet()) {
                        if (!this->_producedOptimalChoices.is_initialized()) {
                            this->_producedOptimalChoices.emplace();
                        }
                        this->_producedOptimalChoices->resize(this->_transitionMatrix.getRowGroupCount());
                    }

                    uint64_t iter = 0;
                    constrainedChoiceValues = std::vector<ValueType>(xL.size(), storm::utility::zero<ValueType>()); // ??

                    while (iter < maxIter) {
                        performIterationStep(env, dir);
                        if (checkConvergence(precision)) {
                            //_multiplier->multiply(env, xNewL(), nullptr, constrainedChoiceValues); // TODO Fabian: ???
                            break;
                        }
                        if (storm::utility::resources::isTerminate()) {
                            break;
                        }
                        ++iter;
                    }
                    xL = xNewL();
                    xU = xNewU();

                    STORM_LOG_DEBUG("result xL: " << xL);
                    STORM_LOG_DEBUG("result xU: " << xU);

                    if (isProduceSchedulerSet()) {
                        // We will be doing one more iteration step and track scheduler choices this time.
                        performIterationStep(env, dir, &_producedOptimalChoices.get());
                    }
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::performIterationStep(Environment const& env, storm::solver::OptimizationDirection const dir, std::vector<uint64_t>* choices) {
                    storm::storage::BitVector reducedMinimizerActions = {storm::storage::BitVector(this->_transitionMatrix.getRowCount(), true)};

                    // under approximation

                    if (!_multiplier) {
                        prepareSolversAndMultipliers(env);
                    }
                    _x1IsCurrent = !_x1IsCurrent;

                    std::vector<ValueType> choiceValues = xNewL();
                    choiceValues.resize(this->_transitionMatrix.getRowCount());

                    _multiplier->multiply(env, xOldL(), nullptr, choiceValues);
                    reduceChoiceValues(choiceValues, &reducedMinimizerActions);
                    xNewL() = choiceValues;

                    // over_approximation

                    _multiplier->multiplyAndReduce(env, dir, xOldU(), nullptr, xNewU(), nullptr, &_statesOfCoalition);

                    // restricting the none optimal minimizer choices
                    storage::SparseMatrix<ValueType> restrictedTransMatrix = this->_transitionMatrix.restrictRows(reducedMinimizerActions);
                    _multiplierRestricted = storm::solver::MultiplierFactory<ValueType>().create(env, restrictedTransMatrix);

                    // STORM_LOG_DEBUG("restricted Transition: \n" << restrictedTransMatrix);

                    // find_MSECs() & deflate()
                    storm::storage::MaximalEndComponentDecomposition<ValueType> MSEC = storm::storage::MaximalEndComponentDecomposition<ValueType>(restrictedTransMatrix, _backwardTransitions);

                    // STORM_LOG_DEBUG("MECD: \n" << MECD);
                    deflate(MSEC,restrictedTransMatrix, xNewU());
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::deflate(storm::storage::MaximalEndComponentDecomposition<ValueType> const MSEC, storage::SparseMatrix<ValueType> const restrictedMatrix,  std::vector<ValueType>& xU) {
                    auto rowGroupIndices = restrictedMatrix.getRowGroupIndices();

                    // iterating over all MSECs
                    for (auto smec_it : MSEC) {
                        ValueType bestExit = 0;
                        auto stateSet = smec_it.getStateSet();
                        for (uint state : stateSet) {
                            uint rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndices[state];
                            if (!_minimizerStates[state]) {                                           // check if current state is maximizer state
                                for (uint choice = 0; choice < rowGroupSize; choice++) {
                                    if (!smec_it.containsChoice(state, choice + rowGroupIndices[state])) {
                                        ValueType choiceValue = 0;
                                        _multiplierRestricted->multiplyRow(choice + rowGroupIndices[state], xU, choiceValue);
                                        if (choiceValue > bestExit)
                                            bestExit = choiceValue;
                                    }
                                }
                            }
                        }
                        // deflating the states of the current MSEC
                        for (uint state : stateSet) {
                            if (!_psiStates[state])
                                xU[state] = std::min(xU[state], bestExit);
                        }
                    }
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::reduceChoiceValues(std::vector<ValueType>& choiceValues, storm::storage::BitVector* result)
                {
                    // result BitVec should be initialized with 1s outside the method

                    auto rowGroupIndices = this->_transitionMatrix.getRowGroupIndices();
                    auto choice_it = choiceValues.begin();

                    for(uint state = 0; state < rowGroupIndices.size() - 1; state++) {
                        uint rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndices[state];
                        ValueType optChoice;
                        if (_minimizerStates[state]) {  // check if current state is minimizer state
                            // getting the optimal minimizer choice for the given state
                            optChoice = *std::min_element(choice_it, choice_it + rowGroupSize);

                            for (uint choice = 0; choice < rowGroupSize; choice++, choice_it++) {
                                if (*choice_it > optChoice) {
                                    result->set(rowGroupIndices[state] + choice, 0);
                                }
                            }
                            // reducing the xNew() (choiceValues) vector for minimizer states
                            choiceValues[state] = optChoice;
                        }
                        else
                        {
                            optChoice = *std::max_element(choice_it, choice_it + rowGroupSize);
                            // reducing the xNew() (choiceValues) vector for maximizer states
                            choiceValues[state] = optChoice;
                            choice_it += rowGroupSize;
                        }
                    }
                    choiceValues.resize(this->_transitionMatrix.getRowGroupCount());
                }


                template <typename ValueType>
                bool SoundGameViHelper<ValueType>::checkConvergence(ValueType threshold) const {
                    STORM_LOG_ASSERT(_multiplier, "tried to check for convergence without doing an iteration first.");
                    // Now check whether the currently produced results are precise enough
                    STORM_LOG_ASSERT(threshold > storm::utility::zero<ValueType>(), "Did not expect a non-positive threshold.");
                    auto x1It = xNewL().begin();
                    auto x1Ite = xNewL().end();
                    auto x2It = xNewU().begin();
                    ValueType maxDiff = (*x2It - *x1It);
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
                void SoundGameViHelper<ValueType>::setProduceScheduler(bool value) {
                    _produceScheduler = value;
                }


                template <typename ValueType>
                bool SoundGameViHelper<ValueType>::isProduceSchedulerSet() const {
                    return _produceScheduler;
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::setShieldingTask(bool value) {
                    _shieldingTask = value;
                }

                template <typename ValueType>
                bool SoundGameViHelper<ValueType>::isShieldingTask() const {
                    return _shieldingTask;
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::updateTransitionMatrix(storm::storage::SparseMatrix<ValueType> newTransitionMatrix) {
                    _transitionMatrix = newTransitionMatrix;
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::updateStatesOfCoalition(storm::storage::BitVector newStatesOfCoalition) {
                    _statesOfCoalition = newStatesOfCoalition;
                }

                template <typename ValueType>
                std::vector<uint64_t> const& SoundGameViHelper<ValueType>::getProducedOptimalChoices() const {
                    STORM_LOG_ASSERT(this->isProduceSchedulerSet(), "Trying to get the produced optimal choices although no scheduler was requested.");
                    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(), "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
                    return this->_producedOptimalChoices.get();
                }

                template <typename ValueType>
                std::vector<uint64_t>& SoundGameViHelper<ValueType>::getProducedOptimalChoices() {
                    STORM_LOG_ASSERT(this->isProduceSchedulerSet(), "Trying to get the produced optimal choices although no scheduler was requested.");
                    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(), "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
                    return this->_producedOptimalChoices.get();
                }

                template <typename ValueType>
                storm::storage::Scheduler<ValueType> SoundGameViHelper<ValueType>::extractScheduler() const{
                    auto const& optimalChoices = getProducedOptimalChoices();
                    storm::storage::Scheduler<ValueType> scheduler(optimalChoices.size());
                    for (uint64_t state = 0; state < optimalChoices.size(); ++state) {
                        scheduler.setChoice(optimalChoices[state], state);
                    }
                    return scheduler;
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::getChoiceValues(Environment const& env, std::vector<ValueType> const& x, std::vector<ValueType>& choiceValues) {
                    _multiplier->multiply(env, x, nullptr, choiceValues);
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::fillChoiceValuesVector(std::vector<ValueType>& choiceValues, storm::storage::BitVector psiStates, std::vector<storm::storage::SparseMatrix<double>::index_type> rowGroupIndices) {
                    std::vector<ValueType> allChoices = std::vector<ValueType>(rowGroupIndices.at(rowGroupIndices.size() - 1), storm::utility::zero<ValueType>());
                    auto choice_it = choiceValues.begin();
                    for(uint state = 0; state < rowGroupIndices.size() - 1; state++) {
                        uint rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndices[state];
                        if (psiStates.get(state)) {
                            for(uint choice = 0; choice < rowGroupSize; choice++, choice_it++) {
                                allChoices.at(rowGroupIndices.at(state) + choice) = *choice_it;
                            }
                        }
                    }
                    choiceValues = allChoices;
                }

                template <typename ValueType>
                std::vector<ValueType>& SoundGameViHelper<ValueType>::xNewL() {
                    return _x1IsCurrent ? _x1L : _x2L;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SoundGameViHelper<ValueType>::xNewL() const {
                    return _x1IsCurrent ? _x1L : _x2L;
                }

                template <typename ValueType>
                std::vector<ValueType>& SoundGameViHelper<ValueType>::xOldL() {
                    return _x1IsCurrent ? _x2L : _x1L;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SoundGameViHelper<ValueType>::xOldL() const {
                    return _x1IsCurrent ? _x2L : _x1L;
                }

                template <typename ValueType>
                std::vector<ValueType>& SoundGameViHelper<ValueType>::xNewU() {
                    return _x1IsCurrent ? _x1U : _x2U;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SoundGameViHelper<ValueType>::xNewU() const {
                    return _x1IsCurrent ? _x1U : _x2U;
                }

                template <typename ValueType>
                std::vector<ValueType>& SoundGameViHelper<ValueType>::xOldU() {
                    return _x1IsCurrent ? _x2U : _x1U;
                }

                template <typename ValueType>
                std::vector<ValueType> const& SoundGameViHelper<ValueType>::xOldU() const {
                    return _x1IsCurrent ? _x2U : _x1U;
                }

                template class SoundGameViHelper<double>;
#ifdef STORM_HAVE_CARL
                template class SoundGameViHelper<storm::RationalNumber>;
#endif
            }
        }
    }
}
