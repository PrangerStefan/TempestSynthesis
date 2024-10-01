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
                SoundGameViHelper<ValueType>::SoundGameViHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> b, storm::storage::BitVector statesOfCoalition, storm::storage::BitVector psiStates, OptimizationDirection const& optimizationDirection) : _transitionMatrix(transitionMatrix), _backwardTransitions(backwardTransitions), _statesOfCoalition(statesOfCoalition), _psiStates(psiStates), _optimizationDirection(optimizationDirection), _b(b) {
                    // Intentionally left empty.
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::prepareSolversAndMultipliers(const Environment& env) {
                    STORM_LOG_DEBUG("\n" << _transitionMatrix);
                    _multiplier = storm::solver::MultiplierFactory<ValueType>().create(env, _transitionMatrix);
                    _x1IsCurrent = false;
                    _minimizerStates = _optimizationDirection == OptimizationDirection::Maximize ? _statesOfCoalition : ~_statesOfCoalition;
                    _oldPolicy = storm::storage::BitVector(_transitionMatrix.getRowCount(), false);
                    _timing = std::vector<double>(5, 0);
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::performValueIteration(Environment const& env, std::vector<ValueType>& xL, std::vector<ValueType>& xU, storm::solver::OptimizationDirection const dir, std::vector<ValueType>& constrainedChoiceValues) {

                    prepareSolversAndMultipliers(env);
                    // Get precision for convergence check.
                    ValueType precision = storm::utility::convertNumber<ValueType>(env.solver().game().getPrecision());

                    uint64_t maxIter = env.solver().game().getMaximalNumberOfIterations();
                    //_x1.assign(_transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                    _x1L = xL;
                    _x2L = _x1L;
                    _x1Test = _x1L;
                    _x2Test = _x1L;

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

                    // for profiling
                    STORM_PRINT(_timing[0] << ", " << _timing[1] << ", " << _timing[2] << ", " << _timing[3] << ", " << _timing[4] << std::endl);

                    if (isProduceSchedulerSet()) {
                        // We will be doing one more iteration step and track scheduler choices this time.
                        performIterationStep(env, dir, &_producedOptimalChoices.get());
                    }
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::performIterationStep(Environment const& env, storm::solver::OptimizationDirection const dir, std::vector<uint64_t>* choices) {
                    storm::storage::BitVector reducedMinimizerActions = {storm::storage::BitVector(this->_transitionMatrix.getRowCount(), true)};

                    // under approximation
                    auto start = std::chrono::steady_clock::now();
                    if (!_multiplier) {
                        prepareSolversAndMultipliers(env);
                    }
                    _x1IsCurrent = !_x1IsCurrent;
                    std::vector<ValueType> choiceValuesL = std::vector<ValueType>(this->_transitionMatrix.getRowCount(), storm::utility::zero<ValueType>());

                    _multiplier->multiply(env, xOldL(), nullptr, choiceValuesL);
                    reduceChoiceValues(choiceValuesL, &reducedMinimizerActions, xNewL());


                    // just for debugging
                    _multiplier->multiplyAndReduce(env, _optimizationDirection, xOldTest(), nullptr, xNewTest(), nullptr, &_statesOfCoalition);

                    // over_approximation
                    std::vector<ValueType> choiceValuesU = std::vector<ValueType>(this->_transitionMatrix.getRowCount(), storm::utility::zero<ValueType>());

                    _multiplier->multiply(env, xOldU(), nullptr, choiceValuesU);
                    reduceChoiceValues(choiceValuesU, nullptr, xNewU());

                    auto finish = std::chrono::steady_clock::now();
                    double elapsed_seconds = std::chrono::duration_cast<
                                                 std::chrono::duration<double>>(finish - start).count();
                    _timing[0] += elapsed_seconds;

                    if (reducedMinimizerActions != _oldPolicy) { // new MECs only if Policy changed
                        start = std::chrono::steady_clock::now();

                        // restricting the none optimal minimizer choices
                        _restrictedTransitions = this->_transitionMatrix.restrictRows(reducedMinimizerActions);

                        finish = std::chrono::steady_clock::now();
                        elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
                        _timing[1] += elapsed_seconds;

                        // STORM_LOG_DEBUG("restricted Transition: \n" << restrictedTransMatrix);
                        start = std::chrono::steady_clock::now();
                        // find_MSECs()
                        _MSECs = storm::storage::MaximalEndComponentDecomposition<ValueType>(_restrictedTransitions, _restrictedTransitions.transpose(true));

                        finish = std::chrono::steady_clock::now();
                        elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
                        _timing[2] += elapsed_seconds;
                    }

                    start = std::chrono::steady_clock::now();
                    // reducing the choiceValuesU
                    size_t i = 0;
                    auto new_end = std::remove_if(choiceValuesU.begin(), choiceValuesU.end(), [&reducedMinimizerActions, &i](const auto& item) {
                        bool ret = !(reducedMinimizerActions[i]);
                        i++;
                        return ret;
                    });
                    choiceValuesU.erase(new_end, choiceValuesU.end());
                    finish = std::chrono::steady_clock::now();
                    elapsed_seconds = std::chrono::duration_cast<
                                          std::chrono::duration<double>>(finish - start).count();
                    _timing[3] += elapsed_seconds;
                    _oldPolicy = reducedMinimizerActions;

                    // deflating the MSECs
                    start = std::chrono::steady_clock::now();
                    deflate(_MSECs, _restrictedTransitions, xNewU(), choiceValuesU);
                    finish = std::chrono::steady_clock::now();
                    elapsed_seconds = std::chrono::duration_cast<
                                          std::chrono::duration<double>>(finish - start).count();
                    _timing[4] += elapsed_seconds;
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::deflate(storm::storage::MaximalEndComponentDecomposition<ValueType> const MSEC, storage::SparseMatrix<ValueType> const restrictedMatrix,  std::vector<ValueType>& xU, std::vector<ValueType> choiceValues) {
                    auto rowGroupIndices = restrictedMatrix.getRowGroupIndices();
                    auto choice_begin = choiceValues.begin();
                    // iterating over all MSECs
                    for (auto smec_it : MSEC) {
                        ValueType bestExit = 0;
                        if (smec_it.isErgodic(restrictedMatrix)) continue;
                        auto stateSet = smec_it.getStateSet();
                        for (uint state : stateSet) {
                            if (_minimizerStates[state]) continue;
                            uint rowGroupIndex = rowGroupIndices[state];
                            auto exitingCompare = [&state, &smec_it, &choice_begin](const ValueType &lhs, const ValueType &rhs)
                            {
                                bool lhsExiting = !smec_it.containsChoice(state, (&lhs - &(*choice_begin)));
                                bool rhsExiting = !smec_it.containsChoice(state, (&rhs - &(*choice_begin)));
                                if( lhsExiting && !rhsExiting) return false;
                                if(!lhsExiting &&  rhsExiting) return true;
                                if(!lhsExiting && !rhsExiting) return false;
                                return lhs < rhs;
                            };
                            uint rowGroupSize = rowGroupIndices[state + 1] - rowGroupIndex;

                            auto choice_it = choice_begin + rowGroupIndex;
                            ValueType newBestExit = *std::max_element(choice_it, choice_it + rowGroupSize, exitingCompare);
                            if (newBestExit > bestExit)
                                bestExit = newBestExit;
                        }
                        // deflating the states of the current MSEC
                        for (uint state : stateSet) {
                            if (_psiStates[state]) continue;
                            xU[state] = std::min(xU[state], bestExit);
                        }
                    }
                }

                template <typename ValueType>
                void SoundGameViHelper<ValueType>::reduceChoiceValues(std::vector<ValueType>& choiceValues, storm::storage::BitVector* result, std::vector<ValueType>& x)
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

                            if (result != nullptr) {
                                for (uint choice = 0; choice < rowGroupSize; choice++, choice_it++) {
                                    if (*choice_it > optChoice) {
                                        result->set(rowGroupIndices[state] + choice, 0);
                                    }
                                }
                            }
                            else {
                                choice_it += rowGroupSize;
                            }
                            // reducing the xNew() vector for minimizer states
                            x[state] = optChoice;
                        }
                        else
                        {
                            optChoice = *std::max_element(choice_it, choice_it + rowGroupSize);
                            // reducing the xNew() vector for maximizer states
                            x[state] = optChoice;
                            choice_it += rowGroupSize;
                        }
                    }
                }


                template <typename ValueType>
                bool SoundGameViHelper<ValueType>::checkConvergence(ValueType threshold) const {
                    STORM_LOG_ASSERT(_multiplier, "tried to check for convergence without doing an iteration first.");
                    // Now check whether the currently produced results are precise enough
                    STORM_LOG_ASSERT(threshold > storm::utility::zero<ValueType>(), "Did not expect a non-positive threshold.");
                    auto x1It = xNewL().begin();
                    auto x1Ite = xNewL().end();
                    auto x2It = xNewU().begin();
                    // The difference between maxDiff and minDiff is zero at this point. Thus, it doesn't make sense to check the threshold now.
                    for (; x1It != x1Ite; x1It++, x2It++) {
                        ValueType diff = (*x2It - *x1It);
                        // Potentially update maxDiff or minDiff
                        if (diff > threshold) {
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

                template <typename ValueType>
                std::vector<ValueType>& SoundGameViHelper<ValueType>::xOldTest() {
                    return _x1IsCurrent ? _x2Test : _x1Test;
                }

                template <typename ValueType>
                std::vector<ValueType>& SoundGameViHelper<ValueType>::xNewTest() {
                    return _x1IsCurrent ? _x1Test : _x2Test;
                }

                template class SoundGameViHelper<double>;
#ifdef STORM_HAVE_CARL
                template class SoundGameViHelper<storm::RationalNumber>;
#endif
            }
        }
    }
}
