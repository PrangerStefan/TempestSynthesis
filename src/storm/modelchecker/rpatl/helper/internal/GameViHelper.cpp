#include "GameViHelper.h"

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
                GameViHelper<ValueType>::GameViHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::BitVector statesOfCoalition) : _transitionMatrix(transitionMatrix), _statesOfCoalition(statesOfCoalition) {
                }

                template <typename ValueType>
                void GameViHelper<ValueType>::prepareSolversAndMultipliers(const Environment& env) {
                    _multiplier = storm::solver::MultiplierFactory<ValueType>().create(env, _transitionMatrix);

                    _x1IsCurrent = false;
                }

                template <typename ValueType>
                void GameViHelper<ValueType>::performValueIteration(Environment const& env, std::vector<ValueType>& x, std::vector<ValueType> b, storm::solver::OptimizationDirection const dir) {
                    prepareSolversAndMultipliers(env);
                    ValueType precision = storm::utility::convertNumber<ValueType>(env.solver().game().getPrecision());
                    uint64_t maxIter = env.solver().game().getMaximalNumberOfIterations();
                    _b = b;

                    _x1.assign(_transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                    _x2 = _x1;

                    if (this->isProduceSchedulerSet()) {
                        if (!this->_producedOptimalChoices.is_initialized()) {
                            this->_producedOptimalChoices.emplace();
                        }
                        this->_producedOptimalChoices->resize(this->_transitionMatrix.getRowGroupCount());
                    }

                    uint64_t iter = 0;
                    std::vector<ValueType> result = x;
                    while (iter < maxIter) {
                        ++iter;
                        performIterationStep(env, dir);

                        if (checkConvergence(precision)) {
                            break;
                        }
                        if (storm::utility::resources::isTerminate()) {
                            break;
                        }
                    }
                    x = xNew();

                    if (isProduceSchedulerSet()) {
                        // We will be doing one more iteration step and track scheduler choices this time.
                        performIterationStep(env, dir, &_producedOptimalChoices.get());
                    }
                }

                template <typename ValueType>
                void GameViHelper<ValueType>::performIterationStep(Environment const& env, storm::solver::OptimizationDirection const dir, std::vector<uint64_t>* choices) {
                    if (!_multiplier) {
                        prepareSolversAndMultipliers(env);
                    }
                    _x1IsCurrent = !_x1IsCurrent;

                    if (choices == nullptr) {
                        _multiplier->multiplyAndReduce(env, dir, xOld(), &_b, xNew(), nullptr, &_statesOfCoalition);
                    } else {
                        _multiplier->multiplyAndReduce(env, dir, xOld(), &_b, xNew(), choices, &_statesOfCoalition);
                    }

                    // compare applyPointwise
                    storm::utility::vector::applyPointwise<ValueType, ValueType, ValueType>(xOld(), xNew(), xNew(), [&dir] (ValueType const& a, ValueType const& b) -> ValueType {
                       if(a > b) return a;
                       else return b;
                    });
                }

                template <typename ValueType>
                bool GameViHelper<ValueType>::checkConvergence(ValueType threshold) const {
                    STORM_LOG_ASSERT(_multiplier, "tried to check for convergence without doing an iteration first.");

                    // Now check whether the currently produced results are precise enough
                    STORM_LOG_ASSERT(threshold > storm::utility::zero<ValueType>(), "Did not expect a non-positive threshold.");
                    auto x1It = xOld().begin();
                    auto x1Ite = xOld().end();
                    auto x2It = xNew().begin();
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
                void GameViHelper<ValueType>::fillResultVector(std::vector<ValueType>& result, storm::storage::BitVector relevantStates, storm::storage::BitVector psiStates) {
                    std::vector<ValueType> filledVector = std::vector<ValueType>(relevantStates.size(), storm::utility::zero<ValueType>());
                    uint bitIndex = 0;
                    for(uint i = 0; i < filledVector.size(); i++) {
                        if (relevantStates.get(i)) {
                            filledVector.at(i) = result.at(bitIndex);
                            bitIndex++;
                        }
                        else if (psiStates.get(i)) {
                            filledVector.at(i) = 1;
                        }
                    }
                    result = filledVector;
                }

                template <typename ValueType>
                void GameViHelper<ValueType>::fillChoiceValuesVector(std::vector<ValueType>& choiceValues, storm::storage::BitVector psiStates, std::vector<storm::storage::SparseMatrix<double>::index_type> rowGroupIndices) {
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
                void GameViHelper<ValueType>::setProduceScheduler(bool value) {
                    _produceScheduler = value;
                }

                template <typename ValueType>
                bool GameViHelper<ValueType>::isProduceSchedulerSet() const {
                    return _produceScheduler;
                }

                template <typename ValueType>
                std::vector<uint64_t> const& GameViHelper<ValueType>::getProducedOptimalChoices() const {
                    STORM_LOG_ASSERT(this->isProduceSchedulerSet(), "Trying to get the produced optimal choices although no scheduler was requested.");
                    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(), "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
                    return this->_producedOptimalChoices.get();
                }

                template <typename ValueType>
                std::vector<uint64_t>& GameViHelper<ValueType>::getProducedOptimalChoices() {
                    STORM_LOG_ASSERT(this->isProduceSchedulerSet(), "Trying to get the produced optimal choices although no scheduler was requested.");
                    STORM_LOG_ASSERT(this->_producedOptimalChoices.is_initialized(), "Trying to get the produced optimal choices but none were available. Was there a computation call before?");
                    return this->_producedOptimalChoices.get();
                }

                template <typename ValueType>
                storm::storage::Scheduler<ValueType> GameViHelper<ValueType>::extractScheduler() const{
                    auto const& optimalChoices = getProducedOptimalChoices();
                    storm::storage::Scheduler<ValueType> scheduler(optimalChoices.size());
                    for (uint64_t state = 0; state < optimalChoices.size(); ++state) {
                        scheduler.setChoice(optimalChoices[state], state);
                    }
                    return scheduler;
                }

                template <typename ValueType>
                void GameViHelper<ValueType>::getChoiceValues(Environment const& env, std::vector<ValueType> const& x, std::vector<ValueType>& choiceValues) {
                    _multiplier->multiply(env, x, &_b, choiceValues);
                }

                template <typename ValueType>
                std::vector<ValueType>& GameViHelper<ValueType>::xNew() {
                    return _x1IsCurrent ? _x1 : _x2;
                }

                template <typename ValueType>
                std::vector<ValueType> const& GameViHelper<ValueType>::xNew() const {
                    return _x1IsCurrent ? _x1 : _x2;
                }

                template <typename ValueType>
                std::vector<ValueType>& GameViHelper<ValueType>::xOld() {
                    return _x1IsCurrent ? _x2 : _x1;
                }

                template <typename ValueType>
                std::vector<ValueType> const& GameViHelper<ValueType>::xOld() const {
                    return _x1IsCurrent ? _x2 : _x1;
                }

                template class GameViHelper<double>;
#ifdef STORM_HAVE_CARL
                template class GameViHelper<storm::RationalNumber>;
#endif
            }
        }
    }
}
