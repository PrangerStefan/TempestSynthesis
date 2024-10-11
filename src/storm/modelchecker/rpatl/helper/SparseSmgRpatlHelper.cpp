#include <utility/graph.h>
#include <environment/solver/GameSolverEnvironment.h>
#include "SparseSmgRpatlHelper.h"

#include "storm/environment/Environment.h"
#include "storm/environment/solver/MultiplierEnvironment.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/utility/vector.h"
#include "storm/utility/graph.h"
#include "storm/modelchecker/rpatl/helper/internal/GameViHelper.h"
#include "storm/modelchecker/rpatl/helper/internal/SoundGameViHelper.h"

namespace storm {
    namespace modelchecker {
        namespace helper {

            template<typename ValueType>
            SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeUntilProbabilities(Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, ModelCheckerHint const& hint) {
                auto solverEnv = env;
                solverEnv.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration, false);

                // Relevant states are those states which are phiStates and not PsiStates.
                storm::storage::BitVector relevantStates = phiStates & ~psiStates;
                    // Initialize the x vector and solution vector result.
                std::vector<ValueType> x = std::vector<ValueType>(relevantStates.getNumberOfSetBits(), storm::utility::zero<ValueType>());
                std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates);
                std::vector<ValueType> constrainedChoiceValues = std::vector<ValueType>(b.size(), storm::utility::zero<ValueType>());
                std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;

                storm::storage::BitVector clippedStatesOfCoalition(relevantStates.getNumberOfSetBits());
                clippedStatesOfCoalition.setClippedStatesOfCoalition(relevantStates, statesOfCoalition);

                if(!relevantStates.empty()) {
                    // Reduce the matrix to relevant states.
                    storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);
                    // Create GameViHelper for computations.
                    storm::modelchecker::helper::internal::GameViHelper<ValueType> viHelper(submatrix, clippedStatesOfCoalition);
                    if (produceScheduler) {
                        viHelper.setProduceScheduler(true);
                    }
                    viHelper.performValueIteration(env, x, b, goal.direction(), constrainedChoiceValues);
                    if(goal.isShieldingTask()) {
                        viHelper.getChoiceValues(env, x, constrainedChoiceValues);
                    }

                    // Fill up the constrainedChoice Values to full size.
                    viHelper.fillChoiceValuesVector(constrainedChoiceValues, relevantStates, transitionMatrix.getRowGroupIndices());

                    if (produceScheduler) {
                        scheduler = std::make_unique<storm::storage::Scheduler<ValueType>>(expandScheduler(viHelper.extractScheduler(), psiStates, ~phiStates));
                    }
                }

                // Fill up the result vector with the values of x for the relevant states, with 1s for psi states (0 is default)
                storm::utility::vector::setVectorValues(result, relevantStates, x);
                storm::utility::vector::setVectorValues(result, psiStates, storm::utility::one<ValueType>());
                return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(relevantStates), std::move(scheduler), std::move(constrainedChoiceValues));
            }

            template<typename ValueType>
            SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeUntilProbabilitiesSound(Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, ModelCheckerHint const& hint) {

                storm::storage::BitVector probGreater0 = storm::utility::graph::performProbGreater0(backwardTransitions, phiStates, psiStates);
                std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;
                storm::storage::BitVector relevantStates = phiStates;

                storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);

                // Initialize the x vector and solution vector result.
                std::vector<ValueType> xL = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                // assigning 1s to the xL vector for all Goal states
                auto xL_begin = xL.begin();
                std::for_each(xL.begin(), xL.end(), [&psiStates, &xL_begin](ValueType &it)
                              {
                                  if (psiStates[&it - &(*xL_begin)])
                                      it = 1;
                              });
                size_t i = 0;
                auto new_end = std::remove_if(xL.begin(), xL.end(), [&relevantStates, &i](const auto& item) {
                    bool ret = !(relevantStates[i]);
                    i++;
                    return ret;
                });
                xL.erase(new_end, xL.end());
                xL.resize(relevantStates.getNumberOfSetBits());
                std::vector<ValueType> xU = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                // assigning 1s to the xU vector for all states except the states s where Prob(sEf) = 0 for all goal states f
                auto xU_begin = xU.begin();
                std::for_each(xU.begin(), xU.end(), [&probGreater0, &xU_begin](ValueType &it)
                              {
                                  if (probGreater0[&it - &(*xU_begin)])
                                      it = 1;
                              });
                i = 0;
                auto new_end_U = std::remove_if(xU.begin(), xU.end(), [&relevantStates, &i](const auto& item) {
                    bool ret = !(relevantStates[i]);
                    i++;
                    return ret;
                });
                xU.erase(new_end_U, xU.end());
                xU.resize(relevantStates.getNumberOfSetBits());

                storm::storage::BitVector clippedPsiStates(relevantStates.getNumberOfSetBits());
                clippedPsiStates.setClippedStatesOfCoalition(relevantStates, psiStates);

                std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates);
                std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());

                storm::storage::BitVector clippedStatesOfCoalition(relevantStates.getNumberOfSetBits());
                clippedStatesOfCoalition.setClippedStatesOfCoalition(relevantStates, statesOfCoalition);
                std::vector<ValueType> constrainedChoiceValues = std::vector<ValueType>(b.size(), storm::utility::zero<ValueType>());

                storm::modelchecker::helper::internal::SoundGameViHelper<ValueType> viHelper(submatrix, submatrix.transpose(), b, clippedStatesOfCoalition, clippedPsiStates, goal.direction());

                if (produceScheduler) {
                    viHelper.setProduceScheduler(true);
                }

                viHelper.performValueIteration(env, xL, xU, goal.direction(), constrainedChoiceValues);

                viHelper.fillChoiceValuesVector(constrainedChoiceValues, relevantStates, transitionMatrix.getRowGroupIndices());
                storm::utility::vector::setVectorValues(result, relevantStates, xL);

                if (produceScheduler) {
                    scheduler = std::make_unique<storm::storage::Scheduler<ValueType>>(expandScheduler(viHelper.extractScheduler(), psiStates, ~phiStates));
                }

                return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(relevantStates), std::move(scheduler), std::move(constrainedChoiceValues));
            }

            template<typename ValueType>
            storm::storage::Scheduler<ValueType> SparseSmgRpatlHelper<ValueType>::expandScheduler(storm::storage::Scheduler<ValueType> scheduler, storm::storage::BitVector psiStates, storm::storage::BitVector notPhiStates, bool sound) {
                storm::storage::Scheduler<ValueType> completeScheduler(psiStates.size());
                uint_fast64_t maybeStatesCounter = 0;
                uint schedulerSize = psiStates.size();
                for(uint stateCounter = 0; stateCounter < schedulerSize; stateCounter++) {
                    // psiStates already fulfill formulae so we can set an arbitrary action
                    if(psiStates.get(stateCounter)) {
                        completeScheduler.setChoice(0, stateCounter);
                        if (sound) {
                            maybeStatesCounter++;
                        }
                    // ~phiStates do not fulfill formulae so we can set an arbitrary action
                    } else if(notPhiStates.get(stateCounter)) {
                        completeScheduler.setChoice(0, stateCounter);
                    } else {
                        completeScheduler.setChoice(scheduler.getChoice(maybeStatesCounter), stateCounter);
                        maybeStatesCounter++;
                    }
                }
                return completeScheduler;
            }

            template<typename ValueType>
            SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeGloballyProbabilities(Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, ModelCheckerHint const& hint) {
                // G psi = not(F(not psi)) = not(true U (not psi))
                // The psiStates are flipped, then the true U part is calculated, at the end the result is flipped again.
                storm::storage::BitVector notPsiStates = ~psiStates;
                statesOfCoalition.complement();

                auto result = computeUntilProbabilitiesSound(env, std::move(goal), transitionMatrix, backwardTransitions, storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), notPsiStates, qualitative, statesOfCoalition, produceScheduler, hint);
                for (auto& element : result.values) {
                    element = storm::utility::one<ValueType>() - element;
                }
                for (auto& element : result.choiceValues) {
                    element = storm::utility::one<ValueType>() - element;
                }
                return result;
            }

            template<typename ValueType>
            SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeNextProbabilities(Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, ModelCheckerHint const& hint) {
                // Create vector result, bitvector allStates with a true for each state and a vector b for the probability for each state to get to a psi state, choiceValues is to store choices for shielding.
                std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                storm::storage::BitVector allStates = storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true);
                std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(allStates, psiStates);
                std::vector<ValueType> choiceValues = std::vector<ValueType>(transitionMatrix.getRowCount(), storm::utility::zero<ValueType>());
                statesOfCoalition.complement();

                if (produceScheduler) {
                    STORM_LOG_WARN("Next formula does not expect that produceScheduler is set to true.");
                }
                // Create a multiplier for reduction.
                auto multiplier = storm::solver::MultiplierFactory<ValueType>().create(env, transitionMatrix);
                auto rowGroupIndices = transitionMatrix.getRowGroupIndices();
                rowGroupIndices.erase(rowGroupIndices.begin());
                multiplier->reduce(env, goal.direction(), rowGroupIndices, b, result, nullptr, &statesOfCoalition);
                if (goal.isShieldingTask()) {
                    choiceValues = b;
                }
                return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(allStates), nullptr, std::move(choiceValues));
            }

            template<typename ValueType>
            SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeBoundedGloballyProbabilities(Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, ModelCheckerHint const& hint,uint64_t lowerBound, uint64_t upperBound) {
                // G psi = not(F(not psi)) = not(true U (not psi))
                // The psiStates are flipped, then the true U part is calculated, at the end the result is flipped again.
                storm::storage::BitVector notPsiStates = ~psiStates;
                statesOfCoalition.complement();

                auto result = computeBoundedUntilProbabilities(env, std::move(goal), transitionMatrix, backwardTransitions, storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), notPsiStates, qualitative, statesOfCoalition, produceScheduler, hint, lowerBound, upperBound, true);
                for (auto& element : result.values) {
                    element = storm::utility::one<ValueType>() - element;
                }
                for (auto& element : result.choiceValues) {
                    element = storm::utility::one<ValueType>() - element;
                }
                return result;
            }

            template<typename ValueType>
            SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeBoundedUntilProbabilities(Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, ModelCheckerHint const& hint,uint64_t lowerBound, uint64_t upperBound, bool computeBoundedGlobally) {
                auto solverEnv = env;
                solverEnv.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration, false);

                // boundedUntil formulas look like:
                // phi U [lowerBound, upperBound] psi
                // --
                // We solve this by look at psiStates, finding phiStates which have paths to psiStates in the given step bounds,
                // then we find all states which have a path to those phiStates in the given lower bound
                // (which states the paths pass before the lower bound does not matter).

                // First initialization of relevantStates between the step bounds.
                storm::storage::BitVector relevantStates = phiStates & ~psiStates;

                // Initializations.
                std::vector<ValueType> x = std::vector<ValueType>(relevantStates.getNumberOfSetBits(), storm::utility::zero<ValueType>());
                std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates);
                std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                std::vector<ValueType> constrainedChoiceValues = std::vector<ValueType>(transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates).size(), storm::utility::zero<ValueType>());
                std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;

                storm::storage::BitVector clippedStatesOfCoalition(relevantStates.getNumberOfSetBits());
                clippedStatesOfCoalition.setClippedStatesOfCoalition(relevantStates, statesOfCoalition);

                // If there are no relevantStates or the upperBound is 0, no computation is needed.
                if(!relevantStates.empty() && upperBound > 0) {
                    // Reduce the matrix to relevant states. - relevant states are all states.
                    storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);
                    // Create GameViHelper for computations.
                    storm::modelchecker::helper::internal::GameViHelper<ValueType> viHelper(submatrix, clippedStatesOfCoalition);
                    if (produceScheduler) {
                        viHelper.setProduceScheduler(true);
                    }
                    // If the lowerBound = 0, value iteration is done until the upperBound.
                    if(lowerBound == 0) {
                        solverEnv.solver().game().setMaximalNumberOfIterations(upperBound);
                        viHelper.performValueIteration(solverEnv, x, b, goal.direction(), constrainedChoiceValues);
                    } else {
                        // The lowerBound != 0, the first computation between the given bound steps is done.
                        solverEnv.solver().game().setMaximalNumberOfIterations(upperBound - lowerBound);
                        viHelper.performValueIteration(solverEnv, x, b, goal.direction(), constrainedChoiceValues);

                        // Initialization of subResult, fill it with the result of the first computation and 1s for the psiStates in full range.
                        std::vector<ValueType> subResult = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                        storm::utility::vector::setVectorValues(subResult, relevantStates, x);
                        storm::utility::vector::setVectorValues(subResult, psiStates, storm::utility::one<ValueType>());

                        // The newPsiStates are those states which can reach the psiStates in the steps between the bounds - the !=0 values in subResult.
                        storm::storage::BitVector newPsiStates(subResult.size(), false);
                        storm::utility::vector::setNonzeroIndices(subResult, newPsiStates);

                        // The relevantStates for the second part of the computation are all states.
                        relevantStates = storm::storage::BitVector(phiStates.size(), true);
                        submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);

                        // Update the viHelper for the (full-size) submatrix and statesOfCoalition.
                        viHelper.updateTransitionMatrix(submatrix);
                        viHelper.updateStatesOfCoalition(statesOfCoalition);

                        // Reset constrainedChoiceValues and b to 0-vector in the correct dimension.
                        constrainedChoiceValues = std::vector<ValueType>(transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, newPsiStates).size(), storm::utility::zero<ValueType>());
                        b = std::vector<ValueType>(transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, newPsiStates).size(), storm::utility::zero<ValueType>());

                        // The second computation is done between step 0 and the lowerBound
                        solverEnv.solver().game().setMaximalNumberOfIterations(lowerBound);
                        viHelper.performValueIteration(solverEnv, subResult, b, goal.direction(), constrainedChoiceValues);

                        x = subResult;
                    }
                    viHelper.fillChoiceValuesVector(constrainedChoiceValues, relevantStates, transitionMatrix.getRowGroupIndices());
                    if (produceScheduler) {
                        scheduler = std::make_unique<storm::storage::Scheduler<ValueType>>(expandScheduler(viHelper.extractScheduler(), relevantStates, ~relevantStates));
                    }
                    storm::utility::vector::setVectorValues(result, relevantStates, x);
                }
                // In bounded until and bounded eventually formula the psiStates have probability 1 to satisfy the formula,
                // because once reaching a state where psi holds those formulas are satisfied.
                // In bounded globally formulas we cannot set those states to 1 because it is possible to leave a set of safe states after reaching a psiState
                // and in globally the formula has to hold in every time step (between the bounds).
                // e.g. phiState -> phiState -> psiState -> unsafeState
                if(!computeBoundedGlobally){
                    storm::utility::vector::setVectorValues(result, psiStates, storm::utility::one<ValueType>());
                }
                return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(relevantStates), std::move(scheduler), std::move(constrainedChoiceValues));
            }

            template class SparseSmgRpatlHelper<double>;
#ifdef STORM_HAVE_CARL
            template class SparseSmgRpatlHelper<storm::RationalNumber>;
#endif
        }
    }
}
