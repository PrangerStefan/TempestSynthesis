#include "storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h"

#include <sstream>

#include "storm/utility/constants.h"
#include "storm/utility/macros.h"
#include "storm/utility/vector.h"
#include "storm/utility/graph.h"
#include "storm/utility/FilteredRewardModel.h"

#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitParetoCurveCheckResult.h"

#include "storm/logic/FragmentSpecification.h"

#include "storm/transformer/DAProductBuilder.h"
#include "storm/logic/ExtractMaximalStateFormulasVisitor.h"
#include "storm/automata/LTL2DeterministicAutomaton.h"

#include "storm/models/sparse/StandardRewardModel.h"

#include "storm/modelchecker/prctl/helper/SparseMdpPrctlHelper.h"
#include "storm/modelchecker/helper/infinitehorizon/SparseNondeterministicInfiniteHorizonHelper.h"
#include "storm/modelchecker/helper/finitehorizon/SparseNondeterministicStepBoundedHorizonHelper.h"
#include "storm/modelchecker/helper/utility/SetInformationFromCheckTask.h"

#include "storm/modelchecker/prctl/helper/rewardbounded/QuantileHelper.h"
#include "storm/modelchecker/multiobjective/multiObjectiveModelChecking.h"

#include "storm/solver/SolveGoal.h"
#include "storm/storage/BitVector.h"

#include "storm/shields/ShieldHandling.h"

#include "storm/settings/SettingsManager.h"

#include "storm/settings/modules/GeneralSettings.h"
#include "storm/settings/modules/DebugSettings.h"

#include "storm/exceptions/InvalidStateException.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/storage/expressions/Expressions.h"

#include "storm/storage/MaximalEndComponentDecomposition.h"

#include "storm/exceptions/InvalidPropertyException.h"

namespace storm {
    namespace modelchecker {
        template<typename SparseMdpModelType>
        SparseMdpPrctlModelChecker<SparseMdpModelType>::SparseMdpPrctlModelChecker(SparseMdpModelType const& model) : SparsePropositionalModelChecker<SparseMdpModelType>(model) {
            // Intentionally left empty.
        }

        template<typename SparseMdpModelType>
        bool SparseMdpPrctlModelChecker<SparseMdpModelType>::canHandleStatic(CheckTask<storm::logic::Formula, ValueType> const& checkTask, bool* requiresSingleInitialState) {
            storm::logic::Formula const& formula = checkTask.getFormula();
            if (formula.isInFragment(storm::logic::prctlstar().setLongRunAverageRewardFormulasAllowed(true).setLongRunAverageProbabilitiesAllowed(true).setConditionalProbabilityFormulasAllowed(true).setOnlyEventuallyFormuluasInConditionalFormulasAllowed(true).setTotalRewardFormulasAllowed(true).setRewardBoundedUntilFormulasAllowed(true).setRewardBoundedCumulativeRewardFormulasAllowed(true).setMultiDimensionalBoundedUntilFormulasAllowed(true).setMultiDimensionalCumulativeRewardFormulasAllowed(true).setTimeOperatorsAllowed(true).setReachbilityTimeFormulasAllowed(true).setRewardAccumulationAllowed(true))) {
                return true;
            } else if (checkTask.isOnlyInitialStatesRelevantSet()) {
                auto multiObjectiveFragment = storm::logic::multiObjective().setCumulativeRewardFormulasAllowed(true).setTimeBoundedCumulativeRewardFormulasAllowed(true).setStepBoundedCumulativeRewardFormulasAllowed(true).setRewardBoundedCumulativeRewardFormulasAllowed(true).setTimeBoundedUntilFormulasAllowed(true).setStepBoundedUntilFormulasAllowed(true).setRewardBoundedUntilFormulasAllowed(true).setMultiDimensionalBoundedUntilFormulasAllowed(true).setMultiDimensionalCumulativeRewardFormulasAllowed(true).setRewardAccumulationAllowed(true);
                if (formula.isInFragment(multiObjectiveFragment) || formula.isInFragment(storm::logic::quantiles())) {
                    if (requiresSingleInitialState) {
                        *requiresSingleInitialState = true;
                    }
                    return true;
                }
            }
            return false;
        }

        template<typename SparseMdpModelType>
        bool SparseMdpPrctlModelChecker<SparseMdpModelType>::canHandle(CheckTask<storm::logic::Formula, ValueType> const& checkTask) const {
            bool requiresSingleInitialState = false;
            if (canHandleStatic(checkTask, &requiresSingleInitialState)) {
                return !requiresSingleInitialState || this->getModel().getInitialStates().getNumberOfSetBits() == 1;
            } else {
                return false;
            }
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeBoundedUntilProbabilities(Environment const& env, CheckTask<storm::logic::BoundedUntilFormula, ValueType> const& checkTask) {
            storm::logic::BoundedUntilFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            if (pathFormula.isMultiDimensional() || pathFormula.getTimeBoundReference().isRewardBound()) {
                STORM_LOG_THROW(checkTask.isOnlyInitialStatesRelevantSet(), storm::exceptions::InvalidOperationException, "Checking non-trivial bounded until probabilities can only be computed for the initial states of the model.");
                STORM_LOG_WARN_COND(!checkTask.isQualitativeSet(), "Checking non-trivial bounded until formulas is not optimized w.r.t. qualitative queries");
                storm::logic::OperatorInformation opInfo(checkTask.getOptimizationDirection());
                if (checkTask.isBoundSet()) {
                    opInfo.bound = checkTask.getBound();
                }
                auto formula = std::make_shared<storm::logic::ProbabilityOperatorFormula>(checkTask.getFormula().asSharedPointer(), opInfo);
                helper::rewardbounded::MultiDimensionalRewardUnfolding<ValueType, true> rewardUnfolding(this->getModel(), formula);
                auto numericResult = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeRewardBoundedValues(env, checkTask.getOptimizationDirection(), rewardUnfolding, this->getModel().getInitialStates());
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            } else {
                STORM_LOG_THROW(pathFormula.hasUpperBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have (a single) upper step bound.");
                STORM_LOG_THROW(pathFormula.hasIntegerLowerBound(), storm::exceptions::InvalidPropertyException, "Formula lower step bound must be discrete/integral.");
                STORM_LOG_THROW(pathFormula.hasIntegerUpperBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have discrete upper time bound.");
                std::unique_ptr<CheckResult> leftResultPointer = this->check(env, pathFormula.getLeftSubformula());
                std::unique_ptr<CheckResult> rightResultPointer = this->check(env, pathFormula.getRightSubformula());
                ExplicitQualitativeCheckResult const& leftResult = leftResultPointer->asExplicitQualitativeCheckResult();
                ExplicitQualitativeCheckResult const& rightResult = rightResultPointer->asExplicitQualitativeCheckResult();
                storm::modelchecker::helper::SparseNondeterministicStepBoundedHorizonHelper<ValueType> helper;
                std::vector<ValueType> numericResult;

                //This works only with empty vectors, no nullptr
                storm::storage::BitVector resultMaybeStates;
                std::vector<ValueType> choiceValues;

                numericResult = helper.compute(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), leftResult.getTruthValuesVector(), rightResult.getTruthValuesVector(), pathFormula.getNonStrictLowerBound<uint64_t>(), pathFormula.getNonStrictUpperBound<uint64_t>(), checkTask.getHint(), resultMaybeStates, choiceValues);
                if(checkTask.isShieldingTask()) {
                    tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), std::move(resultMaybeStates), storm::storage::BitVector(resultMaybeStates.size(), true));
                }
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            }
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeNextProbabilities(Environment const& env, CheckTask<storm::logic::NextFormula, ValueType> const& checkTask) {
            storm::logic::NextFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, pathFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeNextProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), checkTask.getOptimizationDirection(), this->getModel().getTransitionMatrix(), subResult.getTruthValuesVector());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if(checkTask.isShieldingTask()) {
                tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(ret.choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), std::move(ret.maybeStates), storm::storage::BitVector(ret.maybeStates.size(), true));
            } else if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeUntilProbabilities(Environment const& env, CheckTask<storm::logic::UntilFormula, ValueType> const& checkTask) {
            storm::logic::UntilFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> leftResultPointer = this->check(env, pathFormula.getLeftSubformula());
            std::unique_ptr<CheckResult> rightResultPointer = this->check(env, pathFormula.getRightSubformula());
            ExplicitQualitativeCheckResult const& leftResult = leftResultPointer->asExplicitQualitativeCheckResult();
            ExplicitQualitativeCheckResult const& rightResult = rightResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeUntilProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), leftResult.getTruthValuesVector(), rightResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if(checkTask.isShieldingTask()) {
                tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(ret.choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(), std::move(ret.maybeStates), storm::storage::BitVector(ret.maybeStates.size(), true));
            } else if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeGloballyProbabilities(Environment const& env, CheckTask<storm::logic::GloballyFormula, ValueType> const& checkTask) {
            storm::logic::GloballyFormula const& pathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, pathFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeGloballyProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), subResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if(checkTask.isShieldingTask()) {
                tempest::shields::createShield<ValueType>(std::make_shared<storm::models::sparse::Mdp<ValueType>>(this->getModel()), std::move(ret.choiceValues), checkTask.getShieldingExpression(), checkTask.getOptimizationDirection(),subResult.getTruthValuesVector(), storm::storage::BitVector(ret.maybeStates.size(), true));
            } else if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeStateFormulaProbabilities(Environment const& env, CheckTask<storm::logic::Formula, ValueType> const& checkTask) {
            storm::logic::Formula const& formula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> resultPointer = this->check(env, formula);
            ExplicitQualitativeCheckResult const& result = resultPointer->asExplicitQualitativeCheckResult();
            return std::make_unique<ExplicitQuantitativeCheckResult<ValueType>>(result);
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeLTLProbabilities(Environment const& env, CheckTask<storm::logic::PathFormula, ValueType> const& checkTask) {
            storm::logic::PathFormula const& pathFormula = checkTask.getFormula();

            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");

            std::vector<storm::logic::ExtractMaximalStateFormulasVisitor::LabelFormulaPair> extracted;
            std::shared_ptr<storm::logic::Formula> ltlFormula = storm::logic::ExtractMaximalStateFormulasVisitor::extract(pathFormula, extracted);

            STORM_LOG_INFO("Extracting maximal state formulas and computing satisfaction sets for path formula: " << pathFormula);

            std::map<std::string, storm::storage::BitVector> apSets;

            for (auto& p : extracted) {
                STORM_LOG_INFO(" Computing satisfaction set for atomic proposition \"" << p.first << "\" <=> " << *p.second << "...");

                std::unique_ptr<CheckResult> subResultPointer = this->check(env, *p.second);
                ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
                auto sat = subResult.getTruthValuesVector();

                STORM_LOG_INFO(" Atomic proposition \"" << p.first << "\" is satisfied by " << sat.getNumberOfSetBits() << " states.");

                apSets[p.first] = std::move(sat);
            }

            bool minimize = false;
            CheckTask<storm::logic::Formula, ValueType> subTask(checkTask);

            if (checkTask.getOptimizationDirection() == OptimizationDirection::Minimize) {
                minimize = true;
                // negate
                ltlFormula = std::make_shared<storm::logic::UnaryBooleanPathFormula>(storm::logic::UnaryBooleanOperatorType::Not, ltlFormula);
                STORM_LOG_INFO("Computing Pmin, proceeding with negated LTL formula.");

                subTask = checkTask.negate().substituteFormula(*ltlFormula);
            } else {
                subTask = checkTask.substituteFormula(*ltlFormula);
            }

            STORM_LOG_INFO("Resulting LTL path formula: " << *ltlFormula);
            STORM_LOG_INFO(" in prefix format: " << ltlFormula->toPrefixString());

            std::shared_ptr<storm::automata::DeterministicAutomaton> da = storm::automata::LTL2DeterministicAutomaton::ltl2da(*ltlFormula);

            STORM_LOG_INFO("Deterministic automaton for LTL formula has "
                    << da->getNumberOfStates() << " states, "
                    << da->getAPSet().size() << " atomic propositions and "
                    << *da->getAcceptance()->getAcceptanceExpression() << " as acceptance condition.");

            const SparseMdpModelType& mdp = this->getModel();
            storm::solver::SolveGoal<ValueType> goal(mdp, subTask);

            std::vector<ValueType> numericResult = computeDAProductProbabilities(env, std::move(goal), *da, apSets, checkTask.isQualitativeSet());

            if (minimize) {
                // compute 1-Pmax[!ltl]
                for (auto& value : numericResult) {
                    value = storm::utility::one<ValueType>() - value;
                }
            }
            return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
        }


        template<typename SparseMdpModelType>
        std::vector<typename SparseMdpPrctlModelChecker<SparseMdpModelType>::ValueType> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeDAProductProbabilities(Environment const& env, storm::solver::SolveGoal<typename SparseMdpPrctlModelChecker<SparseMdpModelType>::ValueType>&& goal, storm::automata::DeterministicAutomaton const& da, std::map<std::string, storm::storage::BitVector>& apSatSets, bool qualitative) const {
            STORM_LOG_THROW(goal.hasDirection() && goal.direction() == OptimizationDirection::Maximize, storm::exceptions::InvalidPropertyException, "Can only compute maximizing probabilties for DA product with MDP");

            const storm::automata::APSet& apSet = da.getAPSet();

            std::vector<storm::storage::BitVector> apLabels;
            for (const std::string& ap : apSet.getAPs()) {
                auto it = apSatSets.find(ap);
                STORM_LOG_THROW(it != apSatSets.end(), storm::exceptions::InvalidOperationException, "Deterministic automaton has AP " << ap << ", does not appear in formula");

                apLabels.push_back(std::move(it->second));
            }

            const SparseMdpModelType& mdp = this->getModel();

            storm::storage::BitVector statesOfInterest;
            if (goal.hasRelevantValues()) {
                statesOfInterest = goal.relevantValues();
            } else {
                // product from all model states
                statesOfInterest = storm::storage::BitVector(mdp.getNumberOfStates(), true);
            }

            STORM_LOG_INFO("Building MDP-DA product with deterministic automaton, starting from " << statesOfInterest.getNumberOfSetBits() << " model states...");
            storm::transformer::DAProductBuilder productBuilder(da, apLabels);
            auto product = productBuilder.build(mdp, statesOfInterest);
            STORM_LOG_INFO("Product MDP has " << product->getProductModel().getNumberOfStates() << " states and "
                    << product->getProductModel().getNumberOfTransitions() << " transitions.");

            if (storm::settings::getModule<storm::settings::modules::DebugSettings>().isTraceSet()) {
                STORM_LOG_TRACE("Writing model to model.dot");
                std::ofstream modelDot("model.dot");
                mdp.writeDotToStream(modelDot);
                modelDot.close();

                STORM_LOG_TRACE("Writing product model to product.dot");
                std::ofstream productDot("product.dot");
                product->getProductModel().writeDotToStream(productDot);
                productDot.close();

                STORM_LOG_TRACE("Product model mapping:");
                std::stringstream str;
                product->printMapping(str);
                STORM_LOG_TRACE(str.str());
            }

            STORM_LOG_INFO("Computing accepting end components...");
            storm::storage::BitVector accepting = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeSurelyAcceptingPmaxStates(*product->getAcceptance(), product->getProductModel().getTransitionMatrix(), product->getProductModel().getBackwardTransitions());
            if (accepting.empty()) {
                STORM_LOG_INFO("No accepting states, skipping probability computation.");
                std::vector<ValueType> numericResult(this->getModel().getNumberOfStates(), storm::utility::zero<ValueType>());
                return numericResult;
            }

            STORM_LOG_INFO("Computing probabilities for reaching accepting BSCCs...");

            storm::storage::BitVector bvTrue(product->getProductModel().getNumberOfStates(), true);

            storm::solver::SolveGoal<ValueType> solveGoalProduct(goal);
            storm::storage::BitVector soiProduct(product->getStatesOfInterest());
            solveGoalProduct.setRelevantValues(std::move(soiProduct));

            std::vector<ValueType> prodNumericResult
                = std::move(storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeUntilProbabilities(env,
                                                                                                                    std::move(solveGoalProduct),
                                                                                                                    product->getProductModel().getTransitionMatrix(),
                                                                                                                    product->getProductModel().getBackwardTransitions(),
                                                                                                                    bvTrue,
                                                                                                                    accepting,
                                                                                                                    qualitative,
                                                                                                                    false  // no schedulers (at the moment)
                                                                                                                   ).values);

            std::vector<ValueType> numericResult = product->projectToOriginalModel(this->getModel(), prodNumericResult);
            return numericResult;
        }


        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeConditionalProbabilities(Environment const& env, CheckTask<storm::logic::ConditionalFormula, ValueType> const& checkTask) {
            storm::logic::ConditionalFormula const& conditionalFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            STORM_LOG_THROW(this->getModel().getInitialStates().getNumberOfSetBits() == 1, storm::exceptions::InvalidPropertyException, "Cannot compute conditional probabilities on MDPs with more than one initial state.");
            STORM_LOG_THROW(conditionalFormula.getSubformula().isEventuallyFormula(), storm::exceptions::InvalidPropertyException, "Illegal conditional probability formula.");
            STORM_LOG_THROW(conditionalFormula.getConditionFormula().isEventuallyFormula(), storm::exceptions::InvalidPropertyException, "Illegal conditional probability formula.");

            std::unique_ptr<CheckResult> leftResultPointer = this->check(env, conditionalFormula.getSubformula().asEventuallyFormula().getSubformula());
            std::unique_ptr<CheckResult> rightResultPointer = this->check(env, conditionalFormula.getConditionFormula().asEventuallyFormula().getSubformula());
            ExplicitQualitativeCheckResult const& leftResult = leftResultPointer->asExplicitQualitativeCheckResult();
            ExplicitQualitativeCheckResult const& rightResult = rightResultPointer->asExplicitQualitativeCheckResult();

            return storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeConditionalProbabilities(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), leftResult.getTruthValuesVector(), rightResult.getTruthValuesVector());
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeCumulativeRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::CumulativeRewardFormula, ValueType> const& checkTask) {
            storm::logic::CumulativeRewardFormula const& rewardPathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            if (rewardPathFormula.isMultiDimensional() || rewardPathFormula.getTimeBoundReference().isRewardBound()) {
                STORM_LOG_THROW(checkTask.isOnlyInitialStatesRelevantSet(), storm::exceptions::InvalidOperationException, "Checking reward bounded cumulative reward formulas can only be done for the initial states of the model.");
                STORM_LOG_THROW(!checkTask.getFormula().hasRewardAccumulation(), storm::exceptions::InvalidOperationException, "Checking reward bounded cumulative reward formulas is not supported if reward accumulations are given.");
                STORM_LOG_WARN_COND(!checkTask.isQualitativeSet(), "Checking reward bounded until formulas is not optimized w.r.t. qualitative queries");
                storm::logic::OperatorInformation opInfo(checkTask.getOptimizationDirection());
                if (checkTask.isBoundSet()) {
                    opInfo.bound = checkTask.getBound();
                }
                auto formula = std::make_shared<storm::logic::RewardOperatorFormula>(checkTask.getFormula().asSharedPointer(), checkTask.getRewardModel(), opInfo);
                helper::rewardbounded::MultiDimensionalRewardUnfolding<ValueType, true> rewardUnfolding(this->getModel(), formula);
                auto numericResult = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeRewardBoundedValues(env, checkTask.getOptimizationDirection(), rewardUnfolding, this->getModel().getInitialStates());
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            } else {
                STORM_LOG_THROW(rewardPathFormula.hasIntegerBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have a discrete time bound.");
                auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
                std::vector<ValueType> numericResult = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeCumulativeRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), rewardModel.get(), rewardPathFormula.getNonStrictBound<uint64_t>());
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
            }
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeInstantaneousRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::InstantaneousRewardFormula, ValueType> const& checkTask) {
            storm::logic::InstantaneousRewardFormula const& rewardPathFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            STORM_LOG_THROW(rewardPathFormula.hasIntegerBound(), storm::exceptions::InvalidPropertyException, "Formula needs to have a discrete time bound.");
            std::vector<ValueType> numericResult = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeInstantaneousRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), checkTask.isRewardModelSet() ? this->getModel().getRewardModel(checkTask.getRewardModel()) : this->getModel().getRewardModel(""), rewardPathFormula.getBound<uint64_t>());
            return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(std::move(numericResult)));
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeReachabilityRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) {
            storm::logic::EventuallyFormula const& eventuallyFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, eventuallyFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeReachabilityRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), rewardModel.get(), subResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeReachabilityTimes(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::EventuallyFormula, ValueType> const& checkTask) {
            storm::logic::EventuallyFormula const& eventuallyFormula = checkTask.getFormula();
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            std::unique_ptr<CheckResult> subResultPointer = this->check(env, eventuallyFormula.getSubformula());
            ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeReachabilityTimes(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), subResult.getTruthValuesVector(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeTotalRewards(Environment const& env, storm::logic::RewardMeasureType, CheckTask<storm::logic::TotalRewardFormula, ValueType> const& checkTask) {
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
            auto ret = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>::computeTotalRewards(env, storm::solver::SolveGoal<ValueType>(this->getModel(), checkTask), this->getModel().getTransitionMatrix(), this->getModel().getBackwardTransitions(), rewardModel.get(), checkTask.isQualitativeSet(), checkTask.isProduceSchedulersSet(), checkTask.getHint());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(ret.values)));
            if (checkTask.isProduceSchedulersSet() && ret.scheduler) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::move(ret.scheduler));
            }
            return result;
        }

		template<typename SparseMdpModelType>
		std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeLongRunAverageProbabilities(Environment const& env, CheckTask<storm::logic::StateFormula, ValueType> const& checkTask) {
		    storm::logic::StateFormula const& stateFormula = checkTask.getFormula();
			STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
			std::unique_ptr<CheckResult> subResultPointer = this->check(env, stateFormula);
			ExplicitQualitativeCheckResult const& subResult = subResultPointer->asExplicitQualitativeCheckResult();

			storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> helper(this->getModel().getTransitionMatrix());
            storm::modelchecker::helper::setInformationFromCheckTaskNondeterministic(helper, checkTask, this->getModel());
			auto values = helper.computeLongRunAverageProbabilities(env, subResult.getTruthValuesVector());

            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(values)));
            if (checkTask.isProduceSchedulersSet()) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::make_unique<storm::storage::Scheduler<ValueType>>(helper.extractScheduler()));
            }
            return result;
		}

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::computeLongRunAverageRewards(Environment const& env, storm::logic::RewardMeasureType rewardMeasureType, CheckTask<storm::logic::LongRunAverageRewardFormula, ValueType> const& checkTask) {
            STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "Formula needs to specify whether minimal or maximal values are to be computed on nondeterministic model.");
            auto rewardModel = storm::utility::createFilteredRewardModel(this->getModel(), checkTask);
            storm::modelchecker::helper::SparseNondeterministicInfiniteHorizonHelper<ValueType> helper(this->getModel().getTransitionMatrix());
            storm::modelchecker::helper::setInformationFromCheckTaskNondeterministic(helper, checkTask, this->getModel());
			auto values = helper.computeLongRunAverageRewards(env, rewardModel.get());
            std::unique_ptr<CheckResult> result(new ExplicitQuantitativeCheckResult<ValueType>(std::move(values)));
            if (checkTask.isProduceSchedulersSet()) {
                result->asExplicitQuantitativeCheckResult<ValueType>().setScheduler(std::make_unique<storm::storage::Scheduler<ValueType>>(helper.extractScheduler()));
            }
            return result;
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::checkMultiObjectiveFormula(Environment const& env, CheckTask<storm::logic::MultiObjectiveFormula, ValueType> const& checkTask) {
            return multiobjective::performMultiObjectiveModelChecking(env, this->getModel(), checkTask.getFormula());
        }

        template<typename SparseMdpModelType>
        std::unique_ptr<CheckResult> SparseMdpPrctlModelChecker<SparseMdpModelType>::checkQuantileFormula(Environment const& env, CheckTask<storm::logic::QuantileFormula, ValueType> const& checkTask) {
            STORM_LOG_THROW(checkTask.isOnlyInitialStatesRelevantSet(), storm::exceptions::InvalidOperationException, "Computing quantiles is only supported for the initial states of a model.");
            STORM_LOG_THROW(this->getModel().getInitialStates().getNumberOfSetBits() == 1, storm::exceptions::InvalidOperationException, "Quantiles not supported on models with multiple initial states.");
            uint64_t initialState = *this->getModel().getInitialStates().begin();

            helper::rewardbounded::QuantileHelper<SparseMdpModelType> qHelper(this->getModel(), checkTask.getFormula());
            auto res = qHelper.computeQuantile(env);

            if (res.size() == 1 && res.front().size() == 1) {
                return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<ValueType>(initialState, std::move(res.front().front())));
            } else {
                return std::unique_ptr<CheckResult>(new ExplicitParetoCurveCheckResult<ValueType>(initialState, std::move(res)));
            }
        }

        template class SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>>;

#ifdef STORM_HAVE_CARL
        template class SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<storm::RationalNumber>>;
#endif
    }
}
