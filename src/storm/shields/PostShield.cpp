#include "storm/shields/PostShield.h"

#include <algorithm>

namespace tempest {
    namespace shields {

        template<typename ValueType, typename IndexType>
        PostShield<ValueType, IndexType>::PostShield(std::vector<IndexType> const& rowGroupIndices, std::vector<ValueType> const& choiceValues, std::shared_ptr<storm::logic::ShieldExpression const> const& shieldingExpression, storm::OptimizationDirection optimizationDirection, storm::storage::BitVector relevantStates, boost::optional<storm::storage::BitVector> coalitionStates) : AbstractShield<ValueType, IndexType>(rowGroupIndices, shieldingExpression, optimizationDirection, relevantStates, coalitionStates), choiceValues(choiceValues) {
            // Intentionally left empty.
        }

        template<typename ValueType, typename IndexType>
        storm::storage::PostScheduler<ValueType> PostShield<ValueType, IndexType>::construct() {
            if (this->getOptimizationDirection() == storm::OptimizationDirection::Minimize) {
                if(this->shieldingExpression->isRelative()) {
                    return constructWithCompareType<storm::utility::ElementLessEqual<ValueType>, true>();
                } else {
                    return constructWithCompareType<storm::utility::ElementLessEqual<ValueType>, false>();
                }
            } else {
                if(this->shieldingExpression->isRelative()) {
                    return constructWithCompareType<storm::utility::ElementGreaterEqual<ValueType>, true>();
                } else {
                    return constructWithCompareType<storm::utility::ElementGreaterEqual<ValueType>, false>();
                }
            }
        }

        template<typename ValueType, typename IndexType>
        template<typename Compare, bool relative>
        storm::storage::PostScheduler<ValueType> PostShield<ValueType, IndexType>::constructWithCompareType() {
            tempest::shields::utility::ChoiceFilter<ValueType, Compare, relative> choiceFilter;
            storm::storage::PostScheduler<ValueType> shield(this->rowGroupIndices.size() - 1, this->computeRowGroupSizes());
            auto choice_it = this->choiceValues.begin();
            if(this->coalitionStates.is_initialized()) {
                this->relevantStates &= ~this->coalitionStates.get();
            }
            for(uint state = 0; state < this->rowGroupIndices.size() - 1; state++) {
                uint rowGroupSize = this->rowGroupIndices[state + 1] - this->rowGroupIndices[state];
                if(this->relevantStates.get(state)) {
                    auto optProbabilityIndex = std::min_element(choice_it, choice_it + rowGroupSize) - choice_it;
                    if(std::is_same<Compare, storm::utility::ElementGreaterEqual<ValueType>>::value) {
                        optProbabilityIndex = std::max_element(choice_it, choice_it + rowGroupSize) - choice_it;
                    }
                    ValueType optProbability = *(choice_it + optProbabilityIndex);
                    if(!relative && !choiceFilter(optProbability, optProbability, this->shieldingExpression->getValue())) {
                        STORM_LOG_WARN("No shielding action possible with absolute comparison for state with index " << state);
                        shield.setChoice(storm::storage::PostSchedulerChoice<ValueType>(), state, 0);
                        choice_it += rowGroupSize;
                        continue;
                    }
                    storm::storage::PostSchedulerChoice<ValueType> choiceMapping;
                    for(uint choice = 0; choice < rowGroupSize; choice++, choice_it++) {
                        if(choiceFilter(*choice_it, optProbability, this->shieldingExpression->getValue())) {
                            choiceMapping.addChoice(choice, choice);
                        } else {
                            choiceMapping.addChoice(choice, optProbabilityIndex);
                        }
                    }
                    shield.setChoice(choiceMapping, state, 0);
                } else {
                    shield.setChoice(storm::storage::PostSchedulerChoice<ValueType>(), state, 0);
                    choice_it += rowGroupSize;
                }
            }
            return shield;
        }


        template<typename ValueType, typename IndexType>
        void PostShield<ValueType, IndexType>::printToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> const& model) {
           this->construct().printToStream(out, this->shieldingExpression, model);
        }

        template<typename ValueType, typename IndexType>
        void PostShield<ValueType, IndexType>::printJsonToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> const& model) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidOperationException, "not supported yet");
        }


        // Explicitly instantiate appropriate classes
        template class PostShield<double, typename storm::storage::SparseMatrix<double>::index_type>;
#ifdef STORM_HAVE_CARL
        template class PostShield<storm::RationalNumber, typename storm::storage::SparseMatrix<storm::RationalNumber>::index_type>;
       // template class PostShield<storm::RationalFunction, typename storm::storage::SparseMatrix<storm::RationalFunction>::index_type>;

#endif
    }
}
