#pragma once

#include "storm/shields/AbstractShield.h"
#include "storm/storage/PostScheduler.h"

namespace tempest {
    namespace shields {

        template<typename ValueType, typename IndexType>
        class PostShield : public AbstractShield<ValueType, IndexType> {
        public:
            PostShield(std::vector<IndexType> const& rowGroupIndices, std::vector<ValueType> const& choiceValues, std::shared_ptr<storm::logic::ShieldExpression const> const& shieldingExpression, storm::OptimizationDirection optimizationDirection, storm::storage::BitVector relevantStates, boost::optional<storm::storage::BitVector> coalitionStates);

            storm::storage::PostScheduler<ValueType> construct();
            template<typename Compare, bool relative>
            storm::storage::PostScheduler<ValueType> constructWithCompareType();

            virtual void printToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> const& model) override;
            virtual void printJsonToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> const& model) override;

        private:
            std::vector<ValueType> choiceValues;
        };
    }
}
