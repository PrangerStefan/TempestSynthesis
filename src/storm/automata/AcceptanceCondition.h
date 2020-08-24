#pragma once

#include <functional>
#include <vector>
#include <memory>
#include "storm/storage/BitVector.h"
#include "storm/storage/StateBlock.h"
#include "cpphoafparser/consumer/hoa_consumer.hh"

namespace storm {
    namespace automata {
        class AcceptanceCondition {
        public:
            typedef std::shared_ptr<AcceptanceCondition> ptr;
            typedef cpphoafparser::HOAConsumer::acceptance_expr acceptance_expr;

            AcceptanceCondition(std::size_t numberOfStates, unsigned int numberOfAcceptanceSets, acceptance_expr::ptr acceptance);
            bool isAccepting(const storm::storage::StateBlock& scc) const ;

            unsigned int getNumberOfAcceptanceSets() const;
            storm::storage::BitVector& getAcceptanceSet(unsigned int index);
            const storm::storage::BitVector& getAcceptanceSet(unsigned int index) const;

            acceptance_expr::ptr getAcceptanceExpression() const;

            AcceptanceCondition::ptr lift(std::size_t productNumberOfStates, std::function<std::size_t (std::size_t)> mapping) const;
        private:
            bool isAccepting(const storm::storage::StateBlock& scc, acceptance_expr::ptr expr) const;

            std::size_t numberOfStates;
            unsigned int numberOfAcceptanceSets;
            acceptance_expr::ptr acceptance;
            std::vector<storm::storage::BitVector> acceptanceSets;
        };
    }
}
