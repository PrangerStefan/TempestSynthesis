//
// Created by Jip Spel on 24.09.18.
//

#ifndef STORM_VALUETYPETOEXPRESSION_H
#define STORM_VALUETYPETOEXPRESSION_H

#include "storm/storage/expressions/ExpressionManager.h"
#include "storm/storage/expressions/Expression.h"

namespace storm {
    namespace expressions {
        template<typename ValueType>
        class ValueTypeToExpression {
        public:
            ValueTypeToExpression(std::shared_ptr<ExpressionManager> manager);

            /*!
            * Retrieves the manager responsible for the variables of this valuation.
            *
            * @return The pointer to the manager.
            */
            std::shared_ptr<ExpressionManager> getManager();

            /*!
             * Transforms the function into an expression.
             *
             * @param function The function to transform
             * @return The created expression.
             */
            Expression toExpression(ValueType function);

        private:
            // The manager responsible for the variables of this valuation.
            std::shared_ptr<ExpressionManager> manager;
        };
    }
}

#endif //STORM_VALUETYPETOEXPRESSION_H
