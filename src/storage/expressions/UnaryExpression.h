#ifndef STORM_STORAGE_EXPRESSIONS_UNARYEXPRESSION_H_
#define STORM_STORAGE_EXPRESSIONS_UNARYEXPRESSION_H_

#include "src/storage/expressions/BaseExpression.h"
#include "src/utility/OsDetection.h"

namespace storm {
    namespace expressions {
        class UnaryExpression : public BaseExpression {
        public:
            /*!
             * Creates a unary expression with the given return type and operand.
             *
             * @param returnType The return type of the expression.
             * @param operand The operand of the unary expression.
             */
            UnaryExpression(ExpressionReturnType returnType, std::shared_ptr<BaseExpression const> const& operand);

            // Instantiate constructors and assignments with their default implementations.
            UnaryExpression(UnaryExpression const& other);
            UnaryExpression& operator=(UnaryExpression const& other);
#ifndef WINDOWS
            UnaryExpression(UnaryExpression&&) = default;
            UnaryExpression& operator=(UnaryExpression&&) = default;
#endif
            virtual ~UnaryExpression() = default;
            
            // Override base class methods.
            virtual bool containsVariables() const override;
            virtual uint_fast64_t getArity() const override;
            virtual std::shared_ptr<BaseExpression const> getOperand(uint_fast64_t operandIndex) const override;
            virtual std::set<std::string> getVariables() const override;
            
            /*!
             * Retrieves the operand of the unary expression.
             *
             * @return The operand of the unary expression.
             */
            std::shared_ptr<BaseExpression const> const& getOperand() const;
            
        private:
            // The operand of the unary expression.
            std::shared_ptr<BaseExpression const> operand;
        };
    }
}

#endif /* STORM_STORAGE_EXPRESSIONS_UNARYEXPRESSION_H_ */