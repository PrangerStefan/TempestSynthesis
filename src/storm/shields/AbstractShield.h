#pragma once

#include <boost/optional.hpp>
#include <iostream>
#include <string>
#include <memory>

#include "storm/storage/Scheduler.h"
#include "storm/storage/SchedulerChoice.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/Distribution.h"

#include "storm/utility/constants.h"

#include "storm/solver/OptimizationDirection.h"

#include "storm/logic/ShieldExpression.h"

#include "storm/exceptions/NotSupportedException.h"


namespace tempest {
    namespace shields {
        template<typename ValueType, typename IndexType>
        class PreShield;
        template<typename ValueType, typename IndexType>
        class PostShield;
        template<typename ValueType, typename IndexType>
        class OptimalShield;

        namespace utility {
            template<typename ValueType, typename Compare, bool relative>
            struct ChoiceFilter {
                bool operator()(ValueType v, ValueType opt, double shieldValue) {
                    if constexpr (std::is_same_v<ValueType, storm::RationalNumber> || std::is_same_v<ValueType, double>) {
                        Compare compare;
                        if(relative && std::is_same<Compare, storm::utility::ElementLessEqual<ValueType>>::value) {
                            return compare(v, opt + opt * shieldValue);
                        } else if(relative && std::is_same<Compare, storm::utility::ElementGreaterEqual<ValueType>>::value) {
                            return compare(v, opt * shieldValue);
                        }
                        else return compare(v, shieldValue);
                    } else {
                        STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Cannot create shields for parametric models");
                    }
                }
            };
        }

        template<typename ValueType, typename IndexType>
        class AbstractShield {
        public:
            typedef IndexType index_type;
            typedef ValueType value_type;

            virtual ~AbstractShield() = 0;

            /*!
             * Computes the sizes of the row groups based on the indices given.
             */
            std::vector<IndexType> computeRowGroupSizes();

            storm::OptimizationDirection getOptimizationDirection();
            void setShieldingExpression(std::shared_ptr<storm::logic::ShieldExpression const> const& shieldingExpression);

            std::string getClassName() const;

            virtual bool isPreShield() const;
            virtual bool isPostShield() const;
            virtual bool isOptimalShield() const;

            PreShield<ValueType, IndexType>& asPreShield();
            PreShield<ValueType, IndexType> const& asPreShield() const;

            PostShield<ValueType, IndexType>& asPostShield();
            PostShield<ValueType, IndexType> const& asPostShield() const;

            OptimalShield<ValueType, IndexType>& asOptimalShield();
            OptimalShield<ValueType, IndexType> const& asOptimalShield() const;


            virtual void printToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> const& model) = 0;
            virtual void printJsonToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> const& model) = 0;


        protected:
            AbstractShield(std::vector<IndexType> const& rowGroupIndices, std::shared_ptr<storm::logic::ShieldExpression const> const& shieldingExpression, storm::OptimizationDirection optimizationDirection, storm::storage::BitVector relevantStates, boost::optional<storm::storage::BitVector> coalitionStates);

            std::vector<index_type> rowGroupIndices;
            //std::vector<value_type> choiceValues;

            std::shared_ptr<storm::logic::ShieldExpression const> shieldingExpression;
            storm::OptimizationDirection optimizationDirection;

            storm::storage::BitVector relevantStates;
            boost::optional<storm::storage::BitVector> coalitionStates;
        };
    }
}
