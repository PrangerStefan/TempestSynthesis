#include "Property.h"

namespace storm {
    namespace jani {
        
        std::ostream& operator<<(std::ostream& os, FilterExpression const& fe) {
            return os << "Obtain " << toString(fe.getFilterType()) << " of the '" << *fe.getStatesFormula() << "'-states with values described by '" << *fe.getFormula() << "'";
        }
        
        Property::Property(std::string const& name, std::shared_ptr<storm::logic::Formula const> const& formula, std::set<storm::expressions::Variable> const& undefinedConstants, std::string const& comment)
        : name(name), comment(comment), filterExpression(FilterExpression(formula)), undefinedConstants(undefinedConstants) {
            // Intentionally left empty.
        }
        
        Property::Property(std::string const& name, FilterExpression const& fe, std::set<storm::expressions::Variable> const& undefinedConstants, std::string const& comment)
        : name(name), comment(comment), filterExpression(fe), undefinedConstants(undefinedConstants) {
            // Intentionally left empty.
        }

        std::string const& Property::getName() const {
            return this->name;
        }

        std::string const& Property::getComment() const {
            return this->comment;
        }
        
        Property Property::substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution) const {
            std::set<storm::expressions::Variable> remainingUndefinedConstants;
            for (auto const& constant : undefinedConstants) {
                if (substitution.find(constant) == substitution.end()) {
                    remainingUndefinedConstants.insert(constant);
                }
            }
            return Property(name, filterExpression.substitute(substitution), remainingUndefinedConstants, comment);
        }
        
        Property Property::substituteLabels(std::map<std::string, std::string> const& substitution) const {
            return Property(name, filterExpression.substituteLabels(substitution), undefinedConstants, comment);
        }
        
        FilterExpression const& Property::getFilter() const {
            return this->filterExpression;
        }
        
        std::shared_ptr<storm::logic::Formula const> Property::getRawFormula() const {
            return this->filterExpression.getFormula();
        }
        
        std::set<storm::expressions::Variable> const& Property::getUndefinedConstants() const {
            return undefinedConstants;
        }
        
        bool Property::containsUndefinedConstants() const {
            return !undefinedConstants.empty();
        }
        
        std::ostream& operator<<(std::ostream& os, Property const& p) {
            return os << "(" << p.getName() << "): " << p.getFilter();
        }
        
    }
}
