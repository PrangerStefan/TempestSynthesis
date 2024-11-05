#include "storm/utility/vector.h"
#include "storm/storage/PreScheduler.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/NotImplementedException.h"
#include <boost/algorithm/string/join.hpp>

namespace storm {
    namespace storage {
        template <typename ValueType>
        PreScheduler<ValueType>::PreScheduler(uint_fast64_t numberOfModelStates, boost::optional<storm::storage::MemoryStructure> const& memoryStructure) : memoryStructure(memoryStructure) {
            uint_fast64_t numOfMemoryStates = memoryStructure ? memoryStructure->getNumberOfStates() : 1;
            schedulerChoices = std::vector<std::vector<PreSchedulerChoice<ValueType>>>(numOfMemoryStates, std::vector<PreSchedulerChoice<ValueType>>(numberOfModelStates));
            //dontCareStates =  std::vector<storm::storage::BitVector>(numOfMemoryStates, storm::storage::BitVector(numberOfModelStates, false));
            numOfUndefinedChoices = numOfMemoryStates * numberOfModelStates;
            numOfDeterministicChoices = 0;
            numOfDontCareStates = 0;
        }

        template <typename ValueType>
        PreScheduler<ValueType>::PreScheduler(uint_fast64_t numberOfModelStates, boost::optional<storm::storage::MemoryStructure>&& memoryStructure) : memoryStructure(std::move(memoryStructure)) {
        }

        template <typename ValueType>
        bool PreScheduler<ValueType>::isMemorylessScheduler() const {
            return getNumberOfMemoryStates() == 1;
        }

        template <typename ValueType>
        uint_fast64_t PreScheduler<ValueType>::getNumberOfMemoryStates() const {
            return memoryStructure ? memoryStructure->getNumberOfStates() : 1;
        }

        template <typename ValueType>
        void PreScheduler<ValueType>::setChoice(PreSchedulerChoice<ValueType> const& choice, uint_fast64_t modelState, uint_fast64_t memoryState) {
            STORM_LOG_ASSERT(memoryState < this->getNumberOfMemoryStates(), "Illegal memory state index");
            STORM_LOG_ASSERT(modelState < this->schedulerChoices[memoryState].size(), "Illegal model state index");

            auto& schedulerChoice = schedulerChoices[memoryState][modelState];
            schedulerChoice = choice;
        }

        template <typename ValueType>
        PreSchedulerChoice<ValueType> const& PreScheduler<ValueType>::getChoice(uint_fast64_t modelState, uint_fast64_t memoryState) const {
            STORM_LOG_ASSERT(memoryState < getNumberOfMemoryStates(), "Illegal memory state index");
            STORM_LOG_ASSERT(modelState < schedulerChoices[memoryState].size(), "Illegal model state index");
            return schedulerChoices[memoryState][modelState];
        }


        template <typename ValueType>
        void PreScheduler<ValueType>::printToStream(std::ostream& out, std::shared_ptr<storm::logic::ShieldExpression const> shieldingExpression, std::shared_ptr<storm::models::sparse::Model<ValueType>> model, bool skipUniqueChoices) const {
            STORM_LOG_THROW(model == nullptr || model->getNumberOfStates() == this->schedulerChoices.front().size(), storm::exceptions::InvalidOperationException, "The given model is not compatible with this scheduler.");
            STORM_LOG_THROW(this->isMemorylessScheduler(), storm::exceptions::InvalidOperationException, "The given scheduler is incompatible.");

            bool const stateValuationsGiven = model != nullptr && model->hasStateValuations();
            bool const choiceLabelsGiven = model != nullptr && model->hasChoiceLabeling();
            bool const choiceOriginsGiven = model != nullptr && model->hasChoiceOrigins();
            uint_fast64_t widthOfStates = std::to_string(this->schedulerChoices.front().size()).length();
            if (stateValuationsGiven) {
                widthOfStates += model->getStateValuations().getStateInfo(this->schedulerChoices.front().size() - 1).length() + 5;
            }
            widthOfStates = std::max(widthOfStates, (uint_fast64_t)12);
            uint_fast64_t numOfSkippedStatesWithUniqueChoice = 0;

            out << "___________________________________________________________________" << std::endl;
            out << shieldingExpression->prettify() << std::endl;
            STORM_LOG_WARN_COND(!(skipUniqueChoices && model == nullptr), "Can not skip unique choices if the model is not given.");
            out << std::setw(widthOfStates) << "model state:" << "    " << "choice(s)";
            if(choiceLabelsGiven) {
                out << " [<value>: (<action {action label})>]";
            } else {
                out << " [<value>: (<action>)}";
            }
            out << ":" << std::endl;
            for (uint_fast64_t state = 0; state < this->schedulerChoices.front().size(); ++state) {
                std::stringstream stateString;
                // Check whether the state is skipped
                if (skipUniqueChoices && model != nullptr && model->getTransitionMatrix().getRowGroupSize(state) == 1) {
                    ++numOfSkippedStatesWithUniqueChoice;
                    continue;
                }

                // Print the state info
                if (stateValuationsGiven) {
                    stateString << std::setw(widthOfStates)  << (std::to_string(state) + ": " + model->getStateValuations().getStateInfo(state));
                } else {
                    stateString << std::setw(widthOfStates) << state;
                }
                stateString << "    ";

                bool firstMemoryState = true;
                for (uint_fast64_t memoryState = 0; memoryState < this->getNumberOfMemoryStates(); ++memoryState) {
                    // Indent if this is not the first memory state
                    if (firstMemoryState) {
                        firstMemoryState = false;
                    } else {
                        stateString << std::setw(widthOfStates) << "";
                        stateString << "    ";
                    }

                    // Print choice info
                    PreSchedulerChoice<ValueType> const& choices = this->schedulerChoices[memoryState][state];
                    if (!choices.isEmpty()) {
                        bool firstChoice = true;
                        for (auto const& choiceProbPair : choices.getChoiceMap()) {
                            if (firstChoice) {
                                firstChoice = false;
                            } else {
                                stateString << ";    ";
                            }
                            stateString << std::get<0>(choiceProbPair) << ": (";
                            if (choiceOriginsGiven) {
                                stateString << model->getChoiceOrigins()->getChoiceInfo(model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<1>(choiceProbPair));
                            } else {
                                stateString << std::get<1>(choiceProbPair);
                            }
                            if (choiceLabelsGiven) {
                                auto choiceLabels = model->getChoiceLabeling().getLabelsOfChoice(model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<1>(choiceProbPair));
                                stateString << " {" << boost::join(choiceLabels, ", ") << "}";
                            }
                            stateString << ")";
                        }
                    } else {
                        if(!this->printUndefinedChoices) continue;
                        stateString << "undefined.";
                    }

                    // Todo: print memory updates
                    out << stateString.str();
                    out << std::endl;
                }
            }
            if (numOfSkippedStatesWithUniqueChoice > 0) {
                out << "Skipped " << numOfSkippedStatesWithUniqueChoice << " deterministic states with unique choice." << std::endl;
            }
            out << "___________________________________________________________________" << std::endl;
        }




        template <typename ValueType>
        void PreScheduler<ValueType>::printJsonToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> model, bool skipUniqueChoices) const {
            STORM_LOG_THROW(model == nullptr || model->getNumberOfStates() == schedulerChoices.front().size(), storm::exceptions::InvalidOperationException, "The given model is not compatible with this scheduler.");
            STORM_LOG_WARN_COND(!(skipUniqueChoices && model == nullptr), "Can not skip unique choices if the model is not given.");
            storm::json<storm::RationalNumber> output;
            for (uint64_t state = 0; state < schedulerChoices.front().size(); ++state) {
                // Check whether the state is skipped
                if (skipUniqueChoices && model != nullptr && model->getTransitionMatrix().getRowGroupSize(state) == 1) {
                    continue;
                }

                for (uint_fast64_t memoryState = 0; memoryState < getNumberOfMemoryStates(); ++memoryState) {
                    storm::json<storm::RationalNumber> stateChoicesJson;
                    if (model && model->hasStateValuations()) {
                        stateChoicesJson["s"] = model->getStateValuations().template toJson<storm::RationalNumber>(state);
                    } else {
                        stateChoicesJson["s"] = state;
                    }

                    if (!isMemorylessScheduler()) {
                        stateChoicesJson["m"] = memoryState;
                    }

                    auto const &choice = schedulerChoices[memoryState][state];
                    storm::json<storm::RationalNumber> choicesJson;
                    if (!choice.getChoiceMap().empty()) {
                        for (auto const &choiceProbPair : choice.getChoiceMap()) {
                            uint64_t globalChoiceIndex = model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<uint_fast64_t>(choiceProbPair);
                            storm::json<storm::RationalNumber> choiceJson;
                            if (model && model->hasChoiceOrigins() &&
                                model->getChoiceOrigins()->getIdentifier(globalChoiceIndex) !=
                                model->getChoiceOrigins()->getIdentifierForChoicesWithNoOrigin()) {
                                auto choiceOriginJson = model->getChoiceOrigins()->getChoiceAsJson(globalChoiceIndex);
                                std::string choiceActionLabel = choiceOriginJson["action-label"];
                                choiceOriginJson["action-label"] = choiceActionLabel.append("\n");
                                choiceJson["origin"] = choiceOriginJson;
                            }
                            if (model && model->hasChoiceLabeling()) {
                                auto choiceLabels = model->getChoiceLabeling().getLabelsOfChoice(globalChoiceIndex);
                                choiceJson["labels"] = std::vector<std::string>(choiceLabels.begin(),
                                                                                choiceLabels.end());
                            }
                            choiceJson["index"] = globalChoiceIndex;
                            choiceJson["prob"] = storm::utility::convertNumber<storm::RationalNumber>(
                                    std::get<ValueType>(choiceProbPair));

                            // Memory updates
                            if(!isMemorylessScheduler()) {
                                choiceJson["memory-updates"] = std::vector<storm::json<storm::RationalNumber>>();
                                uint64_t row = model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<uint_fast64_t>(choiceProbPair);
                                for (auto entryIt = model->getTransitionMatrix().getRow(row).begin(); entryIt < model->getTransitionMatrix().getRow(row).end(); ++entryIt) {
                                    storm::json<storm::RationalNumber> updateJson;
                                    // next model state
                                    if (model && model->hasStateValuations()) {
                                        updateJson["s'"] = model->getStateValuations().template toJson<storm::RationalNumber>(entryIt->getColumn());
                                    } else {
                                        updateJson["s'"] = entryIt->getColumn();
                                    }
                                    // next memory state
                                    updateJson["m'"] = this->memoryStructure->getSuccessorMemoryState(memoryState, entryIt - model->getTransitionMatrix().begin());
                                    choiceJson["memory-updates"].push_back(std::move(updateJson));
                                }
                            }

                            choicesJson.push_back(std::move(choiceJson));
                        }
                    } else {
                        choicesJson = "undefined";
                    }

                    stateChoicesJson["c"] = std::move(choicesJson);
                    output.push_back(std::move(stateChoicesJson));
                }
            }
            out << output.dump(4);
        }




        template class PreScheduler<double>;
#ifdef STORM_HAVE_CARL
        template class PreScheduler<storm::RationalNumber>;
        template class PreScheduler<storm::RationalFunction>;
#endif
    }
}
