#include "storm/utility/vector.h"
#include "storm/storage/PostScheduler.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/NotImplementedException.h"
#include <boost/algorithm/string/join.hpp>

namespace storm {
    namespace storage {

        template <typename ValueType>
        PostScheduler<ValueType>::PostScheduler(uint_fast64_t numberOfModelStates, std::vector<uint_fast64_t> numberOfChoicesPerState, boost::optional<storm::storage::MemoryStructure> const& memoryStructure) : memoryStructure(memoryStructure) {
            //STORM_LOG_ASSERT(numberOfChoicesPerState.size() == numberOfModelStates, "Need to know amount of choices per model state");
            uint_fast64_t numOfMemoryStates = memoryStructure ? memoryStructure->getNumberOfStates() : 1;
            schedulerChoiceMapping = std::vector<std::vector<PostSchedulerChoice<ValueType>>>(numOfMemoryStates, std::vector<PostSchedulerChoice<ValueType>>(numberOfModelStates));
            numberOfChoices = 0;
            for(std::vector<uint_fast64_t>::iterator it = numberOfChoicesPerState.begin(); it != numberOfChoicesPerState.end(); ++it)
                numberOfChoices += *it;
            this->numOfUndefinedChoices = numOfMemoryStates * numberOfChoices;
            this->numOfDeterministicChoices = 0;
        }

        template <typename ValueType>
        PostScheduler<ValueType>::PostScheduler(uint_fast64_t numberOfModelStates, std::vector<uint_fast64_t> numberOfChoicesPerState, boost::optional<storm::storage::MemoryStructure>&& memoryStructure) : memoryStructure(std::move(memoryStructure)) {
            STORM_LOG_ASSERT(numberOfChoicesPerState.size() == numberOfModelStates, "Need to know amount of choices per model state");
            uint_fast64_t numOfMemoryStates = memoryStructure ? memoryStructure->getNumberOfStates() : 1;
            schedulerChoiceMapping = std::vector<std::vector<PostSchedulerChoice<ValueType>>>(numOfMemoryStates, std::vector<PostSchedulerChoice<ValueType>>(numberOfModelStates));
            numberOfChoices = 0;
            for(std::vector<uint_fast64_t>::iterator it = numberOfChoicesPerState.begin(); it != numberOfChoicesPerState.end(); ++it)
                numberOfChoices += *it;
            this->numOfUndefinedChoices = numOfMemoryStates * numberOfChoices;
            this->numOfDeterministicChoices = 0;
        }

        template <typename ValueType>
        void PostScheduler<ValueType>::setChoice(PostSchedulerChoice<ValueType> const& choice, uint_fast64_t modelState, uint_fast64_t memoryState) {
            STORM_LOG_ASSERT(memoryState == 0, "Currently we do not support PostScheduler with memory");
            STORM_LOG_ASSERT(modelState < schedulerChoiceMapping[memoryState].size(), "Illegal model state index");

            schedulerChoiceMapping[memoryState][modelState] = choice;
        }

        template <typename ValueType>
        PostSchedulerChoice<ValueType> const& PostScheduler<ValueType>::getChoice(uint_fast64_t modelState, uint_fast64_t memoryState) const {
            STORM_LOG_ASSERT(modelState < schedulerChoiceMapping[memoryState].size(), "Illegal model state index");
            return schedulerChoiceMapping[memoryState][modelState];
        }

        template <typename ValueType>
        bool PostScheduler<ValueType>::isDeterministicScheduler() const {
            return true;
        }

        template <typename ValueType>
        bool PostScheduler<ValueType>::isMemorylessScheduler() const {
            return true;
        }

        template <typename ValueType>
        void PostScheduler<ValueType>::printToStream(std::ostream& out, std::shared_ptr<storm::logic::ShieldExpression const> shieldingExpression, std::shared_ptr<storm::models::sparse::Model<ValueType>> model, bool skipUniqueChoices) const {
            STORM_LOG_THROW(this->isMemorylessScheduler(), storm::exceptions::InvalidOperationException, "The given scheduler is incompatible.");

            bool const stateValuationsGiven = model != nullptr && model->hasStateValuations();
            bool const choiceLabelsGiven = model != nullptr && model->hasChoiceLabeling();
            bool const choiceOriginsGiven = model != nullptr && model->hasChoiceOrigins();
            uint_fast64_t widthOfStates = std::to_string(schedulerChoiceMapping.front().size()).length();
            if (stateValuationsGiven) {
                widthOfStates += model->getStateValuations().getStateInfo(schedulerChoiceMapping.front().size() - 1).length() + 5;
            }
            widthOfStates = std::max(widthOfStates, (uint_fast64_t)12);
            out << "___________________________________________________________________" << std::endl;
            out << shieldingExpression->prettify() << std::endl;
            STORM_LOG_WARN_COND(!(skipUniqueChoices && model == nullptr), "Can not skip unique choices if the model is not given.");
            out << std::setw(widthOfStates) << "model state:" << "    " << "correction";
            if(choiceLabelsGiven) {
                out << " [<action> {action label}: <corrected action> {corrected action label}]";
            } else {
                out << " [<action>: (<corrected action>)}";
            }
            out << ":" << std::endl;
            uint_fast64_t numOfSkippedStatesWithUniqueChoice = 0;
            for (uint_fast64_t state = 0; state < schedulerChoiceMapping.front().size(); ++state) {
                PostSchedulerChoice<ValueType> const& choices = schedulerChoiceMapping[0][state];
                if(choices.isEmpty() && !printUndefinedChoices) continue;

                std::stringstream stateString;

                // Print the state info
                if (stateValuationsGiven) {
                    stateString << std::setw(widthOfStates)  << (std::to_string(state) + ": " + model->getStateValuations().getStateInfo(state));
                } else {
                    stateString << std::setw(widthOfStates) << state;
                }
                stateString << "    ";


                bool firstChoiceIndex = true;
                for(auto const& choiceMap : choices.getChoiceMap()) {
                    if(firstChoiceIndex) {
                        firstChoiceIndex = false;
                    } else {
                        stateString << ";    ";
                    }

                    if(choiceLabelsGiven) {
                        auto choiceLabels = model->getChoiceLabeling().getLabelsOfChoice(model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<0>(choiceMap));
                        stateString << std::to_string(std::get<0>(choiceMap)) << " {" << boost::join(choiceLabels, ", ") << "}: ";
                    } else {
                        stateString << std::to_string(std::get<0>(choiceMap)) << ": ";
                    }

                    if (choiceOriginsGiven) {
                        stateString << model->getChoiceOrigins()->getChoiceInfo(model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<1>(choiceMap));
                    } else {
                        stateString << std::to_string(std::get<1>(choiceMap));
                    }
                    if (choiceLabelsGiven) {
                        auto choiceLabels = model->getChoiceLabeling().getLabelsOfChoice(model->getTransitionMatrix().getRowGroupIndices()[state] + std::get<1>(choiceMap));
                        stateString << " {" << boost::join(choiceLabels, ", ") << "}";
                    }

                    // Todo: print memory updates
                }
                out << stateString.str() << std::endl;
                // jump to label if we find one undefined choice.
                //skipStatesWithUndefinedChoices:;
            }
            out << "___________________________________________________________________" << std::endl;
        }

        template <typename ValueType>
        void PostScheduler<ValueType>::printJsonToStream(std::ostream& out, std::shared_ptr<storm::models::sparse::Model<ValueType>> model, bool skipUniqueChoices) const {
            // STORM_LOG_THROW(model == nullptr || model->getNumberOfStates() == schedulerChoices.front().size(), storm::exceptions::InvalidOperationException, "The given model is not compatible with this scheduler.");
            // STORM_LOG_WARN_COND(!(skipUniqueChoices && model == nullptr), "Can not skip unique choices if the model is not given.");
            // storm::json<storm::RationalNumber> output;
            // for (uint64_t state = 0; state < schedulerChoices.front().size(); ++state) {
            //     // Check whether the state is skipped
            //     if (skipUniqueChoices && model != nullptr && model->getTransitionMatrix().getRowGroupSize(state) == 1) {
            //         continue;
            //     }

            //     for (uint_fast64_t memoryState = 0; memoryState < getNumberOfMemoryStates(); ++memoryState) {
            //         storm::json<storm::RationalNumber> stateChoicesJson;
            //         if (model && model->hasStateValuations()) {
            //             stateChoicesJson["s"] = model->getStateValuations().template toJson<storm::RationalNumber>(state);
            //         } else {
            //             stateChoicesJson["s"] = state;
            //         }

            //         if (!isMemorylessScheduler()) {
            //             stateChoicesJson["m"] = memoryState;
            //         }

            //         auto const &choice = schedulerChoices[memoryState][state];
            //         storm::json<storm::RationalNumber> choicesJson;

            //         for (auto const &choiceProbPair : choice.getChoiceMap()) {
            //             uint64_t globalChoiceIndex = model->getTransitionMatrix().getRowGroupIndices()[state] ;//+ choiceProbPair.first;
            //             storm::json<storm::RationalNumber> choiceJson;
            //             if (model && model->hasChoiceOrigins() &&
            //                 model->getChoiceOrigins()->getIdentifier(globalChoiceIndex) !=
            //                 model->getChoiceOrigins()->getIdentifierForChoicesWithNoOrigin()) {
            //                 choiceJson["origin"] = model->getChoiceOrigins()->getChoiceAsJson(globalChoiceIndex);
            //             }
            //             if (model && model->hasChoiceLabeling()) {
            //                 auto choiceLabels = model->getChoiceLabeling().getLabelsOfChoice(globalChoiceIndex);
            //                 choiceJson["labels"] = std::vector<std::string>(choiceLabels.begin(),
            //                                                                 choiceLabels.end());
            //             }
            //             choiceJson["index"] = globalChoiceIndex;
            //             choiceJson["prob"] = storm::utility::convertNumber<storm::RationalNumber>(
            //                     std::get<1>(choiceProbPair));

            //             // Memory updates
            //             if(!isMemorylessScheduler()) {
            //                 choiceJson["memory-updates"] = std::vector<storm::json<storm::RationalNumber>>();
            //                 uint64_t row = model->getTransitionMatrix().getRowGroupIndices()[state]; //+ std::get<0>(choiceProbPair);
            //                 for (auto entryIt = model->getTransitionMatrix().getRow(row).begin(); entryIt < model->getTransitionMatrix().getRow(row).end(); ++entryIt) {
            //                     storm::json<storm::RationalNumber> updateJson;
            //                     // next model state
            //                     if (model && model->hasStateValuations()) {
            //                         updateJson["s'"] = model->getStateValuations().template toJson<storm::RationalNumber>(entryIt->getColumn());
            //                     } else {
            //                         updateJson["s'"] = entryIt->getColumn();
            //                     }
            //                     // next memory state
            //                     updateJson["m'"] = this->memoryStructure->getSuccessorMemoryState(memoryState, entryIt - model->getTransitionMatrix().begin());
            //                     choiceJson["memory-updates"].push_back(std::move(updateJson));
            //                 }
            //             }

            //             choicesJson.push_back(std::move(choiceJson));
            //         }
            //         if (!choicesJson.is_null()) {
            //             stateChoicesJson["c"] = std::move(choicesJson);
            //             output.push_back(std::move(stateChoicesJson));
            //         }
            //     }
            // }
            // out << output.dump(4);
        }


        template class PostScheduler<double>;
#ifdef STORM_HAVE_CARL
        template class PostScheduler<storm::RationalNumber>;
#endif
    }
}
