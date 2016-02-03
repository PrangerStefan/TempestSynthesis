#ifndef STORM_MODELCHECKER_SPARSE_CTMC_CSL_MODELCHECKER_HELPER_H_
#define STORM_MODELCHECKER_SPARSE_CTMC_CSL_MODELCHECKER_HELPER_H_

#include "src/storage/BitVector.h"

#include "src/utility/solver.h"

namespace storm {
    namespace modelchecker {
        namespace helper {
            template <typename ValueType>
            class SparseCtmcCslHelper {
            public:
                static std::vector<ValueType> computeBoundedUntilProbabilities(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, std::vector<ValueType> const& exitRates, bool qualitative, double lowerBound, double upperBound, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);

                static std::vector<ValueType> computeNextProbabilities(storm::storage::SparseMatrix<ValueType> const& rateMatrix, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& nextStates, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                
                static std::vector<ValueType> computeUntilProbabilities(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                static std::vector<ValueType> computeUntilProbabilitiesElimination(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& initialStates, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative);
                
                template <typename RewardModelType>
                static std::vector<ValueType> computeInstantaneousRewards(storm::storage::SparseMatrix<ValueType> const& rateMatrix, std::vector<ValueType> const& exitRateVector, RewardModelType const& rewardModel, double timeBound, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                
                template <typename RewardModelType>
                static std::vector<ValueType> computeCumulativeRewards(storm::storage::SparseMatrix<ValueType> const& rateMatrix, std::vector<ValueType> const& exitRateVector, RewardModelType const& rewardModel, double timeBound, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                
                template <typename RewardModelType>
                static std::vector<ValueType> computeReachabilityRewards(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, RewardModelType const& rewardModel, storm::storage::BitVector const& targetStates, bool qualitative, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                
                static std::vector<ValueType> computeLongRunAverageProbabilities(storm::storage::SparseMatrix<ValueType> const& probabilityMatrix, storm::storage::BitVector const& psiStates, std::vector<ValueType> const* exitRateVector, bool qualitative, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                
                static std::vector<ValueType> computeExpectedTimes(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& initialStates, storm::storage::BitVector const& targetStates, bool qualitative, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& minMaxLinearEquationSolverFactory);
                static std::vector<ValueType> computeExpectedTimesElimination(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, std::vector<ValueType> const& exitRateVector, storm::storage::BitVector const& initialStates, storm::storage::BitVector const& targetStates, bool qualitative);

                /*!
                 * Computes the matrix representing the transitions of the uniformized CTMC.
                 *
                 * @param transitionMatrix The matrix to uniformize.
                 * @param maybeStates The states that need to be considered.
                 * @param uniformizationRate The rate to be used for uniformization.
                 * @param exitRates The exit rates of all states.
                 * @return The uniformized matrix.
                 */
                static storm::storage::SparseMatrix<ValueType> computeUniformizedMatrix(storm::storage::SparseMatrix<ValueType> const& rateMatrix, storm::storage::BitVector const& maybeStates, ValueType uniformizationRate, std::vector<ValueType> const& exitRates);
                
                /*!
                 * Computes the transient probabilities for lambda time steps.
                 *
                 * @param uniformizedMatrix The uniformized transition matrix.
                 * @param addVector A vector that is added in each step as a possible compensation for removing absorbing states
                 * with a non-zero initial value. If this is not supposed to be used, it can be set to nullptr.
                 * @param timeBound The time bound to use.
                 * @param uniformizationRate The used uniformization rate.
                 * @param values A vector mapping each state to an initial probability.
                 * @param linearEquationSolverFactory The factory to use when instantiating new linear equation solvers.
                 * @param useMixedPoissonProbabilities If set to true, instead of taking the poisson probabilities,  mixed
                 * poisson probabilities are used.
                 * @return The vector of transient probabilities.
                 */
                template<bool useMixedPoissonProbabilities = false>
                static std::vector<ValueType> computeTransientProbabilities(storm::storage::SparseMatrix<ValueType> const& uniformizedMatrix, std::vector<ValueType> const* addVector, ValueType timeBound, ValueType uniformizationRate, std::vector<ValueType> values, storm::utility::solver::LinearEquationSolverFactory<ValueType> const& linearEquationSolverFactory);
                
                /*!
                 * Converts the given rate-matrix into a time-abstract probability matrix.
                 *
                 * @param rateMatrix The rate matrix.
                 * @param exitRates The exit rate vector.
                 * @return The †ransition matrix of the embedded DTMC.
                 */
                static storm::storage::SparseMatrix<ValueType> computeProbabilityMatrix(storm::storage::SparseMatrix<ValueType> const& rateMatrix, std::vector<ValueType> const& exitRates);
                
                /*!
                 * Converts the given rate-matrix into the generator matrix.
                 *
                 * @param rateMatrix The rate matrix.
                 * @param exitRates The exit rate vector.
                 * @return The generator matrix.
                 */
                static storm::storage::SparseMatrix<ValueType> computeGeneratorMatrix(storm::storage::SparseMatrix<ValueType> const& rateMatrix, std::vector<ValueType> const& exitRates);
            };
        }
    }
}

#endif /* STORM_MODELCHECKER_SPARSE_CTMC_CSL_MODELCHECKER_HELPER_H_ */