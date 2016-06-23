#ifndef STORM_SOLVER_ELIMINATIONLINEAREQUATIONSOLVER_H_
#define STORM_SOLVER_ELIMINATIONLINEAREQUATIONSOLVER_H_

#include "src/solver/LinearEquationSolver.h"

#include "src/settings/modules/EliminationSettings.h"

namespace storm {
    namespace solver {
        template<typename ValueType>
        class EliminationLinearEquationSolverSettings {
        public:
            EliminationLinearEquationSolverSettings();
            
            void setEliminationOrder(storm::settings::modules::EliminationSettings::EliminationOrder const& order);
            
            storm::settings::modules::EliminationSettings::EliminationOrder getEliminationOrder() const;

        private:
            storm::settings::modules::EliminationSettings::EliminationOrder order;
        };
        
        /*!
         * A class that uses gaussian elimination to implement the LinearEquationSolver interface.
         */
        template<typename ValueType>
        class EliminationLinearEquationSolver : public LinearEquationSolver<ValueType> {
        public:
            EliminationLinearEquationSolver(storm::storage::SparseMatrix<ValueType> const& A, EliminationLinearEquationSolverSettings<ValueType> const& settings = EliminationLinearEquationSolverSettings<ValueType>());
            EliminationLinearEquationSolver(storm::storage::SparseMatrix<ValueType>&& A, EliminationLinearEquationSolverSettings<ValueType> const& settings = EliminationLinearEquationSolverSettings<ValueType>());
            
            virtual void solveEquationSystem(std::vector<ValueType>& x, std::vector<ValueType> const& b, std::vector<ValueType>* multiplyResult = nullptr) const override;
            
            virtual void performMatrixVectorMultiplication(std::vector<ValueType>& x, std::vector<ValueType> const* b, uint_fast64_t n = 1, std::vector<ValueType>* multiplyResult = nullptr) const override;

            EliminationLinearEquationSolverSettings<ValueType>& getSettings();
            EliminationLinearEquationSolverSettings<ValueType> const& getSettings() const;
            
        private:
            void initializeSettings();
            
            // If the solver takes posession of the matrix, we store the moved matrix in this member, so it gets deleted
            // when the solver is destructed.
            std::unique_ptr<storm::storage::SparseMatrix<ValueType>> localA;

            // A reference to the original sparse matrix given to this solver. If the solver takes posession of the matrix
            // the reference refers to localA.
            storm::storage::SparseMatrix<ValueType> const& A;
            
            // The settings used by the solver.
            EliminationLinearEquationSolverSettings<ValueType> settings;
        };
        
        template<typename ValueType>
        class EliminationLinearEquationSolverFactory : public LinearEquationSolverFactory<ValueType> {
        public:
            virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
            virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const override;
            
            EliminationLinearEquationSolverSettings<ValueType>& getSettings();
            EliminationLinearEquationSolverSettings<ValueType> const& getSettings() const;
            
        private:
            EliminationLinearEquationSolverSettings<ValueType> settings;
        };
    }
}

#endif /* STORM_SOLVER_ELIMINATIONLINEAREQUATIONSOLVER_H_ */