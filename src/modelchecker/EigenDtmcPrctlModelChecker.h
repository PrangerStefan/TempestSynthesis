/*
 * EigenDtmcPrctlModelChecker.h
 *
 *  Created on: 07.12.2012
 *      Author: 
 */

#ifndef STORM_MODELCHECKER_EIGENDTMCPRCTLMODELCHECKER_H_
#define STORM_MODELCHECKER_EIGENDTMCPRCTLMODELCHECKER_H_

#include "src/utility/Vector.h"

#include "src/models/Dtmc.h"
#include "src/modelchecker/DtmcPrctlModelChecker.h"
#include "src/utility/GraphAnalyzer.h"
#include "src/utility/ConstTemplates.h"
#include "src/exceptions/NoConvergenceException.h"

#include "Eigen/Sparse"
#include "Eigen/src/IterativeLinearSolvers/BiCGSTAB.h"
#include "src/adapters/EigenAdapter.h"

#include "gmm/gmm_matrix.h"
#include "gmm/gmm_iter_solvers.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

extern log4cplus::Logger logger;

namespace storm {

namespace modelChecker {

/*
 * A model checking engine that makes use of the eigen backend.
 */
template <class Type>
class EigenDtmcPrctlModelChecker : public DtmcPrctlModelChecker<Type> {

public:
	explicit EigenDtmcPrctlModelChecker(storm::models::Dtmc<Type>& dtmc) : DtmcPrctlModelChecker<Type>(dtmc) { }

	virtual ~EigenDtmcPrctlModelChecker() { }

	virtual std::vector<Type>* checkBoundedUntil(const storm::formula::BoundedUntil<Type>& formula) const {
		// First, we need to compute the states that satisfy the sub-formulas of the until-formula.
		storm::storage::BitVector* leftStates = this->checkStateFormula(formula.getLeft());
		storm::storage::BitVector* rightStates = this->checkStateFormula(formula.getRight());

		// Copy the matrix before we make any changes.
		storm::storage::SparseMatrix<Type> tmpMatrix(*this->getModel().getTransitionProbabilityMatrix());

		// Make all rows absorbing that violate both sub-formulas or satisfy the second sub-formula.
		tmpMatrix.makeRowsAbsorbing((~*leftStates | *rightStates) | *rightStates);

		// Transform the transition probability matrix to the eigen format to use its arithmetic.
		Eigen::SparseMatrix<Type, 1, int_fast32_t>* eigenMatrix = storm::adapters::EigenAdapter::toEigenSparseMatrix(tmpMatrix);

		// Create the vector with which to multiply.
		uint_fast64_t stateCount = this->getModel().getNumberOfStates();

		typedef Eigen::Matrix<Type, -1, 1, 0, -1, 1> VectorType;
		typedef Eigen::Map<VectorType> MapType;

		std::vector<Type>* result = new std::vector<Type>(stateCount);
		
		storm::utility::setVectorValues(result, *rightStates, storm::utility::constGetOne<Type>());

		Type *p = &((*result)[0]); // get the address storing the data for result
		MapType vectorMap(p, result->size()); // vectorMap shares data 
		

		// Now perform matrix-vector multiplication as long as we meet the bound of the formula.
		for (uint_fast64_t i = 0, bound = formula.getBound(); i < bound; ++i) {
			vectorMap = (*eigenMatrix) * vectorMap;
		}

		// Delete intermediate results.
		delete leftStates;
		delete rightStates;
		delete eigenMatrix;

		return result;
	}

	virtual std::vector<Type>* checkNext(const storm::formula::Next<Type>& formula) const {
		// First, we need to compute the states that satisfy the sub-formula of the next-formula.
		storm::storage::BitVector* nextStates = this->checkStateFormula(formula.getChild());

		// Transform the transition probability matrix to the eigen format to use its arithmetic.
		Eigen::SparseMatrix<Type, 1, int_fast32_t>* eigenMatrix = storm::adapters::EigenAdapter::toEigenSparseMatrix(this->getModel().getTransitionProbabilityMatrix());

		// Create the vector with which to multiply and initialize it correctly.
		std::vector<Type> x(this->getModel().getNumberOfStates());

		storm::utility::setVectorValues(&x, *nextStates, storm::utility::constGetOne<Type>());

		// Delete not needed next states bit vector.
		delete nextStates;

		typedef Eigen::Matrix<Type, -1, 1, 0, -1, 1> VectorType;
		typedef Eigen::Map<VectorType> MapType;

		Type *px = &(x[0]); // get the address storing the data for x
		MapType vectorX(px, x.size()); // vectorX shares data 

		// Create resulting vector.
		std::vector<Type>* result = new std::vector<Type>(this->getModel().getNumberOfStates());

		// Type *pr = &((*result)[0]); // get the address storing the data for result
		MapType vectorResult(px, result->size()); // vectorResult shares data 

		// Perform the actual computation.
		vectorResult = (*eigenMatrix) * vectorX;

		// Delete temporary matrix and return result.
		delete eigenMatrix;
		return result;
	}

	virtual std::vector<Type>* checkUntil(const storm::formula::Until<Type>& formula) const {
		// First, we need to compute the states that satisfy the sub-formulas of the until-formula.
		storm::storage::BitVector* leftStates = this->checkStateFormula(formula.getLeft());
		storm::storage::BitVector* rightStates = this->checkStateFormula(formula.getRight());

		// Then, we need to identify the states which have to be taken out of the matrix, i.e.
		// all states that have probability 0 and 1 of satisfying the until-formula.
		storm::storage::BitVector statesWithProbability0(this->getModel().getNumberOfStates());
		storm::storage::BitVector statesWithProbability1(this->getModel().getNumberOfStates());
		storm::utility::GraphAnalyzer::performProb01(this->getModel(), *leftStates, *rightStates, &statesWithProbability0, &statesWithProbability1);

		delete leftStates;
		delete rightStates;

		LOG4CPLUS_INFO(logger, "Found " << statesWithProbability0.getNumberOfSetBits() << " 'no' states.");
		LOG4CPLUS_INFO(logger, "Found " << statesWithProbability1.getNumberOfSetBits() << " 'yes' states.");
		storm::storage::BitVector maybeStates = ~(statesWithProbability0 | statesWithProbability1);
		LOG4CPLUS_INFO(logger, "Found " << maybeStates.getNumberOfSetBits() << " 'maybe' states.");

		// Create resulting vector and set values accordingly.
		uint_fast64_t stateCount = this->getModel().getNumberOfStates();
		std::vector<Type>* result = new std::vector<Type>(stateCount);

		// Only try to solve system if there are states for which the probability is unknown.
		if (maybeStates.getNumberOfSetBits() > 0) {
			typedef Eigen::Matrix<Type, -1, 1, 0, -1, 1> VectorType;
			typedef Eigen::Map<VectorType> MapType;

			// Now we can eliminate the rows and columns from the original transition probability matrix.
			storm::storage::SparseMatrix<double>* submatrix = this->getModel().getTransitionProbabilityMatrix()->getSubmatrix(maybeStates);
			// Converting the matrix to the form needed for the equation system. That is, we go from
			// x = A*x + b to (I-A)x = b.
			submatrix->convertToEquationSystem();

			// Transform the submatric matrix to the eigen format to use its solvers
			Eigen::SparseMatrix<Type, 1, int_fast32_t>* eigenSubMatrix = storm::adapters::EigenAdapter::toEigenSparseMatrix<Type>(submatrix);
			delete submatrix;

			// Initialize the x vector with 0.5 for each element. This is the initial guess for
			// the iterative solvers. It should be safe as for all 'maybe' states we know that the
			// probability is strictly larger than 0.
			std::vector<Type> x(maybeStates.getNumberOfSetBits(), Type(0.5));

			// Map for x
			Type *px = &(x[0]); // get the address storing the data for x
			MapType vectorX(px, x.size()); // vectorX shares data 


			// Prepare the right-hand side of the equation system. For entry i this corresponds to
			// the accumulated probability of going from state i to some 'yes' state.
			std::vector<double> b(maybeStates.getNumberOfSetBits());

			Type *pb = &(b[0]); // get the address storing the data for b
			MapType vectorB(pb, b.size()); // vectorB shares data 

			this->getModel().getTransitionProbabilityMatrix()->getConstrainedRowCountVector(maybeStates, statesWithProbability1, &x);

			Eigen::BiCGSTAB<Eigen::SparseMatrix<Type, 1, int_fast32_t>> solver;
			solver.compute(*eigenSubMatrix);
			if(solver.info()!= Eigen::ComputationInfo::Success) {
				// decomposition failed
				LOG4CPLUS_ERROR(logger, "Decomposition of Submatrix failed!");
			}

			// Now do the actual solving.
			LOG4CPLUS_INFO(logger, "Starting iterative solver.");
			
			solver.setTolerance(0.000001);
			
			vectorX = solver.solveWithGuess(vectorB, vectorX);

			if(solver.info() == Eigen::ComputationInfo::InvalidInput) {
				// solving failed
				LOG4CPLUS_ERROR(logger, "Solving of Submatrix failed: InvalidInput");
			} else if(solver.info() == Eigen::ComputationInfo::NoConvergence) {
				// NoConvergence
				throw storm::exceptions::NoConvergenceException("Solving of Submatrix with Eigen failed", solver.iterations(), solver.maxIterations());
			} else if(solver.info() == Eigen::ComputationInfo::NumericalIssue) {
				// NumericalIssue
				LOG4CPLUS_ERROR(logger, "Solving of Submatrix failed: NumericalIssue");
			} else if(solver.info() == Eigen::ComputationInfo::Success) {
				// solving Success
				LOG4CPLUS_INFO(logger, "Solving of Submatrix succeeded: Success");
			} 

			// Set values of resulting vector according to result.
			storm::utility::setVectorValues<Type>(result, maybeStates, x);

			// Delete temporary matrix.
			delete eigenSubMatrix;
		}

		storm::utility::setVectorValues<Type>(result, statesWithProbability0, storm::utility::constGetZero<Type>());
		storm::utility::setVectorValues<Type>(result, statesWithProbability1, storm::utility::constGetOne<Type>());

		return result;
	}
};

} //namespace modelChecker

} //namespace storm

#endif /* STORM_MODELCHECKER_EIGENDTMCPRCTLMODELCHECKER_H_ */