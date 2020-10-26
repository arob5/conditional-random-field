import scala.math._
import breeze.linalg._
import breeze.optimize._

//
// Defines a Linear Chain Conditional Random Field
// Includes functionality for training and prediction
// Used in conjunction with Parser and Main objects
//

object LinearCRF {

	/////////////////
	// Basic Setup //
	////////////////
	val m = 6 // Number of feature functions

	type Label = String
	type Term = String

	val labels: List[Label] = List("NOUN", "VERB", "OTHER")


	///////////////////////
	// Feature Functions //
	//////////////////////

	// y(0) is y_(i-1) and y(1) is y_i
	def f1(y: Array[Label], x: Array[Term], i: Int) : Int = {
		return if ((x(i).length >= 2) && (x(i).takeRight(2) == "ed")) 1 else 0
	}

	def f2(y: Array[Label], x: Array[Term], i: Int) : Int = {
		return if (x(i).takeRight(1) == "s") 1 else 0
	}

	def f3(y: Array[Label], x: Array[Term], i: Int) : Int = {
		return if ((x(i).length >= 2) && (x(i).takeRight(2) == "ly")) 1 else 0	
	}
	
	def f4(y: Array[Label], x: Array[Term], i: Int) : Int = {
		return if (y(0) == "NOUN") 1 else 0
	}

	def f5(y: Array[Label], x: Array[Term], i: Int) : Int = {
		return if (y(0) == "VERB") 1 else 0
	}
	
	def f6(y: Array[Label], x: Array[Term], i: Int) : Int = {
		return if (x(i-1) == ".") 1 else 0
	}

	val f_list = List(f1(_: Array[Label], _: Array[Term], _: Int), f2(_: Array[Label], _: Array[Term], _: Int), 
					  f3(_: Array[Label], _: Array[Term], _: Int), f4(_: Array[Label], _: Array[Term], _: Int), 
					  f5(_: Array[Label], _: Array[Term], _: Int), f6(_: Array[Label], _: Array[Term], _: Int))


	////////////////
	// Parameters //
	////////////////

	/* Returns a random vector of size k */	
	def initialize_params(k: Int) : DenseVector[Double] = {
		return DenseVector.rand(k) 
	}


	//////////////////////////////////////////
	//Calculating Conditional Probabilities //
	//////////////////////////////////////////

	/* Returns product of factor functions (Does not include normalization term) */
	def weightedFeatures(y: Array[Label], x: Array[Term], theta: DenseVector[Double], k: Int) : Double = {
		val features = for(i <- 1 until k) yield {
			val inner_sum = for(j <- 0 until m) yield {
				theta(j) * f_list(j)(y.slice(i-1, i+1), x, i) 
			}	
			exp(inner_sum.reduceLeft(_+_))
		}
	
		features.reduceLeft(_*_)
		
	}
	
	/* Given x, sums over all possible labels; ensures probability is in [0, 1] */
	def normalization(x: Array[Term], theta: DenseVector[Double], k: Int) : Double = {
		val features = for(y1 <- labels; y2 <- labels) yield {
			weightedFeatures(Array(y1, y2), x, theta, k)
		}
		
		features.reduceLeft(_+_)

	}

	/* Returns p(y|x) */ 
	def probability(y: Array[Label], x: Array[Term], theta: DenseVector[Double], k: Int) : Double = {
		weightedFeatures(y, x, theta, k) / normalization(x, theta, k)
	}

	
	///////////////////////////////////
	//Maximum Likelihood Estimation //
	/////////////////////////////////

	/* Calculates conditional log likelihood, log p(y, x; theta) */
	def likelihood(X_train: List[Seq[Array[String]]], theta: DenseVector[Double], k: Int, n: Int, sigma_sq: Double) : Double = {
		val sum_over_n = for(i <- 0 until n) yield {
			val obs = Parser.parse_observation(X_train(i), k)
			val middle_sum = for(j <- 1 until k) yield {
				val inner_sum = for(l <- 0 until m) yield {
					theta(l) * f_list(l)(obs._1.slice(j-1,j+1), obs._2, j)
				}
				inner_sum.reduceLeft(_+_)
			}
			middle_sum.reduceLeft(_+_)
		}
		val triple_sum = sum_over_n.reduceLeft(_+_)

		val normalization_term = for(i <- 0 until n) yield {
			val obs = Parser.parse_observation(X_train(i), k)
			log(normalization(obs._2, theta, k))
		}	
		val norm_sum = normalization_term.reduceLeft(_+_)

		val regularization_term = for(i <- 0 until m) yield {
			theta(i) / (2.0 * sigma_sq)	
		} 
		val reg_sum = regularization_term.reduceLeft(_+_)
		
		return triple_sum - norm_sum - reg_sum
		
	}

	/* Calculates the gradient of the log likelihood */	
	def likelihoodGradient(X_train: List[Seq[Array[String]]], theta: DenseVector[Double], k: Int, n: Int, sigma_sq: Double) : DenseVector[Double] = {

		var grad_1 = DenseVector.zeros[Double](m)	
		for(i <- 0 until m) {
			val outer_sum = for(j <- 0 until n) yield {
				val obs = Parser.parse_observation(X_train(j), k) 
				val inner_sum = for(l <- 1 until k) yield {
					f_list(i)(obs._1.slice(l-1, l+1), obs._2, l)			
				}
				inner_sum.reduceLeft(_+_)
			}
			grad_1(i) = outer_sum.reduceLeft(_+_)
		}

		var grad_2 = DenseVector.zeros[Double](m)
		for(i <- 0 until m) {
			val over_n = for(j <- 0 until n) yield {
				val obs = Parser.parse_observation(X_train(j), k) 
				val over_k = for(l <- 1 until k) yield {
					val over_y = for(y1 <- labels; y2 <- labels; y3 <- labels) yield {
						var y_prime = Array[String](y1, y2, y3)
						f_list(i)(y_prime.slice(l-1, l+1), obs._2, l) * probability(y_prime, obs._2, theta, k)
					}
					over_y.reduceLeft(_+_)
				}	
				over_k.reduceLeft(_+_)
			}
			grad_2(i) = over_n.reduceLeft(_+_)
		}

		var grad_3 = DenseVector.zeros[Double](m)
		for(i <- 0 until m) {
			grad_3(i) = theta(i) / sigma_sq
		}	

		return grad_1 - grad_2 - grad_3

	}

	/* Returns a tuple holding the negative of the coditional log likelihood and its gradient (LBFGS only minimizes) */
	def getObjGrad(theta: DenseVector[Double], X_train: List[Seq[Array[String]]], k: Int, n: Int, sigma_sq: Double) : (Double, DenseVector[Double]) = {
		(-likelihood(X_train, theta, k, n, sigma_sq), -likelihoodGradient(X_train, theta, k, n, sigma_sq))
	}

	/* Maximizes the conditional log likelihood; returns optimal params and optimum value */ 
	def maximizeWithLBFGS(theta: DenseVector[Double], X_train: List[Seq[Array[String]]], k: Int, n: Int, sigma_sq: Double) = {
		val lbfgs = new LBFGS[DenseVector[Double]](maxIter=500, m=3)
		val f = new DiffFunction[DenseVector[Double]] {
			def calculate(theta: DenseVector[Double]) = {
				getObjGrad(theta, X_train, k, n, sigma_sq)
			}
		}
		val	optimum = lbfgs.minimize(f, theta)
		(optimum, f(optimum))
	}
	
	////////////////
	// Prediction //
	////////////////

	/* Calculates the prediction accuracy on a test set on a label by label basis */
	def prediction_accuracy(X_test: List[Seq[Array[String]]], theta: DenseVector[Double], k: Int) : Double = {
		var n_correct = 0d

		for(x <- X_test) {
			val obs = Parser.parse_observation(x, k)
		
			var max_prob = 0d
			var best_label = Array[String]()
			for(y1 <- labels; y2 <- labels; y3 <- labels) {
				val prob = probability(Array(y1, y2, y3), obs._2, theta, k)		
				if(prob > max_prob) {
					max_prob = prob
					best_label = Array[String](y2, y3)
				}		
			} 

			if(obs._1(1) == best_label(0))   				
				n_correct += 1.0
			if(obs._1(2) == best_label(1))
				n_correct += 1.0
		}		


		return return n_correct / (X_test.length * 2)
	}

}
