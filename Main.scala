import breeze.linalg._

//
// Main object for training and testing CRF
//

object Main extends App {

	// SETUP
	val k = 3
	val m = 6
	val n = 5000
	val n_test = 1000
	val sigma_sq = 10d
	val datafile = "test.txt"
	var theta = LinearCRF.initialize_params(m)

	// PARSE TRAINING AND TESTING DATA
	val X = Parser.parse_input(datafile, k, n, n_test)
	val X_train = X._1
	val X_test = X._2

	// TRAIN
	val results = LinearCRF.maximizeWithLBFGS(theta, X_train, k, n, sigma_sq)
	println(results._1)
	println(results._2)
	
	// TEST 	
	println("Prediction Accuracy: " + LinearCRF.prediction_accuracy(X_test, results._1, k))

	println("\n\n\nTESTS")
	for(i <- 0 until 20) {
		var v = DenseVector.rand(m)
		println(LinearCRF.likelihood(X_train, v, k, n, sigma_sq))
	}
}
