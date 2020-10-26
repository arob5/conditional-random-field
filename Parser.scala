import breeze.linalg._
import scala.collection.mutable.ArrayBuffer


//
// Parser provides functionality to parse a file containing labeled sentences into a usable format 
// for LinearCRF
//

object Parser {

	type Label = String
	type Term = String

	/*
	* Given a file with lines of the form "WORD LABEL1 LABEL2", parses text and returns the training and testing datasets
	*/
	def parse_input(filename: String, k: Int, n_examples: Int, n_test: Int) : (List[Seq[Array[String]]], List[Seq[Array[String]]]) = {
		val data = scala.io.Source.fromFile(filename).getLines.filter(!_.isEmpty()).map(_.split("\\s+"))
		val itr = data.sliding(k)
		val X = itr.take(n_examples + n_test).toList
		val X_train = X.slice(0, n_examples)
		val X_test = X.slice(n_examples, n_examples + n_test)

		return (X_train, X_test)
	}

	/* Converts line from file into a training example tuple */
	def parse_observation(obs: Seq[Array[String]], k: Int) : (Array[Label], Array[Term]) = {

		var labels = new Array[Label](k)
		var terms = new Array[Term](k)

		for(i <- 0 until k) {
			labels(i) = change_label(obs(i)(1)) 
			terms(i) = obs(i)(0)
		}
		
		return (labels, terms)

	}
	
	/* Converts original labels to either NOUN, VERB, or OTHER */ 
	def change_label(old_label: String) : String = {
		if(old_label(0) == 'N')
			return "NOUN"
		else if(old_label(0) == 'V')
			return "VERB"
		else
			return "OTHER"
	}

}
