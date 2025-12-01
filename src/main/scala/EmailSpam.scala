import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.functions._
import java.io.File

object EmailSpam {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("EmailSpamDetection")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Load data from CSV
    val allEmails = loadEmailsFromCSV("emails.csv", spark)
    
    // Split data
    val Array(training, test) = allEmails.randomSplit(Array(0.8, 0.2), seed = 42)
    
    // Create ML pipeline
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(1000)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, lr))
    
    // Train model
    val model = pipeline.fit(training)
    
    // Make predictions
    val predictions = model.transform(test)
    
    // Evaluate
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val accuracy = evaluator.evaluate(predictions)
    
    println(s"Accuracy: ${accuracy}")
    
    // Show sample predictions
    predictions.select("text", "label", "prediction", "probability").show(10, false)
    
    spark.stop()
  }
  
  def loadEmailsFromCSV(path: String, spark: SparkSession): org.apache.spark.sql.DataFrame = {
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
    
    // Rename message to text and convert file column to label, filter out nulls
    df.select(
      col("message").as("text"),
      when(col("file").contains("spam"), 1.0).otherwise(0.0).as("label")
    ).filter(col("text").isNotNull && col("text") =!= "")
  }
}