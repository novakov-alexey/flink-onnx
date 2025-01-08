package example

import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.Tensors.Tensor.*
import org.emergentorder.onnx.backends.ORTModelBackend
import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.*
import cats.effect.unsafe.implicits.*

import org.apache.flinkx.api.StreamExecutionEnvironment
import org.apache.flinkx.api.serializers.*
import org.apache.flinkx.api.conv.*
import org.apache.flinkx.api.*

import org.apache.flink.types.Row
import org.apache.flink.table.api.Expressions.*
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment
import org.apache.flink.table.api.*

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.flink.configuration.RestOptions.BIND_PORT
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoder
import org.apache.flink.ml.feature.standardscaler.StandardScaler
import org.apache.flink.ml.feature.stringindexer.{
  StringIndexer,
  StringIndexerParams
}
import org.apache.flink.ml.feature.vectorassembler.VectorAssembler
import org.apache.flink.ml.builder.Pipeline
import org.apache.flink.ml.api.Stage
import org.apache.flink.ml.Functions.vectorToArray

import scala.jdk.CollectionConverters.*
import scala.reflect.ClassTag

case class ChurnPrediction(raw: Float, exited: Boolean)

case class Customer(
    creditScore: Double,
    geography: String,
    gender: String,
    age: Int,
    tenure: Double,
    balance: Double,
    numOfProducts: Int,
    hasCrCard: Boolean,
    isActiveMember: Boolean,
    estimatedSalary: Double
)

class CustomerChurnClassifier(modelPath: String, vectorSize: Int)
    extends RichMapFunction[Array[Float], ChurnPrediction]:
  private val shape = Shape.matrix(1, vectorSize)
  @transient var ann: ORTModelBackend = _

  override def open(parameters: Configuration): Unit =
    val modelBytes = os.read.bytes(os.Path(modelPath))
    ann = ORTModelBackend(modelBytes)

  override def map(features: Array[Float]): ChurnPrediction =
    val inputs = Tensor(features, shape)
    val out = ann.fullModel[
      Float,
      "CustomerChurnClassification",
      "Batch" ##: "Exited" ##: TSNil,
      1 #: 1 #: SNil
    ](Tuple(inputs))

    val predicted = out.data.unsafeRunSync()
    ChurnPrediction(predicted.head, predicted.head > 0.5)

@main def main(args: String*): Unit =
  val localRun = args.isEmpty
  lazy val config = Configuration.fromMap(
    Map(
      BIND_PORT.key -> "8081",
      "execution.checkpointing.interval" -> "5 s"
    ).asJava
  )
  val env =
    if localRun then
      StreamExecutionEnvironment.createLocalEnvironmentWithWebUI(config)
    else StreamExecutionEnvironment.getExecutionEnvironment
  val tEnv = StreamTableEnvironment.create(env)

  val modelPath =
    if localRun then os.pwd / "customer-churn.onnx"
    else
      val path = os.Path(args.head)
      if !path.toIO.exists() then
        sys.error(s"File at path `$path` does not exist")
      path

  val featuresCol = "features"

  val schema = Schema
    .newBuilder()
    .column("RowNumber", DataTypes.INT())
    .column("CustomerId", DataTypes.INT())
    .column("Surname", DataTypes.STRING())
    .column("CreditScore", DataTypes.DOUBLE())
    .column("GeographyStr", DataTypes.STRING())
    .column("GenderStr", DataTypes.STRING())
    .column("Age", DataTypes.DOUBLE())
    .column("Tenure", DataTypes.DOUBLE())
    .column("Balance", DataTypes.DOUBLE())
    .column("NumOfProducts", DataTypes.DOUBLE())
    .column("HasCrCard", DataTypes.DOUBLE())
    .column("IsActiveMember", DataTypes.DOUBLE())
    .column("EstimatedSalary", DataTypes.DOUBLE())
    .column("Exited", DataTypes.DOUBLE())
    .build()

  // 1 - Index Categorical columns
  val indexer = StringIndexer()
    .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
    .setInputCols("GeographyStr", "GenderStr")
    .setOutputCols("GeographyInd", "Gender")

  // 2 - Encode Geography column
  val geographyEncoder =
    OneHotEncoder()
      .setInputCols("GeographyInd")
      .setOutputCols("Geography")
      .setDropLast(false)

  // 3 - Merge to Vector
  val continuesCols = List(
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
  )
  val categoricalCols =
    List("Geography", "Gender", "HasCrCard", "IsActiveMember")
  // Geography is 3 countries + other features
  val encodedFeatures = List(3)
  val vectorSizes =
    encodedFeatures ++ List.fill(
      categoricalCols.length - encodedFeatures.length + continuesCols.length
    )(1)

  val assembler = VectorAssembler()
    .setInputCols((categoricalCols ++ continuesCols)*)
    .setOutputCol("combined_features")
    .setInputSizes(vectorSizes.map(Integer.valueOf)*)

  // 4 - Normalize numbers
  val standardScaler =
    StandardScaler()
      .setWithMean(true)
      .setInputCol("combined_features")
      .setOutputCol(featuresCol)

  val trainDataPath =
    if localRun then s"file://${os.pwd}/data/train/Churn_Modelling.csv"
    else
      if args.length < 1 then
        sys.error(
          s"Remote execution requires file path to CSV training data at position (1)"
        )
      val filePath = args(1)
      if !os.Path(filePath).toIO.exists then
        sys.error(s"File at ${filePath} path does not exist")
      filePath

  val trainData = tEnv.from(
    TableDescriptor
      .forConnector("filesystem")
      .schema(schema)
      .option("path", trainDataPath)
      .option("format", "csv")
      .option("csv.allow-comments", "true")
      .build()
  )

  val stages = List[Stage[?]](
    indexer,
    geographyEncoder,
    assembler,
    standardScaler
  ).asJava

  val pipeline = Pipeline(stages)
  val featureExtractor = pipeline.fit(trainData)

  val testDataPath =
    if localRun then s"file://${os.pwd}/data/test"
    else args(2)
  val testData = tEnv.from(
    TableDescriptor
      .forConnector("filesystem")
      .schema(
        Schema
          .newBuilder()
          .fromColumns(
            // remove label column
            schema.getColumns().asScala.dropRight(1).asJava
          )
          .build()
      )
      .option("path", testDataPath)
      .option("format", "csv")
      .option("csv.allow-comments", "true")
      .option("source.monitor-interval", "3s")
      .build()
  )

  val transformed = featureExtractor.transform(testData).head
  val features = transformed
    .select(
      vectorToArray($(featuresCol))
        .cast(DataTypes.ARRAY(DataTypes.FLOAT()))
    )

  def toArray[T: ClassTag](r: Row): Array[T] =
    try r.getFieldAs[Array[?]](0).map(_.asInstanceOf[T])
    catch
      case e =>
        e.printStackTrace
        sys.error(s"Failed to parse field at row(0): $r")

  DataStream(tEnv.toDataStream(features))
    .map(toArray[Float])
    .map(CustomerChurnClassifier(modelPath.toString, vectorSizes.sum))
    .print()

  env.execute("CustomerChurnAnalysis")
