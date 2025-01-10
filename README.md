# flink-onnx

Example Flink job to inference ONNX-based ANN model for Customer Churn Analysis.

Model is trained and generated using Python: https://github.com/novakov-alexey/tf-onnx-customer-churn. 
See `README.md` file for the instructions.

Pre-requisites:
- JDK 11 or higher
- Latest SBT version

To run Flink job locally:

```bash
sbt 
> run
```

then add a file into the `data/test` folder with the same table schema as existing "Small_test.csv" file.

To build a fat JAR:

```bash
sbt
> assembly
```