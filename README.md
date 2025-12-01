# Email Spam Detection

A Scala-based email spam detection system using Apache Spark MLlib that achieves **95.85% accuracy**.

## Features
- **TF-IDF vectorization** for text feature extraction
- **Logistic Regression** classification
- **80/20 train/test split** for model validation
- **CSV data support** with automatic label detection
- **Null handling** for robust data processing
- **Java 17 compatibility** with proper JVM configurations

## Prerequisites
- Java 17 (OpenJDK recommended)
- Scala 2.12.x
- SBT (Scala Build Tool)

## Installation
1. **Install Java 17**: Download from [OpenJDK](https://openjdk.org/)
2. **Install Scala**: Visit [scala-lang.org](https://www.scala-lang.org/download/)
3. **Install SBT**: Visit [scala-sbt.org](https://www.scala-sbt.org/download.html)

## Data Format
Place your email dataset as `emails.csv` in the project root with columns:
- `file`: Filename (used to detect spam/ham labels)
- `message`: Email content text

## Usage

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Hemanth94780/Email_Spam_Detection.git
cd Email_Spam_Detection

# Place your emails.csv file in the root directory
# Run the application
sbt run
```

### Build JAR
```bash
sbt assembly
```

## Results
- **Accuracy**: 95.85%
- **Dataset**: 1.6M+ emails processed
- **Performance**: Handles large datasets efficiently with Spark

## Technical Details
- **Framework**: Apache Spark 3.3.0
- **Language**: Scala 2.12.15
- **ML Algorithm**: Logistic Regression with TF-IDF features
- **Feature Count**: 1000 hash features
- **Evaluation Metric**: Binary Classification (AUC)

## Sample Output
```
Accuracy: 0.9585109385719012
+---------------------------------------------+-----+----------+-------------------------------------------+
|text                                         |label|prediction|probability                                |
+---------------------------------------------+-----+----------+-------------------------------------------+
|Sample email content...                      |0.0  |0.0       |[0.9999999999999989,1.1102230246251565E-15]|
+---------------------------------------------+-----+----------+-------------------------------------------+
```

## Dataset Sources
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/spambase)