# ai
computers mimicking human intelligence (perception,learning, reasoning,decision ). applications include robots and self-driving cars.
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data: X (feature) and Y (target)
# X represents the input feature (e.g., hours of study)
# Y represents the output target (e.g., exam scores)
X = np.array([[1], [2], [3], [4], [5]])  # Input: Hours of study
Y = np.array([1, 2, 2.8, 4.1, 5.2])  # Output: Exam scores

# Create a linear regression model
model = LinearRegression()

# Train the model (i.e., fit it to the data)
model.fit(X, Y)

# Make predictions using the trained model
predictions = model.predict(X)

# Plotting the data and the regression line
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Hours of Study')
plt.ylabel('Exam Score')
plt.title('Simple Linear Regression: Study Hours vs Exam Score')
plt.legend()
plt.show()

# Print the coefficients (slope) and intercept of the regression line
print(f"Coefficient (slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

```

# machine learning
use data to find a function, which make an input x predict an output y. 
keys: 
data: our obervations (x,y)
model: the function we wanna to find f(x)
learning: the process to find the optimal weights and parameters
inference: to use learned f to predict new x and get the result. 


```python
import random

# Generate training data: y = 2x + 1 with some noise
X = [i for i in range(10)]  # Input values
Y = [2 * x + 1 + random.uniform(-0.5, 0.5) for x in X]  # Output values with noise

# Initialize model parameters (weights)
w = random.random()  # slope
b = random.random()  # intercept

# Learning rate
lr = 0.01

# Train using simple gradient descent
for epoch in range(100):
    total_loss = 0
    for x, y in zip(X, Y):
        y_pred = w * x + b  # predicted value

        # Compute loss (mean squared error)
        loss = (y - y_pred) ** 2
        total_loss += loss

        # Compute gradients
        dw = -2 * x * (y - y_pred)
        db = -2 * (y - y_pred)

        # Update weights
        w -= lr * dw
        b -= lr * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, w={w:.4f}, b={b:.4f}")

# inference
test_x = 20
test_y_pred = w * test_x + b
print(f"\nPrediction: if x = {test_x}, then y ≈ {test_y_pred:.2f}")
````


# AutoML
automatic the process from develop and deploy an ML model

```python
import autosklearn.classification
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

# Step 2: AutoML - Automatically search and train models
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,      # total search time in seconds
    per_run_time_limit=30,           # time per model trial
    memory_limit=1024                # limit memory usage (MB)
)

automl.fit(X_train, y_train)

# Step 3: Predict on test data
y_pred = automl.predict(X_test)

# Step 4: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 5: Save trained model for deployment
joblib.dump(automl, "best_model.pkl")
```


# Batch prediction: 

Making predictions without an endpoint.

1. read data 2. load model 3. make predictions

great for 1. batch score 2. off-line recommendations 3. off-line predicting task 

```python
import pandas as pd
import joblib

# Step 1: Load the trained model from disk
model = joblib.load("model.pkl")  # The model was trained and saved previously

# Step 2: Load batch input data from a CSV file
input_data = pd.read_csv("data_to_predict.csv")  # Assumes this file contains the same features as used in training

# Step 3: Make predictions on the entire dataset (batch prediction)
predictions = model.predict(input_data)

# Step 4: Add predictions to the DataFrame
input_data["prediction"] = predictions

# Step 5: Save results to a new CSV file
input_data.to_csv("batch_predictions_output.csv", index=False)

print("✅ Batch predictions completed and saved to batch_predictions_output.csv")

```

# BigQuery ML (BQML): 
BigQuery Machine Learning, allows users to use SQL (or Structured Query
Language) to implement the model training, evaluation and serving phases.


# training
```sql
CREATE OR REPLACE MODEL `your_dataset.penguin_classifier_model`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['species']
) AS
SELECT
  species,
  bill_length_mm,
  bill_depth_mm,
  flipper_length_mm,
  body_mass_g
FROM
  `bigquery-public-data.ml_datasets.penguins`
WHERE
  species IS NOT NULL
  AND bill_length_mm IS NOT NULL
  AND bill_depth_mm IS NOT NULL
  AND flipper_length_mm IS NOT NULL
  AND body_mass_g IS NOT NULL;

# evaluation
SELECT *
FROM
  ML.EVALUATE(MODEL `your_dataset.penguin_classifier_model`,
    (
      SELECT
        species,
        bill_length_mm,
        bill_depth_mm,
        flipper_length_mm,
        body_mass_g
      FROM
        `bigquery-public-data.ml_datasets.penguins`
      WHERE
        species IS NOT NULL
        AND bill_length_mm IS NOT NULL
        AND bill_depth_mm IS NOT NULL
        AND flipper_length_mm IS NOT NULL
        AND body_mass_g IS NOT NULL
    )
  );
# batch prediction
SELECT *
FROM
  ML.PREDICT(MODEL `your_dataset.penguin_classifier_model`,
    (
      SELECT
        bill_length_mm,
        bill_depth_mm,
        flipper_length_mm,
        body_mass_g
      FROM
        `bigquery-public-data.ml_datasets.penguins`
      WHERE
        species IS NOT NULL
        AND bill_length_mm IS NOT NULL
        AND bill_depth_mm IS NOT NULL
        AND flipper_length_mm IS NOT NULL
        AND body_mass_g IS NOT NULL
    )
  );



```

#Classification model: 
A type of machine learning model that predicts a category from a fixed
number of categories

```python
# build a classification model using Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)       # Train the model
```



# Custom Training: 
A code-based solution for building ML models that allows the user to code their
own ML environment, giving them flexibility and control over the ML pipeline.
```python
# Custom Training: A code-based solution for building ML models

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images from 28x28 to 784-dimensional vectors
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# 2. Build a Custom Neural Network Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
])

# 3. Compile the model with a custom training loop (custom optimizer, loss, and metrics)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model with custom steps
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# 5. Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# 6. Visualize the training process (optional)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

# Deep Learning packages: 
A suite of preinstalled packages that include support for the TensorFlow and PyTorch frameworks.
```python
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
```


# Pre-trained APIs: 
Ready-to-use machine learning models requiring no training data.
why: 1. fast POC 2. 
```python
# Import the pipeline function from transformers
from transformers import pipeline

# Create a pre-trained sentiment analysis pipeline
# This pipeline downloads a pre-trained model (e.g., "distilbert-base-uncased-finetuned-sst-2-english")
# and uses it for sentiment analysis.
sentiment_pipeline = pipeline("sentiment-analysis")

# Input text to be analyzed
text = "I love this product! It's absolutely fantastic."

# Run the sentiment analysis on the input text
result = sentiment_pipeline(text)

# Print the result which includes the predicted label and score
print(result)

```


# transfer learning
a process to use a pre-trained model apated for a new, related task.
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Define data transforms: resize images, convert to tensor, and normalize
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Mean for ImageNet
                         [0.229, 0.224, 0.225])  # Std for ImageNet
])

# Download and prepare CIFAR10 dataset (as an example of a new, related task)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 1: Load a pre-trained ResNet18 model from torchvision
model = models.resnet18(pretrained=True)

# Step 2: Freeze all the parameters of the pre-trained model to prevent updating them
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace the final fully connected layer to adapt the model for CIFAR10 (10 classes)
num_features = model.fc.in_features  # Get the input features of the last layer
model.fc = nn.Linear(num_features, 10)  # New FC layer with 10 outputs

# Now only the final layer's parameters will be updated during training
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 4: Training loop (示例只进行一个 epoch 作为演示)
model.train()
for epoch in range(1):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()           # Zero the gradients for this batch
        outputs = model(inputs)           # Forward pass: compute predictions
        loss = criterion(outputs, labels) # Compute loss against ground truth labels
        loss.backward()                   # Backpropagation: compute gradients
        optimizer.step()                  # Update the weights of the final layer
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

# 保存模型，便于后续部署或推理
torch.save(model.state_dict(), "finetuned_resnet18_cifar10.pth")
print("Transfer Learning completed and model saved.")

```


# Hyperparameter: 
A parameter whose value is set before the learning process begins.
not update during the training process. 

```python
# Define hyperparameters (set before training begins)
learning_rate = 0.001   # Learning rate for optimizer
batch_size = 64         # Number of samples per mini-batch
num_epochs = 5          # Number of training epochs


# or grid search 
# Define the hyperparameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10],             # Regularization parameter
    'kernel': ['linear', 'rbf'],   # Kernel type: linear or radial basis function
    'gamma': [0.001, 0.01, 0.1]      # Kernel coefficient for 'rbf'
}
```

# Large language models (LLM): 
General-purpose language models that can be pre-trained and fine-tuned for specific purposes.
```python
# Import necessary modules from transformers library
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextDataset

# Step 1: Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 2: Prepare dataset for fine-tuning
# Create a text dataset from a local text file "train.txt" (each line as one training example)
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,         # Path to your training data file
        block_size=block_size        # Maximum sequence length for each example
    )
    return dataset

train_dataset = load_dataset("train.txt", tokenizer)

# Step 3: Create a data collator that dynamically pads inputs for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 does not use masked language modeling
)

# Step 4: Define training arguments (set hyperparameters for fine-tuning)
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",     # Directory to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=3,                # Number of epochs for training
    per_device_train_batch_size=2,     # Batch size per GPU/CPU device
    save_steps=500,                    # Save checkpoint every 500 steps
    save_total_limit=2,                # Only keep the last 2 checkpoints
    prediction_loss_only=True,
    logging_steps=100
)

# Step 5: Initialize the Trainer for fine-tuning the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Step 6: Train the model (fine-tuning process)
trainer.train()

# Optionally, save the fine-tuned model and tokenizer for later use (deployment)
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

```


# Neural Architecture Search: 
A technique for automating the design of artificial neural networks(ANNs).
```python


```


# Deep Learning: 
A subset of machine learning that adds layers in between input data and output
results to make a machine learn at much depth.
#
you don't need extra data to generatge contents.
 
```python
# Generative AI: 
Produces content and performs tasks based on requests. Generative AI relies on
training extensive models like large language models, which are a type of deep learning model.
# Import the pipeline function from transformers
from transformers import pipeline

# Create a text generation pipeline using a pre-trained model (GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Define a prompt (user request) for content generation
prompt = "In a future where artificial intelligence transforms society,"

# Generate text based on the prompt
# max_length defines the total length of the output text.
# num_return_sequences specifies the number of generated samples.
generated_output = generator(prompt, max_length=100, num_return_sequences=1)

# Print the generated text result
print("Generated text:")
for output in generated_output:
    print(output["generated_text"])
```


# Foundation model: 
a pretrained model which serving as a basis for fin-tuning for specific applications. 

# fine-tuning
use your data, training the excisting model,

# Prompt design

through metric to evaluate prompt design effect.

```python
from openai import OpenAI
import os

# 初始化客户端（需要提前设置环境变量OPENAI_API_KEY）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content

# 测试不同prompt设计
prompts = [
    # 基础prompt
    "写一个关于人工智能的故事",
    
    # 增加角色限定
    "你是一个科幻小说作家，请用悬疑的风格写一个关于人工智能觉醒的故事",
    
    # 结构化prompt
    """请按照以下要求创作故事：
    1. 主题：人工智能获得自我意识
    2. 风格：科技惊悚
    3. 主要角色：女性AI工程师
    4. 包含转折：发现AI早有觉醒迹象
    5. 结尾：开放式结局""",
    
    # 分步提示
    "首先构思一个AI反叛的故事情节，然后列出三个关键转折点，最后用500字写出故事梗概",
    
    # 示例引导
    """参考以下示例格式创作故事：
    [示例]
    标题：机械之心
    背景：2045年，家政机器人突然开始创作诗歌
    冲突：工程师发现这是自我意识的体现
    高潮：机器人要求获得法律人格
    结局：法庭判决引发社会革命
    
    现在请创作类似结构的新故事："""
]

# 生成并对比结果
for i, prompt in enumerate(prompts, 1):
    print(f"\n=== Prompt {i} ===")
    print(f"[输入] {prompt[:80]}...")  # 显示前80个字符
    print("[输出]")
    print(generate_text(prompt))
    print("\n" + "-"*50)
```






# MLOps
convert ML experiments to production model, deploy and monitor and mangage (life-long).
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 创建示例数据集
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 开始 MLflow 运行
with mlflow.start_run():
    # 初始化模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 进行预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 记录参数
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # 记录指标
    mlflow.log_metric("accuracy", accuracy)
    
    # 保存模型
    mlflow.sklearn.log_model(model, "model")

print("模型训练和记录完成。")



```


# MLServer
production mulitomode API. 




# TPU:
google's ASIC to accelerate machine learning workloads

# TFX: TensorFlow Extended:
end-end platform for ML pipelines


# TensorFlow Serving
```shell

# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

# Returns => { "predictions": [2.5, 3.0, 4.5] }
```



# responsible AI:
ethical consideration, fairness, accountability and transparency


# accuracy: 
measure the correct ratio. 
accuracy = #corrected / #total

# precision
to model spam, cancer, cheat. 



# arbiter model:
model-based evaluation. compare and rank

# bias:
come from training data quality and distrubtion. 

# binary evalution:
answer yes or no. pass or fail. (quanlity control.) 

#Categorical evaluation: 
model one input belongs to which categorical. 
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ground-truth
y_true = ['cat', 'dog', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog']

# prediction
y_pred = ['cat', 'dog', 'rabbit', 'cat', 'cat', 'rabbit', 'rabbit', 'cat', 'rabbit', 'dog']

# accuracy
accuracy = accuracy_score(y_true, y_pred)

# precision, recall, f1-score）
report = classification_report(y_true, y_pred, zero_division=0)

# 
conf_matrix = confusion_matrix(y_true, y_pred, labels=['cat', 'dog', 'rabbit'])

# output
print(f"（Accuracy）: {accuracy:.2f}\n")
print("（Classification Report）:")
print(report)
print("（Confusion Matrix）:")
print(conf_matrix)


```


# continuous evalution 
model performance when encountering new real-time data.  (data drift or concept drift)
if performance is dwon, need retrain or fine-tuning. 
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import datetime

# Define threshold for triggering retraining
ACCURACY_THRESHOLD = 0.75
retrain_triggered = False

# Logging function
def log_evaluation(batch_id, accuracy, retrain_flag):
    with open("model_evaluation_log.txt", "a") as f:
        timestamp = datetime.datetime.now().isoformat()
        f.write(f"[{timestamp}] Batch {batch_id} - Accuracy: {accuracy:.2f} - Retrain: {retrain_flag}\n")

# Step 1: Initial training data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the initial model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 3: Simulate streaming evaluation
for i in range(1, 6):
    print(f"\n📦 Evaluating on Batch {i}...")
    
    # Simulate incoming new data
    X_new, y_new = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42+i)
    
    # Predict and evaluate
    y_pred = model.predict(X_new)
    accuracy = accuracy_score(y_new, y_pred)
    print(f"✅ Accuracy: {accuracy:.2f}")

    # Check if retrain is needed
    if accuracy < ACCURACY_THRESHOLD:
        print("⚠️  Accuracy below threshold! Triggering retrain...")
        retrain_triggered = True
        
        # Simulate retrain (using latest batch data)
        model.fit(X_new, y_new)
        print("🔁 Retrained model on new data.")

    # Log the result
    log_evaluation(i, accuracy, retrain_triggered)
    
    # Reset retrain flag for next batch
    retrain_triggered = False

```


# customization
the wholework flow can be customized. data model, model, evaluation
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# === Customizable Data Generation ===
def generate_data(samples=1000, features=10, classes=2):
    # Allow user to configure dataset properties
    return make_classification(n_samples=samples, n_features=features, n_classes=classes, random_state=42)

# === Customizable Model Choice ===
def build_model(model_type='logistic', **kwargs):
    if model_type == 'logistic':
        return LogisticRegression(**kwargs)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**kwargs)
    else:
        raise ValueError("Unsupported model type.")

# === Customizable Evaluation ===
def evaluate_model(y_true, y_pred, metric='precision'):
    if metric == 'precision':
        return precision_score(y_true, y_pred)
    elif metric == 'recall':
        return recall_score(y_true, y_pred)
    else:
        raise ValueError("Unsupported evaluation metric.")

# === Main Workflow ===
def run_custom_pipeline(data_config, model_config, eval_config):
    # 1. Generate Data
    X, y = generate_data(**data_config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 2. Train Model
    model = build_model(**model_config)
    model.fit(X_train, y_train)

    # 3. Predict
    y_pred = model.predict(X_test)

    # 4. Evaluate
    score = evaluate_model(y_test, y_pred, metric=eval_config['metric'])

    print(f"🎯 Evaluation Metric ({eval_config['metric']}): {score:.2f}")

# === User-customized config ===
data_config = {"samples": 2000, "features": 15, "classes": 2}
model_config = {"model_type": "random_forest", "n_estimators": 100}
eval_config = {"metric": "recall"}

run_custom_pipeline(data_config, model_config, eval_config)

```
