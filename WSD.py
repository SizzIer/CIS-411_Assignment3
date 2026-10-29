
import sys
import math
import string
import html


# Check the file argument
if len(sys.argv) != 2:
    print("Usage: py WSD.py plant.wsd")
    sys.exit(1)

filename = sys.argv[1]


# Read the instances from the file
def readInstances(filename):
    instances = []

    with open(filename, "r", encoding="utf-8") as file:
        text = html.unescape(file.read())

    position = 0

    while True:
        start = text.find("<instance", position)

        if start == -1:
            break

        end = text.find("</instance>", start)

        if end == -1:
            break

        instance = text[start:end]
        position = end + len("</instance>")

        # Get the instance ID
        openingTag = instance[:instance.find(">")]

        idStart = openingTag.find('id="')
        if idStart == -1:
            continue

        idStart += len('id="')
        idEnd = openingTag.find('"', idStart)
        instanceID = openingTag[idStart:idEnd]

        # Get the correct sense
        senseStart = instance.find('senseid="')
        if senseStart == -1:
            continue

        senseStart += len('senseid="')
        senseEnd = instance.find('"', senseStart)
        sense = instance[senseStart:senseEnd]

        # Get the context
        contextStart = instance.find("<context>")
        contextEnd = instance.find("</context>")

        if contextStart == -1 or contextEnd == -1:
            continue

        contextStart += len("<context>")
        context = instance[contextStart:contextEnd]

        # Remove the target word tags
        context = context.replace("<head>", "")
        context = context.replace("</head>", "")

        instances.append((instanceID, sense, context))

    return instances


# Turn context into words
def tokenize(context):
    context = context.lower()

    for mark in string.punctuation:
        context = context.replace(mark, " ")

    return context.split()


# Divide instances into five folds
def createFolds(instances):
    foldSize = (len(instances) + 4) // 5
    folds = []

    for i in range(4):
        start = i * foldSize
        folds.append(instances[start:start + foldSize])

    folds.append(instances[4 * foldSize:])

    return folds


# Count senses and words in the training data
def trainNaiveBayes(trainingData):
    senseCounts = {}
    wordCounts = {}
    totalWords = {}
    vocabulary = set()

    for instanceID, sense, context in trainingData:
        senseCounts[sense] = senseCounts.get(sense, 0) + 1

        words = tokenize(context)
        totalWords[sense] = totalWords.get(sense, 0) + len(words)

        for word in words:
            vocabulary.add(word)
            key = (sense, word)
            wordCounts[key] = wordCounts.get(key, 0) + 1

    return senseCounts, wordCounts, totalWords, vocabulary


# Predict the most likely sense
def predictSense(context, model):
    senseCounts, wordCounts, totalWords, vocabulary = model

    words = tokenize(context)
    totalInstances = sum(senseCounts.values())
    vocabularySize = len(vocabulary)

    bestSense = None
    bestScore = float("-inf")

    for sense in senseCounts:
        # Start with the sense probability
        prior = senseCounts[sense] / totalInstances
        score = math.log(prior)

        for word in words:
            if word not in vocabulary:
                continue

            # Add-one smoothing
            count = wordCounts.get((sense, word), 0)
            probability = (
                (count + 1) /
                (totalWords[sense] + vocabularySize)
            )

            # Use logs to avoid underflow
            score += math.log(probability)

        if score > bestScore:
            bestScore = score
            bestSense = sense

    return bestSense


# Test the model on one fold
def testNaiveBayes(testData, model):
    correct = 0
    errors = []
    predictions = []

    for instanceID, actualSense, context in testData:
        predictedSense = predictSense(context, model)

        predictions.append((instanceID, predictedSense))

        if predictedSense == actualSense:
            correct += 1
        else:
            errors.append((
                instanceID,
                actualSense,
                predictedSense,
                context
            ))

    accuracy = correct / len(testData) if testData else 0

    return accuracy, correct, errors, predictions


# Train and test using five folds
def fivefoldCrossValidation(folds, filename):
    foldAccuracies = []
    allErrors = []
    totalCorrect = 0
    totalTested = 0

    with open(filename + ".out", "w", encoding="utf-8") as output:
        for i in range(5):
            trainingData = []

            # Use four folds for training
            for j in range(5):
                if j != i:
                    trainingData.extend(folds[j])

            # Use the remaining fold for testing
            testData = folds[i]

            model = trainNaiveBayes(trainingData)
            accuracy, correct, errors, predictions = testNaiveBayes(
                testData, model
            )

            foldAccuracies.append(accuracy)
            totalCorrect += correct
            totalTested += len(testData)

            print(f"Fold {i + 1}: {accuracy * 100:.2f}%")

            # Save predictions for this fold
            output.write(f"Fold {i + 1}\n")

            for instanceID, predictedSense in predictions:
                output.write(f"{instanceID} {predictedSense}\n")

            output.write("\n")

            # Save errors for review
            for instanceID, actual, predicted, context in errors:
                allErrors.append((
                    i + 1, instanceID, actual, predicted, context
                ))

    average = sum(foldAccuracies) / 5
    overall = totalCorrect / totalTested if totalTested else 0

    print(f"Average Accuracy: {average * 100:.2f}%")
    print(f"Overall Accuracy: {overall * 100:.2f}%")
    print(f"Created {filename}.out")

    return foldAccuracies, allErrors, average


# Main program
instances = readInstances(filename)

if not instances:
    print("Error: No instances found.")
    sys.exit(1)

# Show dataset information
senseLabels = set()

for instanceID, sense, context in instances:
    senseLabels.add(sense)

print("Total instances:", len(instances))
print("Sense labels:", senseLabels)

folds = createFolds(instances)

for i in range(5):
    print(f"Fold {i + 1} size: {len(folds[i])}")

print()

# Run cross-validation
foldAccuracies, errors, average = fivefoldCrossValidation(
    folds, filename
)

# Show three errors for the written answers
print("\nThree Classification Errors")

for i, error in enumerate(errors[:3], start=1):
    fold, instanceID, actual, predicted, context = error

    print(f"\nError {i}")
    print("Fold:", fold)
    print("Instance:", instanceID)
    print("Correct:", actual)
    print("Predicted:", predicted)
    print("Context:", " ".join(context.split()))
