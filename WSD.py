import sys
import math
import string
import html


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python wsd.py plant.wsd")
    sys.exit(1)

filename = sys.argv[1]


# --------------------------------------------------
# Read all annotated instances
# --------------------------------------------------

def readInstances(filename):
    instances = []

    with open(filename, "r", encoding="utf-8") as file:
        fileText = file.read()

    # Convert encoded tags such as &lt;instance&gt;
    # into normal tags such as <instance>
    fileText = html.unescape(fileText)

    searchPosition = 0

    while True:
        instanceStart = fileText.find(
            "<instance",
            searchPosition
        )

        if instanceStart == -1:
            break

        instanceEnd = fileText.find(
            "</instance>",
            instanceStart
        )

        if instanceEnd == -1:
            break

        instanceText = fileText[
            instanceStart:instanceEnd
        ]

        # Find the correct sense label
        senseStart = instanceText.find('senseid="')

        if senseStart == -1:
            searchPosition = (
                instanceEnd + len("</instance>")
            )
            continue

        senseStart += len('senseid="')
        senseEnd = instanceText.find('"', senseStart)

        sense = instanceText[senseStart:senseEnd]

        # Find the context
        contextStart = instanceText.find("<context>")
        contextEnd = instanceText.find("</context>")

        if contextStart == -1 or contextEnd == -1:
            searchPosition = (
                instanceEnd + len("</instance>")
            )
            continue

        contextStart += len("<context>")

        context = instanceText[
            contextStart:contextEnd
        ]

        # Remove the head tags but keep the target word
        context = context.replace("<head>", "")
        context = context.replace("</head>", "")

        instances.append((sense, context))

        searchPosition = (
            instanceEnd + len("</instance>")
        )

    return instances


# --------------------------------------------------
# Convert the context into word features
# --------------------------------------------------

def tokenize(context):
    # Treat uppercase and lowercase forms as the same word
    context = context.lower()

    # Replace punctuation marks with spaces
    for mark in string.punctuation:
        context = context.replace(mark, " ")

    # All words in the context are features
    words = context.split()

    return words


# --------------------------------------------------
# Create five folds
# --------------------------------------------------

def createFolds(instances):
    foldSize = math.ceil(len(instances) / 5)

    folds = []

    # Folds 1 through 4 get the rounded-up fold size
    for i in range(4):
        start = i * foldSize
        end = start + foldSize

        folds.append(instances[start:end])

    # Fold 5 gets all remaining instances
    folds.append(instances[4 * foldSize:])

    return folds


# --------------------------------------------------
# Train the Naive Bayes classifier
# --------------------------------------------------

def trainNaiveBayes(trainingData):
    senseCounts = {}
    wordCounts = {}
    totalWordsBySense = {}
    vocabulary = set()

    for sense, context in trainingData:

        if sense not in senseCounts:
            senseCounts[sense] = 0
            totalWordsBySense[sense] = 0

        # Count the number of instances for this sense
        senseCounts[sense] += 1

        # Extract all context-word features
        words = tokenize(context)

        # Count all word occurrences for this sense
        totalWordsBySense[sense] += len(words)

        for word in words:
            vocabulary.add(word)

            if (sense, word) not in wordCounts:
                wordCounts[(sense, word)] = 0

            wordCounts[(sense, word)] += 1

    return (
        senseCounts,
        wordCounts,
        totalWordsBySense,
        vocabulary
    )


# --------------------------------------------------
# Predict the sense of one test instance
# --------------------------------------------------

def predictSense(
    context,
    senseCounts,
    wordCounts,
    totalWordsBySense,
    vocabulary
):
    words = tokenize(context)

    totalTrainingInstances = sum(
        senseCounts.values()
    )

    vocabularySize = len(vocabulary)

    bestSense = None
    bestLogProbability = float("-inf")

    for sense in senseCounts:

        # Prior probability:
        # P(sense) = count(sense) / training instances
        prior = (
            senseCounts[sense]
            / totalTrainingInstances
        )

        # Begin in log space
        logProbability = math.log(prior)

        for word in words:

            # Ignore words that never occurred anywhere
            # in this training run
            if word not in vocabulary:
                continue

            wordCount = wordCounts.get(
                (sense, word),
                0
            )

            totalWords = totalWordsBySense[sense]

            # Add-one smoothing:
            #
            # P(word | sense) =
            # (count(word, sense) + 1)
            # /
            # (total words for sense + vocabulary size)
            probabilityWordGivenSense = (
                (wordCount + 1)
                /
                (totalWords + vocabularySize)
            )

            # Add log probabilities to avoid underflow
            logProbability += math.log(
                probabilityWordGivenSense
            )

        # Keep the sense with the highest probability
        if logProbability > bestLogProbability:
            bestLogProbability = logProbability
            bestSense = sense

    return bestSense


# --------------------------------------------------
# Test one trained model
# --------------------------------------------------

def testNaiveBayes(testData, model):
    (
        senseCounts,
        wordCounts,
        totalWordsBySense,
        vocabulary
    ) = model

    correct = 0
    errors = []

    for actualSense, context in testData:

        predictedSense = predictSense(
            context,
            senseCounts,
            wordCounts,
            totalWordsBySense,
            vocabulary
        )

        if predictedSense == actualSense:
            correct += 1
        else:
            errors.append(
                (
                    actualSense,
                    predictedSense,
                    context
                )
            )

    totalTested = len(testData)

    if totalTested == 0:
        accuracy = 0.0
    else:
        accuracy = correct / totalTested

    return correct, totalTested, accuracy, errors


# --------------------------------------------------
# Perform five-fold cross-validation
# --------------------------------------------------

def fivefoldCrossValidation(folds):
    totalCorrect = 0
    totalTested = 0

    for i in range(5):
        trainingData = []
        testData = []

        for j in range(5):
            if j == i:
                testData.extend(folds[j])
            else:
                trainingData.extend(folds[j])

        model = trainNaiveBayes(trainingData)

        correct, tested, accuracy = testNaiveBayes(
            testData,
            model
        )

        totalCorrect += correct
        totalTested += tested

        print(f"Fold {i + 1} Accuracy: {accuracy * 100:.2f}%")

    if totalTested > 0:
        overallAccuracy = totalCorrect / totalTested
    else:
        overallAccuracy = 0.0

    print(f"Overall Accuracy: {overallAccuracy * 100:.2f}%")

# --------------------------------------------------
# Create WSD.answers
# --------------------------------------------------

def createAnswersFile(
    foldAccuracies,
    errors,
    overallAccuracy
):
    with open(
        "WSD.answers",
        "w",
        encoding="utf-8"
    ) as answerFile:

        # State how complete the program is
        answerFile.write(
            "The program is complete.\n"
        )

        # The dataset name must be on a line by itself
        answerFile.write("plant.wsd\n")

        # Write accuracy for each of the five folds
        for i in range(5):
            answerFile.write(
                f"Fold {i + 1} Accuracy: "
                f"{foldAccuracies.4f} "
                f"({foldAccuracies[i] * 100:.2f}%)\n"
            )

        answerFile.write(
            f"Overall Accuracy: "
            f"{overallAccuracy:.4f} "
            f"({overallAccuracy * 100:.2f}%)\n"
        )

        answerFile.write("\n")
        answerFile.write(
            "Three Classification Errors:\n"
        )

        numberOfErrors = min(3, len(errors))

        for i in range(numberOfErrors):
            (
                foldNumber,
                actualSense,
                predictedSense,
                context
            ) = errors[i]

            # Put the context on one readable line
            cleanContext = " ".join(
                context.split()
            )

            answerFile.write(
                f"Error {i + 1}: "
                f"In Fold {foldNumber}, "
                f"the correct sense was "
                f"{actualSense}, but the system "
                f"predicted {predictedSense}. "
                f"A possible reason is that the "
                f"context words had a higher combined "
                f"Naive Bayes probability for "
                f"{predictedSense} in the training data.\n"
            )

            answerFile.write(
                f"Context: {cleanContext}\n"
            )

        if numberOfErrors < 3:
            answerFile.write(
                "Fewer than three classification "
                "errors were produced.\n"
            )


# --------------------------------------------------
# Main program
# --------------------------------------------------

instances = readInstances(filename)

totalInstances = len(instances)

# Stop if the file was not parsed correctly
if totalInstances == 0:
    print(
        "Error: No annotated instances were found."
    )
    print(
        "Check that the input file contains "
        "<instance>, <context>, and </instance> tags."
    )
    sys.exit(1)


# Determine the possible sense labels
senseLabels = set()

for sense, context in instances:
    senseLabels.add(sense)


# Create the five folds
folds = createFolds(instances)


# Display dataset information
print("Total instances:", totalInstances)
print(
    "Possible sense labels:",
    len(senseLabels)
)
print("Sense labels:", senseLabels)
print()


# Display fold sizes
for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"{len(folds[i])} instances"
    )


# Run five-fold cross-validation
(
    foldAccuracies,
    errors,
    overallAccuracy
) = fivefoldCrossValidation(folds)


# Display three actual errors
print()
print("----------------------------------")
print("Three classification errors")
print("----------------------------------")

for i in range(min(3, len(errors))):
    (
        foldNumber,
        actualSense,
        predictedSense,
        context
    ) = errors[i]

    cleanContext = " ".join(context.split())

    print()
    print(f"Error {i + 1}")
    print("Fold:", foldNumber)
    print("Actual:", actualSense)
    print("Predicted:", predictedSense)
    print("Context:", cleanContext)


# Create the required answer file
createAnswersFile(
    foldAccuracies,
    errors,
    overallAccuracy
)

print()
print("Created WSD.answers")