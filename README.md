# CA02
What this project does

This notebook builds a simple spam classifier using Naive Bayes. It actually possesses TWO codes, the original with in-line notes, and the second being my own and revised version. 

It works in three big steps:

Build a dictionary of the 3000 most common “valid” words from the training emails

Convert each email into numbers (a row of word counts using that dictionary)

Train + test a Naive Bayes model and report accuracy

The result is an accuracy score showing how often the model correctly predicts spam vs not spam on the test set.

Folder setup (data expected)

You need two folders:

train-mails/ → emails used to learn patterns

test-mails/ → emails used to evaluate performance

In Google Colab (with Drive mounted), your paths usually look like:

TRAIN_DIR = "/content/drive/MyDrive/.../train-mails"
TEST_DIR  = "/content/drive/MyDrive/.../test-mails"


Label rule:

If a filename starts with "spmsg" → label = 1 (spam)

Otherwise → label = 0 (not spam)

How the code is organized
1) make_Dictionary(root_dir)

Purpose: Build the vocabulary of “important words” using the training emails only.

What it does:

Reads all emails in root_dir

Splits text into words

Counts word frequency using Counter

Removes:

non-alphabetic tokens (ex: 123, $, !!!)

1-letter tokens (ex: a, I)

Returns the top 3000 most common words as:

[(word1, count1), (word2, count2), ...]


Two required fixes we applied here:

Safer file reading

open(mail, "r", errors="ignore") prevents random crashes from weird characters.

Safer filtering

Instead of deleting words while looping, we rebuild a filtered Counter (more stable + clearer).

2) extract_features(mail_dir, dictionary, n_words=3000)

Purpose: Convert emails into a numeric feature matrix + labels.

Outputs:

features_matrix: shape (num_emails, 3000)

each row = one email

each column = count of that dictionary word in the email

labels: shape (num_emails,)

1 for spam, 0 for not spam (based on filename)

Why this version is fast (main improvement):

We create a word_to_idx lookup once (word → column index), so we don’t scan all 3000 dictionary entries for every word.

We use Counter to count each email’s words once, instead of repeatedly using words.count(...).

This reduces runtime from minutes to seconds.

3) Model training + testing (Gaussian Naive Bayes)

What happens:

Train the classifier:

model.fit(features_matrix, labels)

Predict on test emails:

model.predict(test_features_matrix)

Compute accuracy:

accuracy_score(test_labels, predicted_labels)

The notebook prints progress messages and the final accuracy score.

Expected output (example)

Your exact accuracy may vary by dataset, but the flow should look like:

reading and processing emails from TRAIN and TEST folders

Training Model using Gaussian Naive Bayes algorithm .....

Training completed

testing trained model to predict Test Data labels

Completed classification ... now printing Accuracy Score ...

<accuracy number>

================ END OF PROGRAM =================

Key assumptions (important)

Training and test folders are separate

Spam labeling is determined by filename prefix (spmsg…)

Dictionary is built only from training emails (avoids data leakage)

Words are split using .split() (simple whitespace tokenization)

Why the refactor mattered

The original feature extraction was slow because it:

scanned the entire 3000-word dictionary for every word

re-counted the same word repeatedly using words.count(word)

The refactor made it fast by:

using a hash map (word_to_idx) for instant word → column lookup

using Counter to count words once per email

Libraries used

os

numpy

collections.Counter

scikit-learn

GaussianNB

accuracy_score
