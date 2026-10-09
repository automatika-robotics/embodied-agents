# The answer of a decision model to one typed question about a state.

std_msgs/Header header

# The id of the question answered.
string id
# The kind of question: choice, score or noul (yes/no).
string type

# choice: the most likely option.
string choice
# score: the expected level index, which can fall between two levels.
float32 score
# noul: the probability that the answer is yes.
float32 noul
# From 0, all options equally likely, to 1.
float32 confidence

# The options the model chose between, parallel to `probabilities`:
# the option keys of a choice, the level descriptions of a score, and
# ["false", "true"] for a noul.
string[] options
float32[] probabilities
