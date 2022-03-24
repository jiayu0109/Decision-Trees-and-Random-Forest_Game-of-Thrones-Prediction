# Decision-Trees-and-Random-Forest_Game-of-Thrones-Prediction
Data Science project: Decision Tree& Random Forest model with Scikit-learn
## I. Choice of the dataset — Game of Thrones Character Data
1. Reason:
I've always intrigued by long series of fantasy novels. The setting of the world is epic and the sharply different character images are shaped so vividly that I feel like it projects to the real world. What if we can predict the end of journey of their lives with certain features?
I came across the data on Kaggle and spot the possibility for me to do the prediction! It would be a cool and wonderful experience to practice decision tree algorithm practically with data I'm interested in.

2. Description of the dataset:
(1) What year is it from: This dataset is collected and uploaded on Kaggle on June, 2021.
(2) How was it collected: The source are extracted and organized from a fanmade wiki-site specifically for "A Song of Ice and Fire".
Source (from Kaggle):https://www.kaggle.com/dalmacyali1905/decision-tree-pruning-bagging-random-forest/data
Fanmade Wiki:https://awoiaf.westeros.org/index.php/Main_Page
(3) Notes: The owner of the dataset didn't specify each feature's definition, therefore in the next section, the information are based on my search from other similar sources that seems to be extracted from the same fanmade wiki site. Yet some of the definition is still unclear after the search.

3. Features:
- S.No: an incremental identifier for each character(like ID)
- plod: the probability that a character would die
- name: the character's name
- title: the character's title in the novel
- gender: the character's gender
- culture: which culture the character belongs to(if they have one)
- dateOfBirth: date of birth of each character
- DateoFdeath: date of death of each character
- mother: the character's mother (if mentioned in the novel)
- father: the character's father (if mentioned in the novel)
- heir: the character's heir (if mentioned in the novel)
- house: which house the character belongs to(if they have one)
- spouse: the character's spouse (if mentioned in the novel)
- book1: whether the character appears in the first book of the series (0=no, 1=yes)
- book2: whether the character appears in the second book of the series (0=no, 1=yes)
- book3: whether the character appears in the third book of the series (0=no, 1=yes)
- book4: whether the character appears in the fourth book of the series (0=no, 1=yes)
- book5: whether the character appears in the fifth book of the series (0=no, 1=yes)
- isAliveMother: whether the character's mother survives at the end (0=no, 1=yes)
- isAliveFather: whether the character's father survives at the end (0=no, 1=yes)
- isAliveHeir: whether the character's heir survives at the end (0=no, 1=yes)
- isAliveSpouse: whether the character's spouse survives at the end (0=no, 1=yes)
- isMarried: whether the character is married (0=no, 1=yes)
- isNoble: whether the character is from noble families in the book (0=no, 1=yes)
- age: each character's age
- numDeadRelations: number of dead relatives
- isPopular: 0 if the popularity of the character < 0.34, 1 if popularity>0.34
- popularity: (definition unclear) the popularity if the character (decided by fans)
- isAlive!: whether the character is alive at the end (0=no, 1=yes)

## II. Research Question:
Can we predict whether a character will die judging by his/her personal data and background?
1. What do I try to predict:
Many battles happen and deaths always come out of nowhere in Game of Thrones series. Sometimes we're tired of being anxious about whether our favorite roles would die when binge-watching Game of Throne series. Therefore I would like to try foreseeing their death or survival with personal data.

2. What do I try to accomplish:
We actually have a correct answer of whether people die in the end, so there're labels for each data. With these labeled inputs, first I'll have to make sure each variable is reasonable and efficient enough to train a model for classification and prediction, doing data cleaning if needed. I'll also try to optimize model's accuracy in the process of learning by adjusting parameters of the algorithm and variables of the original data. Hopefully the model's accuracy can reach around 90%.




