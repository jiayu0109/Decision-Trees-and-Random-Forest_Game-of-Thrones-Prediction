# Decision-Trees-and-Random-Forest_Game-of-Thrones-Prediction
Data Science project: Decision Tree& Random Forest model with Scikit-learn
## I. Choice of the dataset — Game of Thrones Character Data
### 1. Reason:
I've always intrigued by long series of fantasy novels. The setting of the world is epic and the sharply different character images are shaped so vividly that I feel like it projects to the real world. What if we can predict the end of journey of their lives with certain features?
I came across the data on Kaggle and spot the possibility for me to do the prediction! It would be a cool and wonderful experience to practice decision tree algorithm practically with data I'm interested in.

### 2. Description of the dataset:
(1) What year is it from: This dataset is collected and uploaded on Kaggle on June, 2021.

(2) How was it collected: The source are extracted and organized from a fanmade wiki-site specifically for "A Song of Ice and Fire".
Source (from Kaggle):https://www.kaggle.com/dalmacyali1905/decision-tree-pruning-bagging-random-forest/data
Fanmade Wiki:https://awoiaf.westeros.org/index.php/Main_Page

(3) Notes: The owner of the dataset didn't specify each feature's definition, therefore in the next section, the information are based on my search from other similar sources that seems to be extracted from the same fanmade wiki site. Yet some of the definition is still unclear after the search.

### 3. Features:
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
### 1. What do I try to predict:
Many battles happen and deaths always come out of nowhere in Game of Thrones series. Sometimes we're tired of being anxious about whether our favorite roles would die when binge-watching Game of Throne series. Therefore I would like to try foreseeing their death or survival with personal data.

### 2. What do I try to accomplish:
We actually have a correct answer of whether people die in the end, so there're labels for each data. With these labeled inputs, first I'll have to make sure each variable is reasonable and efficient enough to train a model for classification and prediction, doing data cleaning if needed. I'll also try to optimize model's accuracy in the process of learning by adjusting parameters of the algorithm and variables of the original data. Hopefully the model's accuracy can reach around 90%.

### 3. Import and examine the data

## III. Choose an Algorithm: Decision Tree
### 1. Reason of choosing this algorithm:
Decision tree model is a supervised learning method for classification or regression. In this dataset, we have specific labels(supervised learning), and our goal is to predict "whether a character will die"(classify each character into different category-die or survive), which make this decision tree method appropriate for my research question.
Besides, this dataset contains both categorical data and real-value, which all can be internal nodes in the decision tree. Finally, decision tree model is good for interpretation and more intuitive for us to think about the process of model training. These are the reasons I chose decision tree algorithm.

### 2. Exploratory Data Analysis
Next I start to peak into each variable's relationship with "is.Alive"(which would be the dependent variable), or at least observe some trends. Then I'll adjust data and clean the data to make it suitable for decision tree model.

## IV. Build, run and evaluate the model
1-1. Build the Decision Tree model

1-2. Evaluate the Tree model with classification report and confusion matrix

2-1. Build the Random Forest model

2-2. Evaluate the Random Forest model with classification report and confusion matrix
- Adjust n_estimator=200, 500, 1000
- Try to improve model performance by adjusting variables

## V. After model evaluation...
### 1. Challenges I ran into:
(1) High portion of null/Nan values:
(Possible reason causing this challenge: dataset itself)
At first when I saw nan values take a big portion-- 13 of the variables have over 1000 null values, I'm afraid that this can't be solved and I have to either eliminate them all or give up on this dataset.
And then I cleaned the data by replacing them with proper values meaningful to our model.
- For "age" column, I filled them with the mean value.
- For other categorical values, I replaced them with "-1"

(2) Accuracy is not improved enough even after tinkering with parameters:
(Possible reason causing this challenge: the choice of parameter or dataset itself)
When I first built up decision tree model, the accuracy is only 78%. So I tried to overcome it with:
- building up random forest model
- changing n_estimator from 200 to 500 & 1000
- dropping different variables
However, accuracy isn't improved significantly (81% is the highest). I can only prove that n_estimator and the choice of variables matter, but still don't know how to find the best combination. I think the dataset can be improved by collecting other variables that have less Nan value.

### 2. Two potential benefits of my model:
(1) Find out the "popular formula" in entertainment industry:
If we can apply this model to other popular novels, TV series or movies, such as "Harry Potter", "Dune", "Lord of the rings", maybe we can know the formula the authors/screenwriters are using. We can learn that in order to keep audience addicted to their piece of work, does any simialr logic exist for them to arrange the characters' fate? Some young writers may refer to this if they aim to gain business success in this field.

(2) Promote data science to general audience:
I also think this model's build up can promote data science to general audience. Some people consider data science complicated and unfriendly to approach, but this decision tree model aims to predict character's death, which is easy to interpret and interesting to read. People may find out that data science is just around us and sometimes fun to explore!

### 3. Two potential harms of my model:
(1) Builder's subjective decision on building the model:
In the process of building the model, we can tell that from choosing variables to tinkering parameters, it's usually on the builder's choice to decide how the model eventually predicts the result. So it would be hard to judge whether the model is really fair and applicable.
If such model is expanded to predict some critical topics like whether a suspect is innocent or how many people will survive if cartain weapons are designed differently, it would be problematic.

(2) Low accuracy:
Another potential harm, of course, is my low accuracy. If I would like to apply this to a real-world problem, the results could make inaccurate prediction.

### 4. One research question I might ask next for future work:
If there's a battle-related data of each kingdom and results, I would like to explore:
### What negative impact a kindom would get if it actively start the war?
From novels/TV series, we have a clear result and correct labels to refer and make predictions. It would be a nice practice to examine this research question and recognize the harm/damage a real war would bring to our reality, especially in such chaotic time.








