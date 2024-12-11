# Interpreting Categorical Features on Recipes to Study and Make Predictions on Their Nutritional Components
## Introduction
### Research Question
#### This research investigates the accuracy of recipe information, focusing on textual features such as descriptions, ingredients, tags, names, as well as recipe ratings to assess their alignment with nutritional components. By analyzing these features, I aim to predict nutritional values without relying on the provided data. While classifier created in this report focuses on predicting sugar level, it provides helpful insights for development of the ultimate model that predicts multiple nutritional values (e.g., calories, fat, protein). This study not only evaluates the reliability of recipe claims but also explores patterns linking text features to nutrition, with implications for improving online recipe data and dietary recommendations.

### Dataset Information

#### The first dataset, `recipe` has a shape of (83,782,10). It contains 83,782 recipes, and 10 attributes of each recipe as described in the following:

| Column          | Description                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------|
| `name`          | Recipe name                                                                                      |
| `id`            | Recipe ID                                                                                        |
| `minutes`       | Minutes to prepare recipe                                                                        |
| `contributor_id`| User ID who submitted this recipe                                                                |
| `submitted`     | Date recipe was submitted                                                                        |
| `tags`          | Food.com tags for recipe                                                                         |
| `nutrition`     | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `n_steps`       | Number of steps in recipe                                                                        |
| `steps`         | Text for recipe steps, in order                                                                  |
| `description`   | User-provided description                                                                        |
| `ingredients`   | Text for recipe ingredients                                                                      |
| `n_ingredients` | Number of ingredients in recipe                                                                  |

---

#### The second dataset, `interactions` has a shape of (731,927,5). It contains 731,927 reviews on recipes recorded in the dataset above. Columns are described in the following:

| Column       | Description                       |
|--------------|-----------------------------------|
| `user_id`    | User ID                           |
| `recipe_id`  | Recipe ID                         |
| `date`       | Date of interaction               |
| `rating`     | Rating given                      |
| `review`     | Review text                       |

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning Steps

To make my analysis of the dataset more efficient and convenient, I conducted the following data cleaning steps:

---

**1. Left Merge the `recipes` and `interactions` Datasets on `id` and `recipe_id`.**  
This step matches unique recipes with their ratings from user interactions. Review column is dropped since it is lengthy and has little relevance to the project. By merging these datasets, I enriched the recipe dataset with user feedback, which is crucial for understanding recipe popularity and quality.

---

**2. Fill All Ratings of `0` with `np.nan`.**  
Ratings ranges from 1 to 5, where 1 indicates the lowest rating and 5 indicates the highest rating. A rating of `0` means the user didn't perform the rating action, therefore is invalid and signifies missing data.

---

**3. Add a Column `average_rating` Containing the Average Rating Per Recipe.**  
Since a recipe can receive multiple ratings from different users, I computed the average rating for each recipe. This provides a more comprehensive and representative understanding of the quality of a given recipe. Incorporating this feature aids in exploring correlations between recipe rating and its nutritional components.

---

**4. Split Values in the `nutrition` Column into Individual Columns of Floats.**  
Although the `nutrition` column appears as a list, it is actually stored as a string object. Based on the dataset documentation, I know the meaning of each value inside the brackets. By applying a lambda function, I split the values into individual columns (e.g., `calories`, `sugar (%)`) and converted them to floats. This is crucial for enabling numerical analysis and preparing the response variable (`sugar (%)`) for prediction.

---

Here are the head of the dataframe we used for our analysis. The cleaned dataframe has 83,782 rows and 12 columns. To keep the website clean, only columns used for the analysis are included. Note that `steps` is also a feature used in the prediction model, it is not shown here because it contains very long strings.

| name                                 | tags                                                                                                                                                                                                                                                                                               |   minutes | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                                                             | nutrition                                                   |   n_steps |   sugar (%) |   rating_avg |
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|----------:|------------:|-------------:|
| 1 brownies in the world    best ever | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        40 | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          | ['138.4', '10.0', '50.0', '3.0', '3.0', '19.0', '6.0']      |        10 |          50 |            4 |
| 1 in canada chocolate chip cookies   | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        45 | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             | ['595.1', '46.0', '211.0', '22.0', '13.0', '51.0', '26.0']  |        12 |         211 |            5 |
| 412 broccoli casserole               | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |        40 | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   | ['194.8', '20.0', '6.0', '32.0', '22.0', '36.0', '3.0']     |         6 |           6 |            5 |
| millionaire pound cake               | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |       120 | why a millionaire pound cake?  because it's super rich!  this scrumptious cake is the pride of an elderly belle from jackson, mississippi.  the recipe comes from "the glory of southern cooking" by james villas.                                                                                                                                                                | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                | ['878.3', '63.0', '326.0', '13.0', '20.0', '123.0', '39.0'] |         7 |         326 |            5 |
| 2000 meatloaf                        | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        90 | ready, set, cook! special edition contest entry: a mediterranean flavor inspired meatloaf dish. featuring: simply potatoes - shredded hash browns, egg, bacon, spinach, red bell pepper, and goat cheese.                                                                                                                                                                         | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] | ['267.0', '30.0', '12.0', '12.0', '29.0', '48.0', '2.0']    |        17 |          12 |            5 |


### Univariate Analysis

In this analysis, I plan to explore the distribution of average ratings for the recipes to understand how recipes are generally rated through the following aspects:

1. **Impact of Advertising Words on Ratings:**  
   Investigate whether the presence of advertising words in recipe names influences their ratings. This will help determine if promotional language affects user perception.
<iframe
 src="asset/ad_words_dist.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

#### Identified Advertising Words
The following advertising words were extracted from recipe names to analyze their impact on recipe popularity and ratings:

- **Advertising Words:**  
  `"top"`, `"great"`, `"perfect"`, `"ultimate"`, `"finest"`, `"supreme"`, `"premium"`,  
  `"delicious"`, `"amazing"`, `"favorite"`, `"choice"`, `"exceptional"`, `"outstanding"`,  
  `"fantastic"`, `"award-winning"`, `"signature"`, `"classic"`, `"legendary"`,  
  `"irresistible"`, `"heavenly"`, `"famous"`, `"perfected"`, `"all-time"`, `"best"`

#### Observations
- Recipes that include advertising words such as **"best"** or **"exceptional"** in their names tend to be **slightly more popular** compared to those without such words.
- Despite this, the **distribution of average ratings** between recipes with and without advertising words is **very similar**, indicating that promotional language does not significantly affect user ratings.


2. **Difference in Ratings Between Advertised and Healthy Recipes:**  
   Compare the ratings of recipes that advertise themselves versus those that claim to be healthy. This analysis will reveal whether users rate "healthy" recipes differently from "advertised" recipes.
<iframe
 src="asset/healthy_ad.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

#### Identified Healthy-Related Words
In addition to analyzing advertised recipes, I also identified recipe names containing terms related to health and wellness. These include:

- **Healthy-Related Words:**  
  `"healthy"`, `"fit"`, `"low-fat"`, `"low-carb"`, `"low-sugar"`, `"fat-free"`, `"sugar-free"`,  
  `"carb-free"`, `"gluten-free"`, `"dairy-free"`, `"plant-based"`, `"organic"`, `"heart-healthy"`,  
  `"anti-inflammatory"`, `"weight-loss"`, `"energy-boosting"`

#### Observations
- Recipes promoting themselves as **advertised** (e.g., with words like "best" or "exceptional") have **distinctively higher ratings** compared to recipes with healthy-related terms.
- Advertised recipes show a significantly higher frequency of **average ratings of 5**, whereas recipes with healthy-related terms are less likely to achieve this top rating.
- This suggests that while recipes labeled as healthy may appeal to a specific audience, they do not receive ratings as high as their advertised counterparts.

### Bivariate Analysis: Sugar Level and Numerical Variables

Since sugar level is selected as the prediction response variable, I analyzed the relationship between sugar level and some key numerical variables mentioned in the introduction. Identifying these correlations can assist in the feature engineering process for building a robust prediction model.

#### Numerical Variables Analyzed:
1. **Minutes (Cooking Time):** The total time required to prepare and cook a recipe.
2. **Number of Steps (`n_steps`):** The number of steps involved in the recipe instructions.
3. **Number of Ingredients (`n_ingredients`):** The total number of ingredients used in a recipe.

<iframe
 src="asset/minutes.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

- **Sugar Level vs. Minutes (Cooking Time):**  
  The plot shows a weak correlation between sugar level and cooking time, suggesting that sugar levels are relatively consistent regardless of preparation duration.

<iframe
 src="asset/n_steps.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

- **Sugar Level vs. Number of Steps (`n_steps`):**  
  Recipes with higher sugar levels tend to have slightly fewer steps. This could indicate that simpler recipes, such as desserts, are more likely to contain higher sugar levels.

<iframe
 src="asset/n_ingredients.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

- **Sugar Level vs. Number of Ingredients (`n_ingredients`):**  
  Recipes with fewer ingredients tend to have higher sugar levels. This aligns with the notion that simpler recipes, particularly desserts or snacks, often rely on sugar as a primary ingredient.

## Step 3: Assessment of Missingness
In the cleaned dataset, only three variables contain missing values:

- **Name:** 1 missing value  
- **Descriptions:** 70 missing values  
- **Average Rating (`average_rating`):** 2609 missing values  

#### Handling Missing Values
The number of missing values in `name` and `description` is insignificant compared to the overall dataset size and does not contribute significantly to the study's objectives. Therefore, the analysis focuses only on the missing values in `average_rating`.

#### NCMR
Given the only feature with missing values to analyze is average_rating, I propose to use this section to discuss how this feature could be characterized as *NMAR* and how it could also be potentially *MAR*. As described in the data cleaning section, ratings of 0 are converted to NaN because it indicates that the reviewers didn't give a rating to the recipe. Therefore, the missing of rating is caused by the feature itself that the reviewer didn't perform rating. However, at the same time, it could be argued that the reviewers refused to give rating because the recipe isn't great nor is it bad, as people tend to express whether they like or dislike something.

#### Analysis of Missing Feature
Again, given the only feature with missing values to analyze is average_rating, analysis is performed between average_rating and other columns. Permutation test results between calories, as well as, minutes based on missingness of average rating are shown as summarized in detail below. However, after performing permutaion tests on all numerical columns, there isn't a test result that fails to reject the hypothesis that missing of rating_avg depends on the column. Therefore test result of minute is shown as it had the highest p-value.

#### Permutation Test to Analyze Dependency of Missingness of `rating_avg` on `calories`

##### Null Hypothesis (H_0):
The missingness of `rating_avg` is independent of `calories`. In other words, there is no relationship between whether `rating_avg` is missing and the values of `calories`.

##### Alternative Hypothesis (H_a):
The missingness of `rating_avg` depends on `calories`. This implies that recipes with missing `rating_avg` have systematically different `calories` compared to those without missing values.

##### Test Statistic:
The test statistic is the difference in mean `calories` between mission and non-missing recipes

##### Significance Level (alpha):
The significance level is set to 0.05. 

##### Conclusion: 
The null hypothesis is rejected (p = 0.0), suggesting a statistically significant dependency between the missingness of rating_avg and minutes, based on the data collected.
<iframe
 src="asset/missing1.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

#### Permutation Test to Analyze Dependency of Missingness of `rating_avg` on `minutes`

##### Null Hypothesis (H_0):
The missingness of `rating_avg` is independent of `minutes`. In other words, there is no relationship between whether `rating_avg` is missing and the values of `minutes`.

##### Alternative Hypothesis (H_a):
The missingness of `rating_avg` depends on `minutes`. This implies that recipes with missing `rating_avg` have systematically different `minutes` compared to those without missing values.

##### Test Statistic:
The test statistic is the difference in mean `minutes` between mission and non-missing recipes

##### Significance Level (alpha):
The significance level is set to 0.05.

##### Conclusion:
The null hypothesis is rejected (p = 0.037), suggesting a statistically significant dependency between the missingness of `rating_avg` and `minutes`, based on the data collected.
<iframe
 src="asset/missing2.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

## Hypothesis testing
As shown earlier, we identified recipes with names or tags related to some "healthy" terms. I performed the following permutation test, to examine whether there is a true significant difference in the nutritional composistion between the two groups.

### Permutation Test to Compare Nutritional Composition of Recipes Claimed to be "Healthy" vs. Recipes not Claimed to be "Healthy" Recipes

#### Null Hypothesis (H_0):
The nutritional compositions of recipes claimed to be "healthy" are not significantly different from those of recipes not claimed to be "healthy." Any observed difference is due to random chance.

#### Alternative Hypothesis (H_a):
The nutritional compositions of recipes claimed to be "healthy" are significantly different from those of recipes not claimed to be "healthy."

#### Test Statistic:
The Total Variation Distance (TVD) between the normalized nutritional distributions of "healthy" and non-"healthy" recipes

#### Significance Level (\(\alpha\)):
The significance level is set to 0.05. If the p-value is less than or equal to 0.05, the null hypothesis is rejected, suggesting a significant difference in nutritional compositions.

#### Results:
- **Observed TVD**: 0.0299
- **P-value**: 0.0

#### Conclusion:
Based on the results:
- \(p < 0.05\): Reject the null hypothesis. Based on the dataset I have collected, recipes claimed to be "healthy" indeed have significantly different nutritional compositions compared to non-"healthy" recipes.

#### Justification for Choices:
1. **Test Statistic (TVD)**: TVD is an appropriate choice as it measures the overall difference between two probability distributions, providing a clear and interpretable metric for nutritional comparisons.
2. **Significance Level (alpha)**: A standard level of 0.05 balances the risk of Type I and Type II errors, ensuring the results are both robust and interpretable.
3. **Permutation Test**: This non-parametric approach makes no assumptions about the data's distribution, making it ideal for comparing groups with potentially non-normal or complex distributions.
<iframe
 src="asset/hp.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

## Prediction Problem
My overall objective for this study is to predict recipes' nutritional compositions without looking at given relavent data. However, for simplicity, I propose to develop a classifier that predicts the sugar level of a recipe.

**Prediction Problem:**  
The task is to predict the sugar level category (`Low`, `Moderate`, `High`) for a recipe based on its features without looking at the nutritional components. Features used in training the model includes textual features, such as ingredients, steps, description, and tags; as well as categorical numerical features such as minutes, rating average, n_steps, and n_ingredients. 

**Type:**  
This is a **multiclass classification** problem because the response variable (`sugar_category`) has three distinct categories (`Low`, `Moderate`, `High`).

---

### Response Variable
**Response Variable:**  
The response variable is `sugar_category`, which indicates the level of sugar in a recipe.  
**Reason for Choosing It:**  
Predicting the sugar category simplifies the problem by converting a continuous variable (`sugar %`) into interpretable categories based on scientifically defined thresholds. These thresholds align with health guidelines, such as those from the World Health Organization (WHO) and the Dietary Guidelines for Americans 2020–2025. 

**Categorization Approach:**  
- **Low Sugar:** Recipes with sugar levels ≤ 25 grams, corresponding to ≤5% of total daily calories for a 2,000-calorie diet.  
- **Moderate Sugar:** Recipes with sugar levels >25 grams and ≤50 grams, corresponding to >5% and ≤10% of total daily calories.  
- **High Sugar:** Recipes with sugar levels >50 grams, corresponding to >10% of total daily calories.  

This categorization was chosen to make predictions more actionable and meaningful for dietary planning, while also simplifying the challenge of predicting continuous data.

---

### Evaluation Metric
**Metric:**  
We use the **F1-score** to evaluate the model's performance.  

**Reason for Choosing F1-score:**  
The F1-score is chosen because it balances precision and recall, which is crucial in this context. The dataset may have imbalanced class distributions (e.g., more `Low` sugar recipes than `High`), and accuracy alone could mislead by favoring the majority class. The F1-score ensures that the classifier's performance is evaluated comprehensively across all categories, especially for minority classes.

## Baseline Model
### Model Explanation

The baseline model uses textual features `ingredients`, `steps` and numerical features `n_steps`, `n_ingredients`. 

---

### Textual Feature Engineering

**1. Text Cleaning:**  
The text data from `ingredients`, `steps` columns is preprocessed using a custom function:
- All text is converted to lowercase.
- Non-alphanumeric characters are removed.  
This ensures consistency and simplifies the analysis.

**2. Feature Selection with TF-IDF and Correlation:**  
- TF-IDF (Term Frequency-Inverse Document Frequency) is applied to the cleaned text to quantify word importance. This approach gives higher importance to unique and relevant words in the context of sugar prediction.
- The top features (words) that correlate most strongly with sugar levels are identified using correlation analysis (`f_regression`).

**3. Binary Feature Creation:**  
For the selected top words from each textual feature (e.g., `ingredients`, `steps`), binary columns are created to indicate whether a recipe contains each word. These binary features form part of the final dataset.

---

### Numerical Features
Two numerical features are included:
- **`n_steps`:** Number of steps in the recipe.
- **`n_ingredients`:** Number of ingredients used.

These features help capture recipe complexity, which may influence sugar levels.

---

### Model Pipeline
1. **Preprocessing:**  
   - Standardize numerical features using `StandardScaler`.
   - Pass binary features as they are.  
2. **Classification:**  
   A `RandomForestClassifier` is trained on the transformed data to predict sugar categories.

---

### Model Evaluation
The model evaluates performance using:
- **Accuracy:** 0.63
- **Classification Report:** Includes precision, recall, and F1-score for each sugar category, providing insights into model performance across all classes.

---

### Summary
           precision    recall  f1-score   support

    High       0.61      0.59      0.60      4887
     Low       0.68      0.81      0.74      8897
    Moderate   0.25      0.13      0.17      2973

    accuracy                       0.63     16757

##### The classifier achieves an overall accuracy of 63%. The `Low` sugar category has the highest precision and recall, making it the easiest class to predict accurately. The `Moderate` category has the lowest performance, likely due to fewer examples and overlap with other categories. This model combines textual and numerical features to predict sugar levels effectively, given that although textual features like ingredients do provide some indication to sugar level, there isn't direct reflections such as how much ingredients are used. Therefore I'm satisfied with the accuracy for the baseline model.

## Final Model
### Additional Features to Improve the Baseline Model

Following the same feature engineering procedure, I included the additional textual features `descriptions` and `tags`, as there is likely to be words with indication of sugar level existed in these two features such as sweet, or dessert.

---

### Hyperparameter Tuning to Improve the Baseline Model

To optimize the RandomForestClassifier, I used **GridSearchCV** to examine and choose the best hyperparameters. This process systematically evaluates combinations of parameters and selects the configuration that maximizes model performance. The goal was to improve the classifier's ability to predict sugar categories (`Low`, `Moderate`, `High`).

---

### Hyperparameter Search Space

The following hyperparameters were examined:
- **`n_estimators`**: Number of trees in the forest. Tested values: `[50, 100, 200]`.
- **`max_depth`**: Maximum depth of each tree. Tested values: `[None, 10, 20]`.
- **`min_samples_split`**: Minimum number of samples required to split an internal node. Tested values: `[2, 5, 10]`.
- **`min_samples_leaf`**: Minimum number of samples required to be at a leaf node. Tested values: `[1, 2, 4]`.

These parameters influence the complexity and generalization of the model. A broader range of values was tested to balance underfitting and overfitting.

---

### Procedure

1. **Train-Test Split:**  
   I split the dataset into training and testing sets. An additional split of the training data was used for hyperparameter optimization.
   
2. **Data Transformation:**  
   Both the training and testing datasets were preprocessed using a pipeline that:
   - Standardized numerical features (`rating_avg`, `n_steps`, `n_ingredients`).
   - Passed binary features unchanged (`contains_<word>` columns from ingredients, steps, description, and tags).

3. **GridSearchCV Setup:**  
   I used **GridSearchCV** with 5-fold cross-validation to evaluate combinations of hyperparameters, optimizing for **accuracy**.

4. **Model Training and Selection:**  
   The best-performing hyperparameters were selected based on the highest cross-validated accuracy.

---

### Results

The best parameters identified by GridSearchCV were:

***{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}***

               precision    recall  f1-score   support

        High       0.67      0.73      0.69      4887
         Low       0.71      0.90      0.79      8897
    Moderate       0.40      0.04      0.07      2973

    accuracy                           0.69     16757

##### Hyperparameter tuning via GridSearchCV significantly improved the model's ability to predict sugar categories, particularly for Low and High sugar recipes. In addition, however, with an increase in overall accuracy of 6%, the f1-score for moderate sugar level prediction has decreased by 10 percent. This may be due to overlap with Low and High categories or fewer samples in this class created in the second split. 

### Fairness Analysis

In this fairness analysis, I evaluated whether the model performs equally well across recipes categorized by their caloric content. Specifically, I examined if the precision for recipes with high calories is comparable to the precision for recipes with low calories.

---

#### Methodology

1. **Grouping Recipes by Caloric Content**:  
   - Recipes were divided into two groups: **high calorie** and **low calorie**.
   - Threshold: Recipes with calories > 301.1 were designated as high calorie, and those ≤ 301.1 were considered low calorie.  
   - The median value (301.1) was chosen as the threshold because the distribution of calories was skewed with many high outliers, making the median a more robust measure than the mean.

2. **Evaluation Metric - Precision Parity**:  
   - I chose precision as the metric for fairness evaluation to ensure that the model correctly identifies true positives (correctly labeled recipes) out of all predicted positives for each group.  
   - This is important because false positives could mislead users by incorrectly labeling recipes, especially for low-calorie recipes that may appeal to health-conscious individuals.

3. **Hypotheses**:
   - **Null Hypothesis:** The model is fair. Its precision for high-calorie and low-calorie recipes is approximately the same, and any observed differences are due to random chance.
   - **Alternative Hypothesis:** The model is unfair. Precision for low-calorie recipes is lower than for high-calorie recipes.

4. **Test Statistic**:  
   - The difference in precision

5. **Significance Level**:  
   - A significance level of alpha = 0.05 was used to evaluate the results.

6. **Permutation Test**:  
   - To test the hypotheses, I shuffled the `is_high_calories` labels 1000 times to simulate the distribution of precision differences under the null hypothesis.  
   - The observed statistic (-0.023) was compared to this distribution to calculate the p-value.
<iframe
 src="asset/fair.html"
 width="1000"
 height="600"
 frameborder="0"
></iframe>

---

#### Results

- **Observed Statistic:** The observed difference in precision was **-0.0285**.  
- **P-value:** After running the permutation test, the p-value was **0.118**.  
  - This p-value indicates that the observed difference could plausibly occur under the null hypothesis.

---

#### Conclusion

Based on the p-value of **0.055** (greater than 0.05), I fail to reject the null hypothesis. This suggests that the precision for high-calorie and low-calorie recipes is not significantly different, and the observed disparity is likely due to random chance. Thus, the model demonstrates fairness in terms of precision across recipes categorized by caloric content.
